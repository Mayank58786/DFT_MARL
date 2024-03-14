



import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from src.env_creator import env_creator

gpu = False
class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return np.array(self.states),\
                np.array(self.actions),\
                np.array(self.probs),\
                np.array(self.vals),\
                np.array(self.rewards),\
                np.array(self.dones),\
                batches

    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []

class ActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, alpha,
            fc1_dims=256, fc2_dims=256, chkpt_dir='tmp/ppo'):
        super(ActorNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_torch_ppo')
        self.actor = nn.Sequential(
                nn.Linear(input_dims, fc1_dims),
                nn.ReLU(),
                nn.Linear(fc1_dims, fc2_dims),
                nn.ReLU(),
                nn.Linear(fc2_dims, n_actions),
                nn.Softmax(dim=-1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
    #feedforward function taking single input(state)
    #calculating probs to get action
    def forward(self, state):

        dist = self.actor(state)
        dist = Categorical(dist)
        
        return dist

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))
    
class CriticNetwork(nn.Module):
    def __init__(self, input_dims, alpha, fc1_dims=256, fc2_dims=256,
            chkpt_dir='tmp/ppo'):
        super(CriticNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'critic_torch_ppo')
        self.critic = nn.Sequential(
                nn.Linear(input_dims, fc1_dims),
                nn.ReLU(),
                nn.Linear(fc1_dims, fc2_dims),
                nn.ReLU(),
                nn.Linear(fc2_dims, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        
        value = self.critic(state)

        return value

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))

class CategoricalMasked(Categorical):
    def __init__(self, probs=None, logits=None, validate_args=None, masks=[]):
        self.masks = masks
        if len(self.masks) == 0:
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)
        else:
            self.masks = masks.type(torch.BoolTensor).to(device)
            logits = torch.where(self.masks, logits, torch.tensor(-1e+8).to(device))
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)
    
    def entropy(self):
        if len(self.masks) == 0:
            return super(CategoricalMasked, self).entropy()
        p_log_p = self.logits * self.probs
        p_log_p = torch.where(self.masks, p_log_p, torch.tensor(0.).to(device))
        return -p_log_p.sum(-1)

class Agent(nn.Module):
    def __init__(self, env, num_actions, num_states, actor_alpha, critic_alpha):
        super().__init__()
        self.device = torch.device("cuda" if (torch.cuda.is_available() and gpu) else "cpu")
        self.env = env
        self.actor = ActorNetwork(n_actions=num_actions, input_dims=num_states, alpha=actor_alpha)
        self.critic = CriticNetwork(input_dims=num_states, alpha= critic_alpha)

    # def _layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
    #     torch.nn.init.orthogonal_(layer.weight, std)
    #     torch.nn.init.constant_(layer.bias, bias_const)
    #     return layer

    def get_value(self, x):
        return self.critic.critic(x / 255.0)

    def get_action_and_value(self, x, action=None, invalid_action_masks=None):
        logits = self.actor.actor(x / 255.0)
        split_logits = torch.split(logits,  1)        
        if invalid_action_masks is not None:
            # print("Not None")
            split_invalid_action_masks = torch.split(invalid_action_masks, 1)
            multi_categoricals = [CategoricalMasked(logits=logits, masks=iam) for (logits, iam) in zip(split_logits, split_invalid_action_masks)]
        else:
            # print("None")
            multi_categoricals = [Categorical(logits=logits) for logits in split_logits]
       
        if action is None:
            action = torch.stack([categorical.sample() for categorical in multi_categoricals])
        logprob = torch.stack([categorical.log_prob(a) for a, categorical in zip(action, multi_categoricals)])
        entropy = torch.stack([categorical.entropy() for categorical in multi_categoricals])
        return action, logprob.sum(0), entropy.sum(0), self.critic.critic(x / 255.0)
    #def learn(self):


def batchify_obs(obs, device):
    """Converts PZ style observations to batch of torch arrays."""
    # convert to list of np arrays
    obs = np.stack([obs[a] for a in obs], axis=0)
    # transpose to be (batch, channel, height, width)
    obs = obs.transpose(0, -1, 1, 2)
    # convert to torch
    obs = torch.tensor(obs).to(device)

    return obs


def batchify(x, device):
    """Converts PZ style returns to batch of torch arrays."""
    # convert to list of np arrays
    x = np.stack([x[a] for a in x], axis=0)
    # convert to torch
    x = torch.tensor(x).to(device)

    return x


def unbatchify(x, env):
    """Converts np array to PZ style arguments."""
    x = x.cpu().numpy()
    x = {a: x[i] for i, a in enumerate(env.possible_agents)}

    return x


if __name__ == "__main__":
    """ALGO PARAMS"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ent_coef = 0.1
    vf_coef = 0.1
    clip_coef = 0.1
    gamma = 0.99
    batch_size = 32
    stack_size = 4
    frame_size = (64, 64)
    max_cycles = 125
    total_episodes = 2

    """ ENV SETUP """
    env = env_creator.create_env()
    num_agents = len(env.possible_agents)
    num_actions = env.action_space(env.possible_agents[0]).n
    observation_size = env.observation_space(env.possible_agents[0]).n
    max_cycles = env.game.get_max_steps() + 3
    depth = len(env.game.get_observations())

    # """ LEARNER SETUP """
    red_agent = Agent(env, num_actions, observation_size, 0.2, 0.2).to(device)
    optimizer = optim.Adam(red_agent.parameters(), lr=0.001, eps=1e-5)

    blue_agent = Agent(env, num_actions, observation_size, 0.2, 0.2).to(device)
    optimizer = optim.Adam(blue_agent.parameters(), lr=0.001, eps=1e-5)
    agents = [red_agent, blue_agent]
    """ ALGO LOGIC: EPISODE STORAGE"""
    end_step = 0
    total_episodic_return = 0

    rb_obs = {}
    rb_actions = {}
    rb_logprobs = {}
    rb_rewards = {}
    rb_terms = {}
    rb_values = {}
    rb_invalid_action_masks = {}

    for i in range(0,len(agents)):
        rb_obs[i] = torch.zeros((max_cycles, num_agents, stack_size, *frame_size)).to(device)
        rb_actions[i] = torch.zeros((max_cycles, num_agents)).to(device)
        rb_logprobs[i] = torch.zeros((max_cycles, num_agents)).to(device)
        rb_rewards[i] = torch.zeros((max_cycles, num_agents)).to(device)
        rb_terms[i] = torch.zeros((max_cycles, num_agents)).to(device)
        rb_values[i] = torch.zeros((max_cycles, num_agents)).to(device)
    #print(rb_obs,rb_actions,rb_logprobs,rb_rewards,rb_terms,rb_values)

    """ TRAINING LOGIC """
    # train for n number of episodes
    actions = {}
    logprobs= {}
    values  = {}
    rewards = {}
    terminations = {}
    invalid_action_masks = {}
    print(red_agent.get_action_and_value())
    # for agent in env.agents:
    #     observation, reward, termination, truncation, info = env.last()
    #     if termination or truncation:
    #         action = None
    #     else:
    #         #invalid_action_masks[agent.get_player_strategy_name()][agent] = env.game.get_mask(agent)
    #         #obs = batchify_obs(observation, agent, device)
    #         #action_mask = batchify(invalid_action_masks[agent.get_player_strategy_name()][agent],device)
    #         #print(observation)
    #         invalid_action_masks[agent.name] = 
    #         obs = batchify_obs(observation, agent, device)
    #         action_mask = {agent:env.game.get_mask(agent)}
    #         action_mask = batchify(action_mask,device)
    #         action, logprob, _, value = red_agent.get_action_and_value(obs)
    #         action = unbatchify(action)[0]
    #         logprob = unbatchify(logprob)
    #         value = unbatchify(value)
    #         actions[agent.get_player_strategy_name()][agent] = action
    #         logprobs[agent.get_player_strategy_name()][agent] = logprob
    #         values[agent.get_player_strategy_name()][agent] = value
    # env.step(action)
