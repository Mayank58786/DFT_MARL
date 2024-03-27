import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from src.env_creator import env_creator
import matplotlib.pyplot as plt
import gym

# to store memory
class ReplayBuffer():
    def __init__(self, max_size, input_shape):
        #input shape = shape of observations
        self.mem_size = max_size
        self.mem_cntr = 0 # index of last stored memory
        #print("input shape",input_shape)
        self.state_memory = np.zeros((self.mem_size, input_shape),
                                     dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, input_shape),
                                     dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int64)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.uint8) # for done flags

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        #print(self.mem_cntr)
        #print(state_)
        
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        #print(done)
        self.action_memory[index] = action
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    # for usinform sampling of memory when agent learns
    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size) # so that non filled valyes zeros are not chosen
        batch = np.random.choice(max_mem, batch_size, replace = False)

        states = self.state_memory[batch]
        #print("states''''''''''''''",states)
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal

    
class Agent():
    def __init__(self, name,gamma, epsilon, lr, n_actions, input_dims,
                 mem_size, batch_size, eps_min=0.01, eps_dec=5e-7,
                 replace=1000, chkpt_dir='tmp/double_dqn'):
        # epsilon is the fraction of timeit spends on taking random actions
        # how often to replace agent network
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.replace_target_cnt = replace
        self.chkpt_dir = chkpt_dir
        self.learn_step_counter = 0
        self.action_space = [i for i in range(self.n_actions)]
        #print("input dimensions,,,,,,,,,,,,,,,,,,",input_dims)
        self.memory = ReplayBuffer(mem_size, len(input_dims))
        self.q_eval = DoubleDeepQNetwork(self.lr, self.n_actions,
                                         input_dims = self.input_dims,
                                         name=name+'DoubleDeepQNetwork_q_eval',
                                         chkpt_dir=self.chkpt_dir)
        self.q_next = DoubleDeepQNetwork(self.lr, self.n_actions,
                                         input_dims = self.input_dims,
                                         name=name+'DoubleDeepQNetwork_q_next',
                                         chkpt_dir=self.chkpt_dir)
        
        #random number less than epsilon it takes a random action
        # if the random number is greater than epsilon ot takes a greedy action

    def choose_action(self, observation,action_mask):
        if action_mask is not None:
            valid_actions = np.where(action_mask)[0]  # Get indices of valid actions
            if np.random.random() > self.epsilon:
                state = T.tensor([observation], dtype=T.float).to(self.q_eval.device)
                _, advantage = self.q_eval.forward(state)
                masked_advantage = advantage[:, valid_actions]  # Consider only valid actions
                action = valid_actions[T.argmax(masked_advantage).item()]
            else:
                action = np.random.choice(valid_actions)
        else:
            if np.random.random() > self.epsilon:
                state = T.tensor([observation], dtype=T.float).to(self.q_eval.device)
                _, advantage = self.q_eval.forward(state)
                action = T.argmax(advantage).item()
            else:
                action = np.random.choice(self.action_space)
        return action
    
    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    # to reduce epsilon over time
    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec\
        if self.epsilon > self.eps_min else self.eps_min

    def save_models(self):
        self.q_eval.save_checkpoint()
        self.q_next.save_checkpoint()

    def load_models(self):
        self.q_eval.load_checkpoint()
        self.q_next.load_checkpoint()

    def learn(self,i):
        # if the model hasn't filled the batch_size of memory
        if self.memory.mem_cntr < self.batch_size:
            return
        #print(i)
        self.q_eval.optimizer.zero_grad()
        self.replace_target_network()

        #sampling of memory
        state, action, reward, new_state, done = \
            self.memory.sample_buffer(self.batch_size)
        # print("state........................",state)
        states = T.tensor(state).to(self.q_eval.device)
        actions = T.tensor(action).to(self.q_eval.device)
        dones = T.tensor(done).bool().to(self.q_eval.device)
        rewards = T.tensor(reward).to(self.q_eval.device)
        states_ = T.tensor(new_state).to(self.q_eval.device)

        indices = np.arange(self.batch_size)
        #print("states........................",states)
        if(states.numel() != 0):
            V_s, A_s = self.q_eval.forward(states)
            V_s_, A_s_ = self.q_next.forward(states_)

            V_s_eval, A_s_eval = self.q_eval.forward(states_)

            q_pred = T.add(V_s,
                           (A_s - A_s.mean(dim=1, keepdim=True)))[indices, actions]
            q_next = T.add(V_s_,
                           (A_s_ - A_s_.mean(dim=1, keepdim=True)))
            q_eval = T.add(V_s_eval,
                           (A_s_eval - A_s_eval.mean(dim=1, keepdim=True)))
            max_actions = T.argmax(q_eval, dim=1)

            q_next[dones] = 0.0
            q_target = rewards + self.gamma*q_next[indices, max_actions]

            loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
            loss.backward()
            self.q_eval.optimizer.step()
            self.learn_step_counter += 1
            self.decrement_epsilon()


class DoubleDeepQNetwork(nn.Module):
    def __init__(self, lr, n_actions, name, input_dims, chkpt_dir):
        super(DoubleDeepQNetwork, self).__init__()
        self.chkpt_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.chkpt_dir, name)
        #input_dims = 
        #print("..............",len(input_dims))
        self.fcl = nn.Linear(len(input_dims), 512)
        self.V = nn.Linear(512,1) # the value
        self.A = nn.Linear(512, n_actions) #the advantage of actions, relative value of each action

        self.optimizer = optim.Adam(self.parameters(), lr= lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)


    def forward(self, state):
        #print(state[0].size())
        #print("state...................",state)
        flat1 = F.relu(self.fcl(state))
        V = self.V(flat1)
        A = self.A(flat1)

        return V, A
    
    def save_checkpoint(self):
        print('.......saving a checkpoint....')
        print(self.state_dict())
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self, file):
        print('... loading checkpoint ....')
        self.load_state_dict(T.load(self.checkpoint_file))

if __name__ == '__main__':
    env = env_creator.create_env()
    num_agents = len(env.possible_agents)
    num_actions = env.action_space.n
    observation_size = env.observation_space.shape
    observation = env.game.observations
    max_cycles = env.game.get_max_steps() + 4
    
    num_games = 1000
    load_checkpoint = False

    red_agent = Agent(name= "red_agent", gamma=0.99, epsilon=0.1, lr=5e-4,
                  input_dims=observation, n_actions=num_actions, mem_size=1000000, eps_min=0.01,
                  batch_size=32, eps_dec=1e-3, replace=100)
    blue_agent = Agent(name= "blue_agent", gamma=0.99, epsilon=0.1, lr=5e-4,
                  input_dims=observation, n_actions=num_actions, mem_size=1000000, eps_min=0.01,
                  batch_size=32, eps_dec=1e-3, replace=100)
    agents = {"red_agent":red_agent, "blue_agent":blue_agent}
    if load_checkpoint:
        for k,v in agents.items():
            v.load_models
    
    filename = 'DFT-DDQN.png'
    scores, eps_history = {},{}
    wins = {"red_agent":0, "blue_agent":0}
    for k in agents.keys():
        scores[k] = []
        eps_history[k] = []

    for i in range(num_games):
        print("episode ", i)
        done = False
        observation = env.reset()
        score = {}
        for k,v in agents.items():
            score[k] = 0
        agent_nn = "red_agent"
        while not done:
            observation, reward, termination, truncation, info = env.last()
            #print(observation)
            if termination or truncation:
                action = None
                break
            env_agent = env.agent_selection
            action_mask = env.game.get_mask(env_agent)
            action = agents[agent_nn].choose_action(observation,action_mask)
            new_observation, reward, termination, truncation, info = env.step(action)
            if termination or truncation:
                done = True
            score[agent_nn] += reward
            agents[agent_nn].store_transition(observation, action, reward, 
                                    new_observation, done)
            agents[agent_nn].learn(i)
            agent_nn = "blue_agent" if agent_nn == "red_agent" else "red_agent"
        for k in scores.keys():
            scores[k].append(score[k])
        if env.system.state == 1:
            wins["blue_agent"] += 1
        else:
            wins["red_agent"] += 1
    print(wins)
            

            
