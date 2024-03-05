from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
from gymnasium.spaces import Discrete, MultiBinary
import numpy as np
class CustomEnvironment(AECEnv):
    metadata = {
        "name": "custom_environment_v0",
    }

    def __init__(self, system, game, resources, render_mode = None):
        super().__init__()
        self.reward_range = (-np.inf, np.inf)
        self.system = system
        self.game = game
        self.timestep = 0
        self.resources = game.get_resources()
        self.render_mode = render_mode
        self.system_state = self.system.state
        self.NUM_ITERS = game.get_max_steps()
        self.done = False
        
        # Agents
        self.agents = game.get_players()
        self.possible_agents = self.agents.copy()
        self.agent_name_mapping = dict(zip(self.possible_agents, list(range(len(self.possible_agents)))))
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()  

        # Spaces
        self.action_spaces = {agent: Discrete(len(self.system.get_actions())) for agent in self.possible_agents}                      
        self.observation_spaces = {agent: MultiBinary(1) for agent in self.possible_agents} 
        self.observations = {agent: self.system.get_observations() for agent in self.agents}

        # Rewards
        self.rewards = {agent : 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}                                                                                                       
        self.infos = {agent: dict() for agent in self.agents}                                                                 
        self.terminations = {agent: False for agent in self.agents}                                                                    
        self.truncations = {agent: False for agent in self.agents} 
    
    #Reset basic events to initial state received from the xml file then update intermediate events
    def reset(self, seed=None, options=None):

        self.agents = self.possible_agents.copy()
        self.resources = self.initial_resources
        self.timestep = 0
        self.game.reset_game()
        self.system.reset_system()
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.system_state = self.system.state
        self.infos = {agent: {} for agent in self.agents}
        self.action_spaces = {agent: Discrete(len(self.system.get_actions())) for agent in self.possible_agents}                      
        self.observation_spaces = {agent: MultiBinary(1) for agent in self.possible_agents} 
        self.observations = {agent: self.system.get_observations() for agent in self.agents}  
        if self.render_mode == "human":
            self.render()
        return self.observations

    def step(self):
        to_be_deleted=[]
        red_agent, blue_agent = (self.game.players[0], self.game.players[1]) if self.game.players[0].name == "red_agent" else (self.game.players[1], self.game.players[0])
        for action in self.system.repairing_dict.keys():
            action_event = self.system.get_object(action)
            action_event.remaining_time_to_repair -=1
            if action_event.remaining_time_to_repair == 0:
                for event_name in self.system.repairing_dict[action]:
                    event = self.system.get_object(event_name)
                    event.state = 1
                to_be_deleted.append(action)
                red_agent.deactivate_action_mask(action)
        for action in to_be_deleted:
            del self.system.repairing_dict[action]
            
        
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            self._was_dead_step(action)
            return
        
        # Selects agent for this step
        agent = self.agent_selection

        action, cost = self.game.choose_action(agent)
        print(action, cost)
        count = 0
        if cost > self.resources[agent]:
            count = self.game.apply_action(agent,'No Action')
            count = 0
        else:
            self.resources[agent] -= cost
            count = self.game.apply_action(agent, action)
        self.rewards[agent] += 1
        self.rewards[agent] += count
        #no reactive repairing instead preventive maintainance
        if agent.name == "red_agent":        
            if self.system.state == 1:
                self.rewards[agent] -= 1
            else:
                self.rewards[agent] += 10000
                # Update termination – game is "won"
                self.terminations = {agent: True for agent in self.agents}
        else:
            if self.system.state == 1:
               self.rewards[agent] += 1
            

        # Increment timestep after all agents have taken their actions
        self.timestep += 1
        # Update truncation - time is over.
        if self.timestep > self.NUM_ITERS:        
            self.truncations = {agent: True for agent in self.agents}
        self.observations = {agent: self.system.observe() for agent in self.agents}
        # Infos
        self.infos = {"agent_red": {}, "agent_blue": {}}

        # DEBUGGING!
        #self.infos = {"agent_red": {'time' : self.timestep, 'state' : self.system_state}, "agent_blue": {'time' : self.timestep, 'state' : self.system_state}}
        self.agent_selection = self._agent_selector.next()
        # Return observations, rewards, done, and info (customize as needed)
        return self.observations, self.rewards, self.done, self.infos

    def render(self):
        pass

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]
    
    def observe(self, agent):
        return np.array(self.observations[agent])
    