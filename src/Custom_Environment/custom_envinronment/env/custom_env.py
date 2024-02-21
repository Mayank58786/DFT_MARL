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
        self.possible_agents = ["red_agent","blue_agent"]
        self.resources = resources
        self.render_mode = render_mode
        self.system_state = self.system.state
        self.NUM_ITERS = game.get_max_steps()
        self.done = False
        # self.event_to_index = {event.name: idx for idx, event in enumerate(self.system.events)}

        # self._action_spaces = {agent: Discrete(len(self.system.get_basicEvents())) for agent in self.possible_agents}
        # self._observation_spaces = {agent: Discrete(len(self.system.events)) for agent in self.possible_agents}
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
    
    #Reset basic events to initial state received from the xml file then update intermediate events
    def reset(self, seed=None, options=None):

        self.agents = self.possible_agents.copy()
        self.timestep = 0
        for event in self.system.events:
            if event.event_type=='BASIC':
                event.state=event.initial_state
        self.system.update_states()

    def step(self, actions):
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            self._was_dead_step(action)
            return
        agent = self.agent_selection
        for agent, action in actions.items():
            # Retrieve the event based on the index specified in the action
            event_index = action["event"]
            event_name = list(self.event_to_index.keys())[event_index]
            event = self.system.get_object(event_name)

            # Modify the state of the specified event based on the agent
            self.system.apply_action(event,agent)

        # Update the states of events and their descendants
        self.system.update_states()

        # Increment timestep after all agents have taken their actions
        self.timestep += 1

        # Return observations, rewards, done, and info (customize as needed)
        return self.observe(agent), {}, False, {}

    def render(self):
        pass

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]