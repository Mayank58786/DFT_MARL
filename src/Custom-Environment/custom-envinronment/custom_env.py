from pettingzoo import ParallelEnv


class CustomEnvironment(ParallelEnv):
    metadata = {
        "name": "custom_environment_v0",
    }

    def __init__(self,system):
        self.system= system
        self.timestep = None
        self.possible_agents = ["red","blue"]
    #Reset basic events to initial state received from the xml file then update intermediate events
    def reset(self, seed=None, options=None):

        self.agents = self.possible_agents.copy()
        self.timestep = 0
        for event in self.system.events:
            if event.event_type=='BASIC':
                event.state=event.initial_state
        self.system.update_states()

    def step(self, actions):
        pass

    def render(self):
        pass

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]