import random
from src.player import Player

class Game:
    def __init__(self,system,max_steps):
        self.system = system
        self.players=[]
        self.current_step=0
        self.max_steps = max_steps
        self.initial_resources = {}
        self.resources = {}
        self.actions = []
        self.costs = []
        self.observations = []
        self._set_actions()
        self._set_observations()
        self._set_action_costs()
        
    
    def reset_game(self):
        self.system = self.system.reset_system()
        self.current_step = 0
        self.resources = self.initial_resources
        for player in self.players:
            player.reset_player()
        
    def _set_actions(self):
        if not self.system.actions:
            self.system._set_actions()
        self.actions = self.system._get_actions()
        
    def _get_actions(self):
        return self.actions
    
    def _set_observations(self):
        if not self.system.observations:
            self.system._set_observations()
        self.observations = self.system._get_observations()
    
    def _get_observations(self):
        return self.observations
    
    def _set_action_costs(self):
        self.costs = self.system.get_action_costs()

    def _get_action_costs(self):
        return self.costs

    def get_system_obj(self):
        return self.system

    def get_current_step(self):
        return self.current_step
    
    def increase_step(self):
        self.current_step += 1

    def get_max_step(self):
        return self.max_steps
    
    def get_initial_resources(self):
        return self.initial_resources
    
    def get_resources(self):
        return self.resources

    def set_player_resource(self,agent,resource):
        self.resources[agent] = resource     
    
    def get_player_resources(self, agent):
        return self.resources[agent]

    def is_game_over(self):
        if self.current_step == self.max_steps:
            return True
        else:
            return False
    
    def create_player(self, name, resources):
        player = Player(name, self.actions)
        self.initial_resources[player] = resources
        self.resources[player] = resources
        self.players.append(player)
        return player
    
    def get_players(self):
        return self.players
    
    def apply_action(self, agent, action):
        if self.players[0].name == agent.name:
            self.system.apply_action(self.players[0].name,action)
            self.players[0].activate_action_mask(action)
            self.players[1].deactivate_action_mask(action)
        else:
            self.system.apply_action(self.players[1].name,action)
            self.players[1].activate_action_mask(action)
            self.players[0].deactivate_action_mask(action)
    
    def choose_action(self,agent):
        valid_actions=agent.valid_actions_mask["Active"]
        return valid_actions[random.randint(0,len(valid_actions) - 1)]
    
    