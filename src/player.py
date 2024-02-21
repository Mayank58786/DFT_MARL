class Player:
    def __init__(self, name, actions):
        self.name = name
        self.valid_actions = actions
        self.valid_actions_mask = {}
        self.score = 0
        self.took_invalid_action = False
        self.reset_masks()
    
    def increase_score(self):
        self.score = self.score + 1
    
    def get_valid_actions(self): 
        return self.valid_actions
    
    def get_num_valid_actions(self): 
        return len(self.valid_actions)
    
    def get_score(self):
        return self.score
    
    def reset_player(self):
        self.score = 0
        self.reset_masks()
        
    
    def activate_action_mask(self, action):
        self.valid_actions_mask["Active"].remove(action)
        self.valid_actions_mask["Not_Active"].append(action)

    def deactivate_action_mask(self, action):
        self.valid_actions_mask["Not_Active"].remove(action)
        self.valid_actions_mask["Active"].append(action)        
    
    def reset_masks(self):
        if self.name == "red_agent":
            self.valid_actions_mask = {"Active": list(self.valid_actions), "Not_Active": []}
        else:
            self.valid_actions_mask = {"Active": self.valid_actions[0:1], "Not_Active": list(self.valid_actions[1:])}
    
    



        