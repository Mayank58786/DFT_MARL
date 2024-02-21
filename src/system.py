from src.element import Event, No_Action
class System:
    def __init__(self):
        self.events = {}
        self.precedences = []
        self.state = 0
        self.actions = []
        self.observations = []
        self.repairing_list = []

    def add_event(self, event):
        self.events[event.name] = event

    def add_precedence(self, precedence):
        self.precedences.append(precedence)
        source=self.get_object(precedence.source)
        target=self.get_object(precedence.target)
        target.input.append(source)
        source.output.append(target)
        if precedence.precedence_type=='CSP':
            competitor=self.get_object(precedence.competitor)
            target.competitor=competitor
            target.spare=source
            target.using_spare=1
    
    def get_top_event(self):
        for event in self.events.values():
            if event.event_type=="TOP":
                return event
    
    def reset_system(self):
        for event in self.events.values():
            if event.event_type=='BASIC':
                event.state=event.initial_state
        Event.update_event(self.top_event)
        self.state = self.top_event.state
        self.repairing_list = []
        return self
        

    def initialize_system(self):
        self.top_event=self.get_top_event()
        self.reset_system()
        Event.update_event(self.top_event)
        
        self._set_actions()
        self._set_observations()

    def set_actions(self):
        no_action = No_Action('No Action')
        actions = [no_action]
        for event in self.get_basicEvents():
            actions.append(event.name)
        self.actions = actions
    
    def get_actions(self):
        return self.actions
    
    def set_observations(self):
        self.observations = list(self.events.keys())
        
    
    def get_observations(self):
        return self.observations

    def get_object(self,object):
        return self.events.get(object, None)
    
    def num_actions(self):
        if not self.actions:
            return len(self._set_actions())
        else:
            return len(self.actions)
    
    def num_observations(self):
        if not self.observations:
            return len(self._set_observations())
        else:
            return len(self.observations)

    def apply_action(self, agent, action):
        #print(stratergy_type)
        event = self.get_object(action)
        if agent == 'red_agent':
            event.red_action()
        elif agent == 'blue_agent':
            event.blue_action()
            self.repairing_list.append(action)

    def get_events(self):
        return self.events
    
    def get_basicEvents(self):
        return [event for event in self.events.values() if event.event_type == "BASIC"]
                
    def get_action_costs(self):
        costs = [0]
        for event in self.get_basicEvents():
            costs.append(int(event.failure_cost))
        return costs
    
    def get_system_state(self):
        self.state = self.get_top_event().state
        return self.state