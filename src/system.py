from src.element import Event
class System:
    def __init__(self):
        self.events = []
        self.precedences = []
        
    def add_event(self, event):
        self.events.append(event)

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
        for event in self.events:
            if event.event_type=="TOP":
                return event

    def initialize_tree(self):
        top_event=self.get_top_event()
        Event.update_event(top_event)
    
    def get_object(self,object):
        for obj in self.events:
            if obj.name == object:
                return obj
        print('Event not found')
        return None
    
    def apply_action(self, event, new_state):
        event.event_partial_update(new_state)
                