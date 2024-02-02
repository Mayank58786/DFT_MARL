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
        if precedence.precedence_type=='CSP':
            competitor=self.get_object(precedence.competitor)
            target.competitor=competitor
            target.spare=source
            target.using_spare=1
    
    def get_object(self,object):
        for obj in self.events:
            if obj.name == object:
                return obj
        print('Event not found')
        return None
    
    def update_states(self):
        for event in self.events:
            Event.update_event(event)
                