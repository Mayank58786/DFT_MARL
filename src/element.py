class Event:
    def __init__(self, name, event_type, state=None):
        self.name = name
        self.event_type = event_type
        self.initial_state=state
        self.state = int(state)
        self.input = []
        self.old_state=int(state)
    
    
    def update_event(self):
        for event in self.input:
            event.update_event()
        if self.event_type != 'BASIC':
                if self.gate_type == 'AND': # If all inputs are 0, then state = 0, otherwise, state = 1
                    if sum([obj.state for obj in self.input]) == 0:
                        self.state = 0
                    else:
                        self.state = 1 
                elif self.gate_type == 'OR': # If any input is != 0, then state = 0, otherwise, state = 1
                    for obj in self.input:
                        if obj.state != 1:
                            self.state = 0
                            break  
                        else:
                            self.state = 1           
                elif self.gate_type == 'FDEP': # FDEP gates only accepts one input precedence, must combine with and or events prior to input signal.
                    self.state = self.input[0].state
                    #print(self.name,self.input[0].name)
                elif self.gate_type == 'CSP': # Cold Spare currently, only accepts 1 competitor, and 1 spare. Also the 'main' input must come first in the input list.
                    self.main_functioning = self.input[0].state # Is main functioning?
                    self.spare_functioning = self.spare.state # Is Spare functioning?     
                    if self.main_functioning == 1:
                        self.state = 1
                        self.using_spare = 0
                    elif self.main_functioning == 0 and self.competitor.using_spare == 0 and self.spare_functioning == 1:
                        self.state = 1
                        self.using_spare = 1
                    else:
                        self.state = 0
                        self.using_spare = 0

class BasicEvent(Event):
    def __init__(self, name, mttr=None, repair_cost=None, failure_cost=None, initial_state=1):
        super().__init__(name, event_type="BASIC", state=initial_state)
        self.mttr = int(mttr)
        self.repair_cost = int(repair_cost)
        self.failure_cost = int(failure_cost)


class IntermediateTopEvent(Event):
    def __init__(self, name,event_type, gate_type=None):
        super().__init__(name, event_type=event_type,state=0)
        self.gate_type=gate_type

class Precedence:
    def __init__(self, source, target, precedence_type, competitor=None):
        self.source = source
        self.target = target
        self.precedence_type = precedence_type
        self.competitor = competitor