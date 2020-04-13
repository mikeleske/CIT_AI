#
# IrishMan
#
# The example is a modified version of the BlindDog example from the aima-python repository.
# It simulates an agent the roams through TempleBar.
# 
# The agent simulates the following <percent, action> pairs:
#    PERCEPT         ACTION
#    -----------------------------
#    Song            sing
#    Guinness        drink
#    Shoes           dance
#    Fiddle          play
#    Bed             sleep
#
# The simlation is supposed to end when the agent finds a bed and goes sleep.
# However, the agent want to have a joyful lift and only accepts the Bed percept, when the beer_threshold is crossed.
#


import sys
sys.path.append('..')

from agents import *

class Song(Thing):
    pass

class Guinness(Thing):
    pass

class Shoes(Thing):
    pass

class Fiddle(Thing):
    pass

class Bed(Thing):
    pass

class TempleBar(Environment):
    def percept(self, agent):
        '''return a list of things that are in our agent's location'''
        things = self.list_things_at(agent.location)
        return things
    
    def execute_action(self, agent, action):
        '''changes the state of the environment based on what the agent does.'''
                
        if action == "move down":
            print('{} decided to {} at location: {}'.format(str(agent)[1:-1], action, agent.location))
            agent.movedown()
        elif action == "sing":
            items = self.list_things_at(agent.location, tclass=Song)
            if len(items) != 0:
                if agent.sing(items[0]): #Have the agent sing the first item
                    print('{} sang into a {} at location {} >> And so Sally can wait, she knows it\'s too late as we\'re walking on by.... <<'
                          .format(str(agent)[1:-1], str(items[0])[1:-1], agent.location))
                    self.delete_thing(items[0]) #Delete it from the TempleBar after.
        elif action == "drink":
            items = self.list_things_at(agent.location, tclass=Guinness)
            if len(items) != 0:
                if agent.drink(items[0]): #Have the agent drink the first item
                    print('{} drank {} at location {} >> So tasty tasty ...'
                          .format(str(agent)[1:-1], str(items[0])[1:-1], agent.location))
                    self.delete_thing(items[0]) #Delete it from the TempleBar after.
        elif action == "dance":
            items = self.list_things_at(agent.location, tclass=Shoes)
            if len(items) != 0:
                if agent.dance(items[0]): #Have the agent dance the first item
                    print('{} danced with {} at location {} >> clack, clack, clack...'
                          .format(str(agent)[1:-1], str(items[0])[1:-1], agent.location))
                    self.delete_thing(items[0]) #Delete it from the TempleBar after.
        elif action == "play":
            items = self.list_things_at(agent.location, tclass=Fiddle)
            if len(items) != 0:
                if agent.play(items[0]): #Have the agent play the first item
                    print('{} played the {} at location {} >> ..-+***+~++-... '
                          .format(str(agent)[1:-1], str(items[0])[1:-1], agent.location))
                    self.delete_thing(items[0]) #Delete it from the TempleBar after.
        elif action == "sleep":
            items = self.list_things_at(agent.location, tclass=Bed)
            if len(items) != 0:
                if agent.sleep(items[0]): #Have the agent play the first item
                    print('{} found a {} at location {} and sleeeeps >> ...zzzzzz..... '
                          .format(str(agent)[1:-1], str(items[0])[1:-1], agent.location))
                    self.delete_thing(items[0]) #Delete it from the TempleBar after.
                else:
                    print('{} rejects the {} at location {} and wants to have more fun '
                          .format(str(agent)[1:-1], str(items[0])[1:-1], agent.location))
                    self.delete_thing(items[0]) #Delete it from the TempleBar after.
                    
    def is_done(self):
        '''By default, we're done when we can't find a live agent, 
        but we also send our IrishMan to sleep when the beer_threshold is crossed'''
        dead_agents = not any(agent.is_alive() for agent in self.agents)
        sleeping_agents = all(agent.is_sleeping() for agent in self.agents)
        return dead_agents or sleeping_agents

class IrishMan(Agent):
    location = 1
    beer_threshold = 0
    beers = 0
    sleeping = False
    
    def movedown(self):
        self.location += 1
        
    def sing(self, thing):
        '''returns True upon success or False otherwise'''
        if isinstance(thing, Song):
            return True
        return False
    
    def drink(self, thing):
        ''' returns True upon success or False otherwise'''
        if isinstance(thing, Guinness):
            self.beers += 1
            return True
        return False
    
    def dance(self, thing):
        '''returns True upon success or False otherwise'''
        if isinstance(thing, Shoes):
            return True
        return False
    
    def play(self, thing):
        ''' returns True upon success or False otherwise'''
        if isinstance(thing, Fiddle):
            return True
        return False

    def sleep(self, thing):
        ''' returns True upon success or False otherwise'''
        if self.beers >= self.beer_threshold:
            self.sleeping = True
            if isinstance(thing, Bed):
                return True
            return False
    
    def is_sleeping(self):
        return self.sleeping
    
    def set_threshold(self, threshold):
        self.beer_threshold = threshold

def program(percepts):
    '''Returns an action based on the IrishMan's percepts'''
    for p in percepts:
        if isinstance(p, Song):
            return 'sing'
        elif isinstance(p, Guinness):
            return 'drink'
        elif isinstance(p, Shoes):
            return 'dance'
        elif isinstance(p, Fiddle):
            return 'play'
        elif isinstance(p, Bed):
            return 'sleep'
    return 'move down'

tp = TempleBar()
man = IrishMan(program)
man.set_threshold(3)

tp.add_thing(man, 1)
tp.add_thing(Guinness(), 2)
tp.add_thing(Guinness(), 4)
tp.add_thing(Bed(), 5)
tp.add_thing(Guinness(), 7)
tp.add_thing(Shoes(), 10)
tp.add_thing(Song(), 12)
tp.add_thing(Bed(), 14)
tp.add_thing(Guinness(), 15)
tp.add_thing(Fiddle(), 18)
tp.add_thing(Bed(), 20)

tp.run(50)