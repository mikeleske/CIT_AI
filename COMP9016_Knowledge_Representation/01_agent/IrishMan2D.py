#
# IrishMan2D
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
# In this 2D environment that agent not only moves linearly, but a level of randomness drives his movements.
#    FACING EDGE         NOT FACING EDGE
#    -----------------------------------------
#    50% Turn Left       25% Turn Left
#    50% Turn Right      25% Turn Right
#                        50% move Forward
#
# The simlation is supposed to end when the agent finds a bed and goes sleep.
# However, the agent want to have a joyful live and only accepts the Bed percept, when the beer_threshold is crossed.
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

from random import choice

class IrishMan2D(Agent):
    location = [0,1]
    direction = Direction("down")
        
    beer_threshold = 0
    beers = 0
    sleeping = False
    
    def moveforward(self, success=True):
        '''moveforward possible only if success (i.e. valid destination location)'''
        if not success:
            return
        if self.direction.direction == Direction.R:
            self.location[0] += 1
        elif self.direction.direction == Direction.L:
            self.location[0] -= 1
        elif self.direction.direction == Direction.D:
            self.location[1] += 1
        elif self.direction.direction == Direction.U:
            self.location[1] -= 1
    
    def turn(self, d):
        self.direction = self.direction + d
        
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
        if isinstance(p,Bump): # then check if you are at an edge and have to turn
            turn = False
            choice = random.choice((1,2));
        else:
            choice = random.choice((1,2,3,4)) # 1-right, 2-left, others-forward
    if choice == 1:
        return 'turnright'
    elif choice == 2:
        return 'turnleft'
    else:
        return 'moveforward'

class TempleBar2D(GraphicEnvironment):
    def percept(self, agent):
        '''return a list of things that are in our agent's location'''
        things = self.list_things_at(agent.location)
        loc = copy.deepcopy(agent.location) # find out the target location
        #Check if agent is about to bump into a wall
        if agent.direction.direction == Direction.R:
            loc[0] += 1
        elif agent.direction.direction == Direction.L:
            loc[0] -= 1
        elif agent.direction.direction == Direction.D:
            loc[1] += 1
        elif agent.direction.direction == Direction.U:
            loc[1] -= 1
        if not self.is_inbounds(loc):
            things.append(Bump())
        return things
    
    def execute_action(self, agent, action):
        '''changes the state of the environment based on what the agent does.'''
        if action == 'turnright':
            print('{} decided to {} at location: {}'.format(str(agent)[1:-1], action, agent.location))
            agent.turn(Direction.R)
        elif action == 'turnleft':
            print('{} decided to {} at location: {}'.format(str(agent)[1:-1], action, agent.location))
            agent.turn(Direction.L)
        elif action == 'moveforward':
            print('{} decided to move {}wards at location: {}'.format(str(agent)[1:-1], agent.direction.direction, agent.location))
            agent.moveforward()
        elif action == "sing":
            items = self.list_things_at(agent.location, tclass=Song)
            if len(items) != 0:
                if agent.sing(items[0]):
                    print('{} sang into a {} at location {} >> And so Sally can wait, she knows it\'s too late as we\'re walking on by.... <<'
                          .format(str(agent)[1:-1], str(items[0])[1:-1], agent.location))
                    self.delete_thing(items[0])
        elif action == "drink":
            items = self.list_things_at(agent.location, tclass=Guinness)
            if len(items) != 0:
                if agent.drink(items[0]):
                    print('{} drank {} at location {} >> So tasty tasty ...'
                          .format(str(agent)[1:-1], str(items[0])[1:-1], agent.location))
                    self.delete_thing(items[0])
        elif action == "dance":
            items = self.list_things_at(agent.location, tclass=Shoes)
            if len(items) != 0:
                if agent.dance(items[0]): 
                    print('{} danced with {} at location {} >> clack, clack, clack...'
                          .format(str(agent)[1:-1], str(items[0])[1:-1], agent.location))
                    self.delete_thing(items[0])
        elif action == "play":
            items = self.list_things_at(agent.location, tclass=Fiddle)
            if len(items) != 0:
                if agent.play(items[0]):
                    print('{} played the {} at location {} >> ..-+***+~++-... '
                          .format(str(agent)[1:-1], str(items[0])[1:-1], agent.location))
                    self.delete_thing(items[0])
        elif action == "sleep":
            items = self.list_things_at(agent.location, tclass=Bed)
            if len(items) != 0:
                if agent.sleep(items[0]):
                    print('{} found a {} at location {} and sleeeeps >> ...zzzzzz..... '
                          .format(str(agent)[1:-1], str(items[0])[1:-1], agent.location))
                    self.delete_thing(items[0])
                else:
                    print('{} rejects the {} at location {} and wants to have more fun '
                          .format(str(agent)[1:-1], str(items[0])[1:-1], agent.location))
                    self.delete_thing(items[0]) 
                    
    def is_done(self):
        '''By default, we're done when we can't find a live agent, 
        but we also send our IrishMan to sleep when the beer_threshold is crossed'''
        dead_agents = not any(agent.is_alive() for agent in self.agents)
        sleeping_agents = all(agent.is_sleeping() for agent in self.agents)
        return dead_agents or sleeping_agents

tp = TempleBar2D(5,5, color=
               {'IrishMan2D': (200,0,0), 
                'Guinness': (0, 200, 200), 
                'Shoes': (230, 115, 40),
                'Song': (100, 200, 100),
                'Fiddle': (230, 0, 200),
                'Bed': (0, 0, 0),
               }
) # TempleBar width is set to 5, and height to 5
man = IrishMan2D(program)
man.set_threshold(3)

tp.add_thing(man, [0,0])
tp.add_thing(Guinness(), [1,1])
tp.add_thing(Guinness(), [0,4])
tp.add_thing(Bed(), [2,4])
tp.add_thing(Guinness(), [3,2])
tp.add_thing(Shoes(), [4,0])
tp.add_thing(Song(), [2,2])
tp.add_thing(Bed(), [3,3])
tp.add_thing(Guinness(), [3,0])
tp.add_thing(Fiddle(), [0,3])
tp.add_thing(Bed(), [4,0])

print("IrishMan starts at (0,0) facing downwards, lets see if he can have some fun!")
tp.run(50)