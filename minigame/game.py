import gym
import gym.spaces
import numpy as np
import torch

##########################################################################################################
# CLASSES:                                                                                               #
#  There are classes for                                                                                 #
#     * Games                                                                                            #
#     * Orders for each unit type                                                                        #
#     * Factions (eg Blue or Re)                                                                         #
#     * Capabilities (Communicating, Jamming, ApproachFlying, PathFlying, Roaming, Occupying, Shooting)  #
#     * Units (Comm, Jammer, Occupying_Troop, Roaming_Troop)                                             #
#     * Parent classes (Unit, Drone, Moving, BlankOrder)                                                 #
##########################################################################################################

########
# GAME #
########

class Game():
    def __init__(self):
        # FACTIONS
        self.blue = None
        self.red = None
        # GLOBAL VARIABLES
        self.clock = 0
        # CONSTANTS
        self.TIMESTEP = None
        self.DEFAULT_ROAMING_RANDOM_PERTURBATION = None
        self.DEFAULT_FLY_SPEED = None
        self.DEFAULT_ROAM_SPEED = None
        self.DEFAULT_POINT_SOURCE_CONSTANT = None
        self.DEFAULT_SENDER_CHARACTERISTIC_DISTANCE = None
        self.DEFAULT_RECEIVER_CHARACTERISTIC_DISTANCE = None
        self.N_STREETS_EW = None
        self.N_STREETS_NS = None
        self.AMBIENT_POWER = None
        # SPACES
        self.observation_space = gym.spaces.Dict({'SELF': gym.spaces.Dict({'posx': gym.spaces.Discrete(32), 'posy': gym.spaces.Discrete(32)}),
                                                    'HQ': gym.spaces.Dict({'posx': gym.spaces.Discrete(32), 'posy': gym.spaces.Discrete(32)}),
                                                 'ASSET': gym.spaces.Dict({'posx': gym.spaces.Discrete(32), 'posy': gym.spaces.Discrete(32)}),
                                            'SELF2wayHQ': gym.spaces.Discrete(2),
                                         'SELF2wayASSET': gym.spaces.Discrete(2)})
        self.action_space = gym.spaces.Dict({'destx': gym.spaces.Discrete(32), 'desty': gym.spaces.Discrete(32), 'speed': gym.spaces.Discrete(8)})

    def add_blue_red(self, blue, red):
        assert blue is not red
        self.blue = blue
        blue.GAME = self
        blue.enemy = red
        self.red = red
        red.GAME = self
        red.enemy = blue
        self.restore_defaults()

    def make_units_dictionaries(self):
        if self.blue is not None:
            self.blue.make_units_dictionary()
        if self.red is not None:
            self.red.make_units_dictionary()

    def restore_defaults(self):
        if self.blue is not None:
            self.blue.restore_defaults()
        if self.red is not None:
            self.red.restore_defaults()

    def still_playing(self):
        '''
        still_playing(): asserts that self is the same game that its units and factions are playing
        '''
        assert self is self.blue.GAME
        assert self is self.red.GAME
        for unit in self.blue.units:
            assert self is unit.GAME
        for unit in self.red.units:
            assert self is unit.GAME

    def initialize(self):
        self.blue.initialize()
        self.red.initialize()

    def implement_ceoi(self):
        self.blue.implement_ceoi()
        self.red.implement_ceoi()

    def move(self):
        self.blue.move()
        self.red.move()

    def post_timestep(self):
        self.blue.post_timestep()
        self.red.post_timestep()

#    def run(self, n=1):
#        self.initialize()
#        self.implement_ceoi()
#        for _ in range(n):
#            self.clock += self.TIMESTEP
#            self.move()
#            self.post_timestep()

    def observe_blue(self):
         return {key : {'posx': int(round(self.blue.units_d[key].x_)),
                        'posy': int(round(self.blue.units_d[key].y_)),
                        'hears': {key2.name: self.blue.units_d[key].radio_message_received(key2) for key2 in self.blue.units_d[key].my_communicators()}}
                        for key in self.blue.units_d}
    def reset(self):
        return self.observe_blue()

#        return { 'COMM': {'posx': 1, 'posy': 1, 'hears': {'HQ': True, 'ASSET': False}},
#                   'HQ': {'posx': 0, 'posy': 0, 'hears': {'COMM': True, 'ASSET': False}},
#                'ASSET': {'posx': 31, 'posy': 31, 'hears': {'COMM': False, 'HQ': False}}}

    def step(self, action):
        # TODO Write this function
        obs = None
        reward = None
        done = None
        info = None
        return obs, reward, done, info

    def create_empty_grid(self):
        return list(('. '*self.N_STREETS_EW+'\n')*self.N_STREETS_NS)

    def add_faction_to_grid(self, faction, grid, character=None):
        for unit in faction.units:
            char = unit.label if character is None else character
            self.add_character_to_grid(grid, char, int(round(unit.y_)), int(round(unit.x_)))

    def add_character_to_grid(self, grid, character, ns, ew):
        grid[ns*(2*self.N_STREETS_NS + 1) + 2*ew] = character

    def convert_grid_to_string(self, grid):
        return ''.join(grid)

    def render(self, mode='human', close=False):
        assert mode == 'human'
        grid = self.create_empty_grid()
        self.add_faction_to_grid(self.blue, grid)
        self.add_faction_to_grid(self.red, grid)
        print(self.convert_grid_to_string(grid))


##########
# ORDERS #
##########

class BlankOrder():  # Default values for orders
    def __init__(self, unit):
        self.unit = unit

    def restore_defaults(self):
        pass

class StationaryOrder(BlankOrder): pass

class ParameterizedOrder(BlankOrder):
    def __init__(self, unit):
        super().__init__(unit)

    def pos_x(self, time):
        return 0.

    def vel_x(self, time):
        return 0.

    def pos_y(self, time):
        return 0.

    def vel_y(self, time):
        return 0.
 
class CircleOrder(BlankOrder):
    def __init__(self, unit):
        super().__init__(unit)
        self.center_x = .5
        self.center_y = .5
        self.radius = .5
        self.initial_phase = 0.
        self.time_at_initial_phase = 0.
        self.signed_speed = 1.  # Positive for counterclockwise, negative for clockwise

    def phase(self, time):
        return self.signed_speed*(time - self.time_at_initial_phase)/self.radius + self.initial_phase

    def pos_x(self, time):
        return self.center_x + self.radius*np.cos(self.phase(time))

    def vel_x(self, time):
        return self.signed_speed*np.sin(self.phase(time))

    def pos_y(self, time):
        return self.center_y + self.radius*np.sin(self.phase(time))

    def vel_y(self, time):
        return self.signed_speed*np.cos(self.phase(time))

class ApproachingOrder(BlankOrder):
    def __init__(self, unit):
        super().__init__(unit)
        self.destination_x = None
        self.destination_y = None

    def set_destination(self, dest_x, dest_y):
        self.destination_x = dest_x
        self.destination_y = dest_y

class ApproachingGaussianOrder(ApproachingOrder):
    def __init__(self, unit):
        super().__init__(unit)
        self.var_major = None
        self.var_minor = None
        self.cov_theta = None
        self.var_x_ = None  # trailing underbars mean variable is internal and not under direct control of agents
        self.var_y_ = None
        self.cov_xy_ = None

    def set_covariance(self, var_major, var_minor, cov_theta):  #TODO this needs to be checked!!
        # "major" axis is actually indicated by max(var_major, var_minor) OK if reversed       
        # How I derived this:
        # The covariance matrix is symmetric and positive definite so its eigen decomposition coincides with this singular decomposition
        # Write down singular values (which in this case are eigenvalues) in a diagonal matrix
        # Write down singular/eigen vector matrices which are rotation in an orthogonal matrix and its transpose.  Then multiply matrices.
        self.var_major = var_major
        self.var_minor = var_minor
        self.cov_theta = cov_theta
        self.var_x_ = var_major*sin(cov_theta)**2 + var_minor*cos(cov_theta)**2
        self.var_y_ = var_major*cos(cov_theta)**2 + var_minor*sin(cov_theta)**2
        self.cov_xy_ = (var_major - var_minor)*sin(cov_theta)*cos(cov_theta)

class ApproachingCommOrder(ApproachingOrder):
    def __init__(self, unit):
        assert isinstance(unit, Comm)
        super().__init__(unit)

class CircleCommOrder(CircleOrder):
    def __init__(self, unit):
        assert isinstance(unit, Comm)
        super().__init__(unit)

class ApproachingJammerOrder(ApproachingOrder):
    def __init__(self, unit):
        assert isinstance(unit, Jammer)
        super().__init__(unit)

class CircleJammerOrder(CircleOrder):
    def __init__(self, unit):
        assert isinstance(unit, Jammer)
        super().__init__(unit)

class OccupyingTroopOrder(StationaryOrder):
    def __init__(self, unit):
        assert isinstance(unit, OccupyingTroop)
        super().__init__(unit)

class ApproachingGaussianRoamingTroopOrder(ApproachingGaussianOrder):
    def __init__(self, unit):
        assert isinstance(unit, RoamingTroop)
        super().__init__(unit)

# Might want to add flags to Order classes __init__, where appropriate (especially jam)
#        self.communicate = True
#        self.jam = True
#        self.occupy_roof = True
#        self.shoot = True

###########################
# FACTIONS -- BLUE or RED #
###########################

class Faction():
    def __init__(self, name, GAME=None):
        self.name = name
        self.enemy = None
        self.GAME = GAME
        self.units = []
        self.units_d = {}
        self.headquarters = None
        self.communication_network = []
        self.jamming_network = []

    # All methods until noted dont't have counterparts in units classes
    def add_headquarters(self, unit, name=None, label=None, x_=None, y_=None):
        self.add_unit(unit, name, label, x_, y_)
        self.headquarters = unit

    def add_unit(self, unit, name=None, label=None, x_=None, y_=None):
        unit.faction = self
        unit.regame(self.GAME)
        if name is not None:
            unit.name = name
        if label is not None:
            unit.label = label
        if x_ is not None:
            unit.x_ = x_
        if y_ is not None:
            unit.y_ = y_
        self.units.append(unit)
        self.make_units_dictionary()

    def make_units_dictionary(self):
        self.units_d = {unit.name: unit for unit in self.units}
        assert len(self.units_d) == len(self.units)

    def pop_unit(self):
        unit = self.units.pop()
        unit.faction = None

    def clear_units(self):
        for u in self.units:
            u.faction = None
        self.units = []

    def add_unit_to_communication_network(self, unit):
        self.communication_network.append(unit)

    def add_unit_to_jamming_network(self, unit):
        self.jamming_network.append(unit)

    # All methods below have counterparts in units classes
    def restore_defaults(self):
        for unit in self.units:
            unit.restore_defaults()

    def initialize(self):
        for unit in self.units:
            unit.initialize()

    def implement_ceoi(self):
        for unit in self.units:
            unit.implement_ceoi()

    def move(self):
        for unit in self.units:
            unit.move()

    def post_timestep(self):
        for unit in self.units:
            unit.post_timestep()

#    def step(self, action):  #TODO NEED WORK!
#        obs_list = []
#        reward_list = []
#        done_list = []
#        info_list = []
#        for i, unit in enumerate(self.units):
#            obs_i, reward_i, done_i, info_i = unit.step(action[i])
#            obs_list.append(obs_i)
#            reward_list.append(reward_i)
#            done_list = append(done_i)
#            info_list = append(info_i)
#        return obs_list, reward_list, done_list, info_list

################
# CAPABILITIES #
################

class Moving():  # Parent class to Flying and Roaming
    def plan_timestep_motion(self):   # Used for Comm, Jammer & RoamingTroop but NOT Occupying Troop
        '''
        plan_timestep_motion(): defines delta_x, delta_y, vx, vy in direction of destination but magnitude not greater than self.max_speed
        '''
        ideal_delta_x = self.order.destination_x - self.x_
        ideal_delta_y = self.order.destination_y - self.y_
        ideal_speed = self.distance_to_target() / self.GAME.TIMESTEP  # distance to target l2 for Flying, l1 for Roaming
        if ideal_speed <= self.max_speed:  # not too fast
            self.delta_x = ideal_delta_x
            self.delta_y = ideal_delta_y
        else:  # too fast
            self.delta_x = ideal_delta_x * self.max_speed/ideal_speed
            self.delta_y = ideal_delta_y * self.max_speed/ideal_speed
        self.vx = self.delta_x / self.GAME.TIMESTEP
        self.vy = self.delta_y / self.GAME.TIMESTEP

    def step_xy(self):
        self.x_ += self.delta_x
        self.y_ += self.delta_y

class Flying(Moving):
    def fly(self): # overload this method
        self.step_xy()

    def vector_norm_2D(self, x, y):  # l2
        return np.sqrt(x**2 + y**2)

    def restore_capability_defaults(self):
        self.max_speed = self.GAME.DEFAULT_FLY_SPEED

class ApproachFlying(Flying): pass

class PathFlying(Flying): pass

class Roaming(Moving):
    def roam(self):
        self.step_xy()
        #TODO Add Randomization

    def vector_norm_2D(self, x, y):  # l1
        return np.abs(x) + np.abs(y)

    def restore_capability_defaults(self):
        self.max_speed = self.GAME.DEFAULT_ROAM_SPEED

class Occupying():
    def place_on_target(self, occupy_x, occupy_y):
        self.x_ = occupy_x
        self.y_ = occupy_y

class Communicating():
    def add_self_to_communication_network(self):
        # TODO Consider: maybe don't need communication_network if have self.communicate flags for each unit; has consistent order; redundant for now
        self.faction.add_unit_to_communication_network(self)  # self becomes "unit" inside faction

    def radio_power(self, x_receiver, y_receiver):
        return self.point_source_constant * self.sender_characteristic_distance**2 / ((x_receiver - self.x_)**2 + (y_receiver - self.y_)**2)

    def sjr_db(self, x_receiver, y_receiver):
        return 10.*np.log10(self.radio_power(x_receiver, y_receiver)/self.faction.GAME.AMBIENT_POWER)

    def radio_message_received(self, unit):  # unit instead of x_ and y_
        assert self.receiver_characteristic_distance == 0  # == 0 determinisitic, \neq 0 not yet implemented
        return self.sjr_db(unit.x_, unit.y_) > 0  #TODO Add probabilistic function like in findjam

class Jamming():
    def add_self_to_jamming_network(self):
        self.faction.add_unit_to_jamming_network(self)  # self becomes "unit" inside faction

class Shooting():
    def shoot_enemy_drones(self):
        pass  #TODO Add this function

#########
# UNITS #
#########

class Unit():  # Parent class to all units
    def __init__(self, GAME=None):
        self.order = BlankOrder(self)  # self is the second arg that becomes unit inside __init__
        if GAME is None:
            self.regame(NoGAME)  # defines self.GAME as NoGAME and calls self.restore_defaults()
        else:
            self.regame(GAME)
        self.faction = None
        self.name = 'GHOST'
        self.label = 'G'
        self.on_roof = False
        self.x_ = 0.1
        self.y_ = 0.1
        self.delta_x = 0
        self.delta_y = 0
        self.vx = 0.
        self.vy = 0.
        self.asset_value = 0.
        self.communicates = False

    def my_communicators(self):
        for unit in self.faction.units:
            if unit is not self and self.communicates and unit.communicates:
                yield unit

    def set_name(self, name):
        self.name = name

    def set_initial(self, init_x, init_y):
        self.x_ = init_x
        self.y_ = init_y

    def reset_xy(self, init_x, init_y):
        self.x_ = init_x
        self.y_ = init_y

    def regame(self, GAME):
        self.GAME = GAME
        self.restore_defaults()

    def restore_defaults(self):
        self.restore_unit_defaults()
        self.restore_capability_defaults()
        self.order.restore_defaults()

    def restore_unit_defaults(self):
        pass

    def restore_capability_defaults(self):
        pass

    def initialize(self):
        pass

    def implement_ceoi(self):
        pass

    def move(self):
        pass

    def post_timestep(self):
        pass

    def x(self):
        return self.x_

    def y(self):
        return self.y_

    def xy(self):
        return (self.x_, self.y_)

    def distance_to_target(self):  # l2 distance with flying, shouldn't have both Roaming and Flying capabilities
        return self.vector_norm_2D(self.order.destination_x - self.x_, self.order.destination_y - self.y_)

class Drone(ApproachFlying, Unit):  # Parent class of Comm and Jammer
    def __init__(self, GAME=None):
        super().__init__(GAME)
        self.name = 'DRONE'
        self.label = 'D'
        self.shot_down = False

    def move(self):
        self.plan_timestep_motion()
        self.fly()


class Comm(Communicating, Drone):
    def __init__(self, GAME=None):
        super().__init__(GAME)
        self.name = 'COMM'
        self.label = 'C'
        self.order = ApproachingCommOrder(self)  # self is the second arg that becomes "unit" inside ApproachingCommOrder.__init__(self, unit)
        self.asset_value = 1.
        self.communicates = True

    def restore_unit_defaults(self):
        self.point_source_constant = self.GAME.DEFAULT_POINT_SOURCE_CONSTANT  # DEFAULT_POINT_SOURCE_CONSTANT is a global constant
        self.receiver_characteristic_distance = self.GAME.DEFAULT_RECEIVER_CHARACTERISTIC_DISTANCE  # DEFAULT_RECEIVER_CHARACTERISTIC_DISTANCE is a global constant
        self.sender_characteristic_distance = self.GAME.DEFAULT_SENDER_CHARACTERISTIC_DISTANCE  # DEFAULT_SENDER_CHARACTERISTIC_DISTANCE is a global constant

    def implement_ceoi(self):
        self.add_self_to_communication_network()
        self.communicates = True


class Jammer(Jamming, Drone):
    def __init__(self, GAME=None):
        super().__init__(GAME)
        self.name = 'JAMMER'
        self.label = 'J'
        self.order = ApproachingJammerOrder(self)  # self is the second arg that becomes "unit" inside ApproachingJammerOrder.__init__(self, unit)

    def restore_unit_defaults(self):
        self.point_source_constant = self.GAME.DEFAULT_POINT_SOURCE_CONSTANT  # DEFAULT_POINT_SOURCE_CONSTANT is a global constant

    def implement_ceoi(self):
        self.add_self_to_jamming_network()
        

class OccupyingTroop(Occupying, Communicating, Shooting, Unit):
    def __init__(self, GAME=None):
        super().__init__(GAME)
        self.name = 'OCCUPYING_TROOP'
        self.label = 'O'
        self.order = OccupyingTroopOrder(self)  # self is the second arg that becomes unit inside __init__
        self.asset_value = 10.
        self.communicates = True

    def restore_unit_defaults(self):
        self.point_source_constant = self.GAME.DEFAULT_POINT_SOURCE_CONSTANT  # DEFAULT_POINT_SOURCE_CONSTANT is a global constant
        self.receiver_characteristic_distance = self.GAME.DEFAULT_RECEIVER_CHARACTERISTIC_DISTANCE  # DEFAULT_RECEPTION_PROBABILITY_SLOPE is a global constant
        self.sender_characteristic_distance = self.GAME.DEFAULT_SENDER_CHARACTERISTIC_DISTANCE  # DEFAULT_SENDER_PROBABILITY_SLOPE is a global constant

    def implement_ceoi(self):
        self.add_self_to_communication_network()
        self.communicates = True

    def post_timestep(self):
        self.shoot_enemy_drones()


class RoamingTroop(Roaming, Shooting, Unit):
    def __init__(self, GAME=None):
        super().__init__(GAME)
        self.name = 'ROAMING_TROOP'
        self.label = 'R'
        self.order = ApproachingGaussianRoamingTroopOrder(self)  # self is the second arg that becomes unit inside __init__

    def move(self):
        self.plan_timestep_motion()
        self.roam()

    def post_timestep(self):
        self.shoot_enemy_drones()

NoGAME = Game()

GAME0 = Game()
GAME0.TIMESTEP = .1
GAME0.DEFAULT_ROAMING_RANDOM_PERTURBATION = 2.
GAME0.DEFAULT_FLY_SPEED = 5.
GAME0.DEFAULT_ROAM_SPEED = 2.
GAME0.DEFAULT_POINT_SOURCE_CONSTANT = 1.
GAME0.DEFAULT_RECEIVER_CHARACTERISTIC_DISTANCE = 0.
GAME0.N_STREETS_EW = 48
GAME0.N_STREETS_NS = 48
GAME0.AMBIENT_POWER = 1.
GAME0.restore_defaults()
GAME0.add_blue_red(Faction('BLUE', GAME0), Faction('RED', GAME0))
GAME0.blue.add_unit(Comm())
GAME0.blue.units[-1].order.set_destination(0.9, 0.9)
GAME0.blue.units[-1].x_ = 3.
GAME0.blue.units[-1].y_ = 4.
GAME0.blue.add_unit(OccupyingTroop())
GAME0.blue.units[-1].x_ = 2.
GAME0.blue.units[-1].y_ = 3.
GAME0.red.add_unit(Jammer())
GAME0.red.units[-1].order.set_destination(0.1, 0.9)
GAME0.red.units[-1].x_ = 5.
GAME0.red.units[-1].y_ = 1.
GAME0.red.add_unit(RoamingTroop())
GAME0.red.units[-1].order.set_destination(0.9, 0.1)
GAME0.red.units[-1].x_ = 5.
GAME0.red.units[-1].y_ = 2.
GAME0.make_units_dictionaries()

def GAME1(ew, ns):
    G1 = Game()
    G1.TIMESTEP = .1
    G1.DEFAULT_ROAMING_RANDOM_PERTURBATION = 2.
    G1.DEFAULT_FLY_SPEED = 5.
    G1.DEFAULT_ROAM_SPEED = 2.
    G1.DEFAULT_POINT_SOURCE_CONSTANT = 1.
    G1.DEFAULT_RECEIVER_CHARACTERISTIC_DISTANCE = 0.
    G1.DEFAULT_SENDER_CHARACTERISTIC_DISTANCE = 1.5
    G1.N_STREETS_EW = ew
    G1.N_STREETS_NS = ns
    G1.AMBIENT_POWER = 1.
    G1.restore_defaults()
    G1.add_blue_red(Faction('BLUE'), Faction('RED'))
    G1.blue.add_unit(Comm(), name='COMM', label='C', x_=1., y_=1.)
    G1.blue.add_headquarters(OccupyingTroop(), name='HQ', label='H', x_=0., y_=0.)
    G1.blue.add_unit(OccupyingTroop(), name='ASSET', label='A', x_=ew-1., y_=ns-1.)  # SE Corner of map
    return G1
