import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np

##########################################################################################################
# CLASSES:                                                                                               #
#  There are classes for                                                                                 #
#     * Maps                                                                                             #
#     * Games                                                                                            #
#     * Orders for each unit type                                                                        #
#     * Factions (eg Blue or Re)                                                                         #
#     * Capabilities (Communicating, Jamming, ApproachFlying, PathFlying, Roaming, Occupying, Shooting)  #
#     * Units (Comm, Jammer, Occupying_Troop, Roaming_Troop)                                             #
#     * Parent classes (Unit, Drone, Moving, BlankOrder)                                                 #
#  Plus predefined games at the end.                                                                     #
##########################################################################################################

########
# MAPS #
########

class Map0():
    def __init__(self, GAME):
        self.GAME = GAME
        self.define_defaults()
        self.restore_defaults()

    def define_defaults(self):  # THESE DEFAULTS SHOULD BE SHARED BY ALL MAPS
        self.DEFAULT_RECEIVER_CHARACTERISTIC_DISTANCE = 0.
        self.DEFAULT_SENDER_CHARACTERISTIC_DISTANCE = 5.
        self.DEFAULT_POINT_SOURCE_CONSTANT = 1.
        self.DEFAULT_N_STREETS_EW = 6
        self.DEFAULT_N_STREETS_NS = 6
        self.define_specific_defaults()

    def define_specific_defaults(self):  # THESE DEFAULTS SPECIFIC TO JUST THIS MAP
        self.DEFAULT_COMMX = 3.
        self.DEFAULT_COMMY = 4.
        self.DEFAULT_OCCX = 2.
        self.DEFAULT_OCCY = 3.
        self.DEFAULT_JAMX = 5.
        self.DEFAULT_JAMY = 1.
        self.DEFAULT_ROAMX = 5.
        self.DEFAULT_ROAMY = 2.

    def restore_defaults(self):  # THESE DEFAULTS SHOULD BE SHARED BY ALL MAPS
        self.receiver_characteristic_distance = self.DEFAULT_RECEIVER_CHARACTERISTIC_DISTANCE
        self.sender_characteristic_distance = self.DEFAULT_SENDER_CHARACTERISTIC_DISTANCE
        self.n_streets_ew = self.DEFAULT_N_STREETS_EW
        self.n_streets_ns = self.DEFAULT_N_STREETS_NS
        self.restore_specific_defaults()

    def restore_specific_defaults(self):  # THESE DEFAULTS SPECIFIC TO JUST THIS MAP
        self.commx = self.DEFAULT_COMMX
        self.commy = self.DEFAULT_COMMY
        self.occx = self.DEFAULT_OCCX
        self.occy = self.DEFAULT_OCCY
        self.jamx = self.DEFAULT_JAMX
        self.jamy = self.DEFAULT_JAMY
        self.roamx = self.DEFAULT_ROAMX
        self.roamy = self.DEFAULT_ROAMY

    def remap(self):
        self.GAME.add_blue_red(Faction('BLUE'), Faction('RED'))
        self.GAME.blue.clear_units()
        self.GAME.red.clear_units()
        self.add_units_to_map()
        self.GAME.define_spaces()

    def add_units_to_map(self):
        self.GAME.blue.add_unit(Comm(self.GAME), name='COMM', label='C', x_=self.commx, y_=self.commy)
        self.GAME.blue.add_headquarters(OccupyingTroop(self.GAME), name='OCC', label='O', x_=self.occx, y_=self.occy)
        self.GAME.blue.add_unit(OccupyingTroop(self.GAME), name='JAM', label='J', x_=self.jamx, y_=self.jamy)  # SE Corner of map
        self.GAME.blue.add_unit(OccupyingTroop(self.GAME), name='ROAM', label='R', x_=self.roamx, y_=self.roamy)  # SE Corner of map


class Map1(Map0):
    def define_specific_defaults(self):
        self.DEFAULT_COMMX = 1.
        self.DEFAULT_COMMY = 1.
        self.DEFAULT_HQX = 0.
        self.DEFAULT_HQY = 0.
        self.DEFAULT_ASSETX = 7.
        self.DEFAULT_ASSETY = 7.

    def restore_specific_defaults(self):
        self.commx = self.DEFAULT_COMMX
        self.commy = self.DEFAULT_COMMY
        self.hqx = self.DEFAULT_HQX
        self.hqy = self.DEFAULT_HQY
        self.assetx = self.DEFAULT_ASSETX
        self.assety = self.DEFAULT_ASSETY

    def add_units_to_map(self):
        self.GAME.blue.add_unit(Comm(self.GAME), name='COMM', label='C', x_=self.commx, y_=self.commy)
        self.GAME.blue.units[-1].order.turn_on_control()
        self.GAME.blue.add_headquarters(OccupyingTroop(self.GAME), name='HQ', label='H', x_=self.hqx, y_=self.hqy)
        self.GAME.blue.add_unit(OccupyingTroop(self.GAME), name='ASSET', label='A', x_=self.assetx, y_=self.assety)  # SE Corner of map

#########
# GAMES #
#########

class Game(gym.Env):
    def __init__(self):
        # FACTIONS
        self.blue = None
        self.red = None
        # GLOBAL VARIABLES
        self.clock = 0
        # CONSTANTS
        self.DEFAULT_TIMESTEP = 5.
        self.DEFAULT_MAX_TIME = 20.
        self.DEFAULT_FLY_SPEED = 2.
        self.DEFAULT_ROAM_SPEED = 1.
        self.DEFAULT_POINT_SOURCE_CONSTANT = 1.
        self.DEFAULT_AMBIENT_POWER = 1.
        # RANDOM NUMBER GENERATORS
        self.example_rng = np.random.RandomState()
        # MAPS
        self.make_map()  # usually calls overloaded method which calls remap() which defines observation_space and action_space

    def make_map(self):  # Overload for specific game
        pass

    def define_observation_space(self):
        return gym.spaces.Dict({
            key : gym.spaces.Dict(
            {'posx': gym.spaces.Box(low=0., high=float(self.map.n_streets_ew-1), shape=[1]),
             'posy': gym.spaces.Box(low=0., high=float(self.map.n_streets_ns-1), shape=[1]),
             'hears': gym.spaces.Dict(
                 # {key2.name: gym.spaces.Discrete(2) for key2 in self.blue.unitd[key].my_communicators()})})
                 {key2.name: gym.spaces.Box(low=-np.inf, high=np.inf, shape=[1]) for key2 in self.blue.unitd[key].my_communicators()})})
            for key in self.blue.unitd})  #TODO Make obsevation include red
#        return gym.spaces.Dict({
#            'COMM': gym.spaces.Dict({'posx': gym.spaces.Discrete(32), 'posy': gym.spaces.Discrete(32),
#                'hears': gym.spaces.Dict({'HQ': gym.spaces.Discrete(2), 'ASSET': gym.spaces.Discrete(2)})}),
#            'HQ': gym.spaces.Dict({'posx': gym.spaces.Discrete(32), 'posy': gym.spaces.Discrete(32),
#                'hears': gym.spaces.Dict({'COMM': gym.spaces.Discrete(2), 'ASSET':  gym.spaces.Discrete(2)})}),
#            'ASSET': gym.spaces.Dict({'posx': gym.spaces.Discrete(32), 'posy': gym.spaces.Discrete(32), 
#                'hears': gym.spaces.Dict({'COMM': gym.spaces.Discrete(2), 'HQ': gym.spaces.Discrete(2)})})})

    def define_action_space(self):  # TODO need to adjust for red actions
        return gym.spaces.Dict(
                {unit.name: 
                      gym.spaces.Dict(
                           {'destx': gym.spaces.Box(low=0., high=float(self.map.n_streets_ew-1), shape=[1]),
                            'desty': gym.spaces.Box(low=0., high=float(self.map.n_streets_ns-1), shape=[1])}
                           )
                      if not unit.order.speed_specification['controlled']
                      else gym.spaces.Dict(
                           {'destx': gym.spaces.Box(low=0., high=float(self.map.n_streets_ew-1), shape=[1]),
                            'desty': gym.spaces.Box(low=0., high=float(self.map.n_streets_ns-1), shape=[1]),
                            'speed': gym.spaces.Discrete(3)} # speed 0, 1, or 2 TODO Generalize
                           )
                      for unit in self.blue.units_with_controlled_destination()})  # TODO Generalize to Red
#               gym.spaces.Dict({'destx': gym.spaces.Discrete(32), 'desty': gym.spaces.Discrete(32), 'speed': gym.spaces.Discrete(8)})
#               { 'COMM': {'destx': 9, 'desty': 9, 'speed': 1} }

    def example_action(self, seed=None):
        self.example_rng.seed(seed)
        return {unit.name:
                           {'destx': self.example_rng.randint(self.map.n_streets_ew),
                            'desty': self.example_rng.randint(self.map.n_streets_ns)}
                      if not unit.order.speed_specification['controlled']
                      else
                           {'destx': self.example_rng.randint(self.map.n_streets_ew),
                            'desty': self.example_rng.randint(self.map.n_streets_ns),
                            'speed': self.example_rng.randint(3)}  # TODO Generalize
                      for unit in self.blue.units_with_controlled_destination()}  # TODO Generalize to Red

    def define_spaces(self):
        self.observation_space = self.define_observation_space()
        self.action_space = self.define_action_space()

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
        self.map.restore_defaults()
        self.timestep = self.DEFAULT_TIMESTEP
        self.max_time = self.DEFAULT_MAX_TIME
        self.fly_speed = self.DEFAULT_FLY_SPEED
        self.roam_speed = self.DEFAULT_ROAM_SPEED
        self.point_source_constant = self.DEFAULT_POINT_SOURCE_CONSTANT
        self.ambient_power = self.DEFAULT_AMBIENT_POWER

    def still_playing(self):
        '''
        still_playing(): asserts that self is the same Game that its map, factions and units are associated with
        '''
        assert self is self.map.GAME
        assert self is self.blue.GAME
        assert self is self.red.GAME
        for unit in self.blue.units:
            assert self is unit.GAME
        for unit in self.red.units:
            assert self is unit.GAME

    def initialize(self):
        self.clock = 0.
        self.blue.initialize()
        self.red.initialize()

    def implement_ceoi(self):
        self.blue.implement_ceoi()
        self.red.implement_ceoi()

    def pre_timestep(self):
        self.clock += self.timestep
        self.blue.pre_timestep()
        self.red.pre_timestep()

    def move(self):
        self.blue.move()
        self.red.move()

    def post_timestep(self):
        self.blue.post_timestep()
        self.red.post_timestep()

    def observe_connections(self, faction):
        return {key: {key2.name for key2 in faction.unitd[key].hears_me()}
                for key in faction.unitd}

    def observe_faction(self, faction):  # Now taking rounding out
        # return {key : {'posx': int(round(faction.unitd[key].x_)),
        #                'posy': int(round(faction.unitd[key].y_)),
        #                'hears': {key2.name: faction.unitd[key].radio_message_received(key2) for key2 in faction.unitd[key].my_communicators()}}
        #                for key in faction.unitd}
#       #         { 'COMM': {'posx': 1, 'posy': 1, 'hears': {'HQ': True, 'ASSET': False}},
        return {key : {'posx': faction.unitd[key].x_,
                       'posy': faction.unitd[key].y_,
                       'hears': {key2.name: faction.unitd[key].radio_message_received(key2) for key2 in faction.unitd[key].my_communicators()}}
                       for key in faction.unitd}
#               { 'COMM': {'posx': 1, 'posy': 1, 'hears': {'HQ': True, 'ASSET': False}},
#                   'HQ': {'posx': 0, 'posy': 0, 'hears': {'COMM': True, 'ASSET': False}},
#                'ASSET': {'posx': 31, 'posy': 31, 'hears': {'COMM': False, 'HQ': False}}}

    def reset(self):
        self.map.remap()
        self.initialize()
        self.implement_ceoi()
        return self.observe_faction(self.blue)  #TODO Need to observe red eventually, too.

    def parse_blue_into_order(self, action):
        for unitname in action:
            assert self.blue.unitd[unitname].order.destination_specification['controlled']
            self.blue.unitd[unitname].order.set_destination(action[unitname]['destx'], action[unitname]['desty'])
            if self.blue.unitd[unitname].order.speed_specification['controlled']:
                self.blue.unitd[unitname].order.set_speed(action[unitname]['speed'])
#       eg. parse something like: action = { 'COMM': {'destx': 9, 'desty': 9, 'speed': 1} }  # add speed 

    def parse_red_into_order(self, action):  #TODO Write this function
        pass

    def parse_action_into_order(self, action):  # eventually action = {'BLUE': {...}, 'RED': {...}} but only if both have controlled units
        self.parse_blue_into_order(action)      # eventually ...into_order(action['BLUE'])
                                                # eventually ...into_order(action['RED'])

    def temp_reward(self):
        assert 'COMM' in self.blue.unitd
        assert 'HQ' in self.blue.unitd
        assert 'ASSET' in self.blue.unitd
        assert len(self.blue.unitd) == 3
        return self.blue.unitd['ASSET'].asset_value*(
                float(self.blue.unitd['HQ'].radio_message_received(self.blue.unitd['COMM'])) +
                float(self.blue.unitd['COMM'].radio_message_received(self.blue.unitd['ASSET'])))

    def step(self, action): # TODO Write this function
        self.parse_action_into_order(action)
        self.pre_timestep()
        self.move()
        self.post_timestep()
        obs = self.observe_faction(self.blue)  #TODO Need to observe red eventually, too.
        reward = self.temp_reward()
        done = self.clock > self.max_time
        info = {}  # No info until debug time
        return obs, reward, done, info

    def run(self, action, n=1):  # Only run if no units are controlled
        for _ in range(n):
            self.step(action)

    def advance(self, action, n=1):
        self.run(action, n)
        self.render()

    def create_empty_grid(self):
        return list(('. '*self.map.n_streets_ew+'\n')*self.map.n_streets_ns)

    def add_faction_to_grid(self, faction, faction_character=None):
        for unit in faction.units:
            char = unit.label if faction_character is None else faction_character
            self.add_character_to_grid(char, int(round(unit.y_)), int(round(unit.x_)))

    def add_character_to_grid(self, character, ns, ew):
        self.grid[ns*(2*self.map.n_streets_ew + 1) + 2*ew] = character

    def convert_grid_to_string(self):
        return ''.join(self.grid)

    def render(self, mode='human', close=False):
        assert mode == 'human'
        self.grid = self.create_empty_grid()
        self.add_faction_to_grid(self.blue)   #TODO add red labels
        self.add_faction_to_grid(self.red)
        print(self.convert_grid_to_string())
        print(self.observe_connections(self.blue))  #TODO add red connections

class Game0(Game):
    def make_map(self):
        self.map = Map0(self)
        self.map.remap()

class Game1(Game):
    def __init__(self, mapsize):
        self.mapsize = mapsize
        super().__init__()

    def make_map(self):
        self.map = Map1(self)
        self.map.DEFAULT_N_STREETS_NS = self.mapsize
        self.map.DEFAULT_N_STREETS_EW = self.mapsize
        self.map.DEFAULT_ASSETX = self.mapsize - 1.
        self.map.DEFAULT_ASSETY = self.mapsize - 1.
        self.map.restore_defaults()
        self.map.remap()

class UavEnv(Game1):
    def __init__(self):
        mapsize = 8
        super().__init__(mapsize)

##########
# ORDERS #
##########

class BlankOrder():  # Default values for orders
    def __init__(self, unit):
        self.unit = unit
        self.destination_specification = {'controlled': None, 'stationary': None}
        self.speed_specification = {'controlled': None, 'zero': None, 'maximum': None}
        self.turn_on_stationary()
    # Destination: if controlled, parse action, if stationary, put in None
    # Speed: [if controlled, parse action], if zero, put in None, if max, put that in.

    # TODO Add specification for random motion

    def turn_on_stationary(self):
        self.destination_specification = {'controlled': False, 'stationary': True}
        self.speed_specification = {'controlled': False, 'zero': True, 'maximum': False}
        self.specification_integrity()

    def turn_on_control(self):
        self.destination_specification = {'controlled': True, 'stationary': False}
        self.speed_specification = {'controlled': True, 'zero': False, 'maximum': False}
        self.specification_integrity()

    def turn_on_controlled_destination_max_speed(self):
        self.destination_specification = {'controlled': True, 'stationary': False}
        self.speed_specification = {'controlled': False, 'zero': False, 'maximum': True}
        self.set_speed()
        self.specification_integrity()

    def specification_integrity(self):
        self.assert_one_hot(self.destination_specification)
        self.assert_one_hot(self.speed_specification)
        self.assert_zero_consistent()
        self.assert_if_speed_controlled_then_destination_controlled()

    def assert_one_hot(self, spec):
        hot = 0
        for key in spec:
            assert spec[key] is True or spec[key] is False
            if spec[key]:
                hot += 1
        assert hot == 1

    def assert_zero_consistent(self):
        assert ((self.destination_specification['stationary'] and self.speed_specification['zero'])
                or not(self.destination_specification['stationary'] and not self.speed_specification['zero']))

    def assert_if_speed_controlled_then_destination_controlled(self):
        if self.speed_specification['controlled']:
            assert self.destination_specification['controlled']

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
        self.speed = None

    def set_destination(self, dest_x, dest_y):
        self.destination_x = dest_x
        self.destination_y = dest_y

    def set_speed(self, speed=None):
        self.speed = speed if speed is not None else self.max_speed

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
        self.unitd = {}
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
        self.unitd = {unit.name: unit for unit in self.units}
        assert len(self.unitd) == len(self.units)

    def pop_unit(self):
        unit = self.units.pop()
        unit.faction = None
        self.make_units_dictionary()

    def clear_units(self):
        for u in self.units:
            u.faction = None
        self.units = []
        self.make_units_dictionary()

    def add_unit_to_communication_network(self, unit):
        self.communication_network.append(unit)

    def add_unit_to_jamming_network(self, unit):
        self.jamming_network.append(unit)

    def units_with_controlled_destination(self):
        for unit in self.units:
            if unit.order.destination_specification['controlled']:
                yield unit

    def units_with_controlled_speed(self):
        for unit in self.units:
            if unit.order.speed_specification['controlled']:
                yield unit

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

    def pre_timestep(self):
        for unit in self.units:
            unit.pre_timestep()

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
        plan_timestep_motion(): defines delta_x, delta_y, vx, vy in direction of destination but magnitude not greater than self.order.speed
        '''
        ideal_delta_x = self.order.destination_x - self.x_
        ideal_delta_y = self.order.destination_y - self.y_
        ideal_speed = self.distance_to_target() / self.GAME.timestep  # distance to target l2 for Flying, l1 for Roaming
        if ideal_speed <= self.order.speed:  # not too fast
            self.delta_x = ideal_delta_x
            self.delta_y = ideal_delta_y
        else:  # too fast
            self.delta_x = ideal_delta_x * self.order.speed/ideal_speed
            self.delta_y = ideal_delta_y * self.order.speed/ideal_speed
        self.vx = self.delta_x / self.GAME.timestep
        self.vy = self.delta_y / self.GAME.timestep

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
        return 10.*np.log10(self.radio_power(x_receiver, y_receiver)/self.faction.GAME.ambient_power)

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
            self.regame(Game())  # defines self.GAME as Empty Game and calls self.restore_defaults()
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

    def hears_me(self):  #TODO only works for determinisic connections, save an adjacency matrix otherwise, or output probabilities
        for unit in self.faction.units:
            if unit is not self and self.communicates and unit.communicates and self.radio_message_received(unit):
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

    def pre_timestep(self):
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
        self.receiver_characteristic_distance = self.GAME.map.DEFAULT_RECEIVER_CHARACTERISTIC_DISTANCE  # DEFAULT_RECEIVER_CHARACTERISTIC_DISTANCE is a global constant
        self.sender_characteristic_distance = self.GAME.map.DEFAULT_SENDER_CHARACTERISTIC_DISTANCE  # DEFAULT_SENDER_CHARACTERISTIC_DISTANCE is a global constant

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
        self.receiver_characteristic_distance = self.GAME.map.DEFAULT_RECEIVER_CHARACTERISTIC_DISTANCE  # DEFAULT_RECEPTION_PROBABILITY_SLOPE is a global constant
        self.sender_characteristic_distance = self.GAME.map.DEFAULT_SENDER_CHARACTERISTIC_DISTANCE  # DEFAULT_SENDER_PROBABILITY_SLOPE is a global constant

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

