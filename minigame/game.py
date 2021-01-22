import numpy as np
import torch

########################################################################################
# CLASSES:                                                                             #
#  There are classes for                                                               #
#     * Games                                                                          #
#     * Orders for each unit type                                                      #
#     * Factions (eg Blue or Re)                                                       #
#     * Capabilities (Communicating, Jamming, Flying, Roaming, Occupying, Shooting)    #
#     * Units (Comm, Jammer, Occupying_Troop, Roaming_Troop)                           #
#     * Parent classes (Unit, Drone, Moving, BlankOrder)                               #
########################################################################################

########
# GAME #
########

class Game():
    def __init__(self):
        # FACTIONS
        self.blue = None
        self.red = None
        # CONSTANTS
        self.TIMESTEP = None
        self.DEFAULT_ROAMING_RANDOM_PERTURBATION = None
        self.DEFAULT_FLY_SPEED = None
        self.DEFAULT_ROAM_SPEED = None
        self.DEFAULT_POINT_SOURCE_CONSTANT = None
        self.DEFAULT_RECEPTION_PROBABILITY_SLOPE = None

    def add_blue_red(self, blue, red):
        assert blue is not red
        self.blue = blue
        blue.game = self
        blue.enemy = red
        self.red = red
        red.game = self
        red.enemy = blue
        self.restore_defaults()

    def restore_defaults(self):
        if self.blue is not None:
            self.blue.restore_defaults()
        if self.red is not None:
            self.red.restore_defaults()

    def still_playing(self):
        assert self is self.blue.GAME
        assert self is self.red.GAME
        for unit in self.blue.units:
            assert self is unit.GAME
        for unit in self.red.units:
            assert self is unit.GAME

NoGAME = Game()
GAME1 = Game()
GAME1.TIMESTEP = .1
GAME1.DEFAULT_ROAMING_RANDOM_PERTURBATION = 2.
GAME1.DEFAULT_FLY_SPEED = 5.
GAME1.DEFAULT_ROAM_SPEED = 2.
GAME1.DEFAULT_POINT_SOURCE_CONSTANT = 1.
GAME1.DEFAULT_RECEPTION_PROBABILITY_SLOPE = 10.
GAME1.restore_defaults()


##########
# ORDERS #
##########

class BlankOrder():  # Default values for orders
    def __init__(self, unit):
        self.unit = unit

    def restore_defaults(self):
        pass

class StationaryOrder(BlankOrder): pass

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

class CommOrder(ApproachingOrder): pass
class JammerOrder(ApproachingOrder): pass
class OccupyingTroopOrder(StationaryOrder): pass
class RoamingTroopOrder(ApproachingGaussianOrder): pass

# Might want to add flags to Order classes __init__, where appropriate (especially jam)
#        self.communicate = True
#        self.jam = True
#        self.occupy_roof = True
#        self.shoot = True

###########################
# FACTIONS -- BLUE or RED #
###########################

class Faction():
    def __init__(self, name, GAME):
        self.name = name
        self.enemy = None
        self.GAME = GAME
        self.units = []
        self.headquarters = None
        self.communication_network = []
        self.jamming_network = []

    # All methods until noted dont't have counterparts in units classes
    def add_headquarters(self, unit):
        self.add_unit(unit)
        self.headquarters = unit

    def add_unit(self, unit):
        unit.faction = self
        unit.regame(self.GAME)
        self.units.append(unit)

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

    def step(self, action):  #TODO NEED WORK!
        obs_list = []
        reward_list = []
        done_list = []
        info_list = []
        for i, unit in enumerate(self.units):
            obs_i, reward_i, done_i, info_i = unit.step(action[i])
            obs_list.append(obs_i)
            reward_list.append(reward_i)
            done_list = append(done_i)
            info_list = append(info_i)
        return obs_list, reward_list, done_list, info_list


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
        self.faction.add_unit_to_communication_network(self)  # self becomes "unit" inside faction

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
        self.on_roof = False
        self.x_ = 0.1
        self.y_ = 0.1
        self.delta_x = 0
        self.delta_y = 0
        self.vx = 0.
        self.vy = 0.
        self.asset_value = 0.

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

class Drone(Flying, Unit):  # Parent class of Comm and Jammer
    def __init__(self, GAME=None):
        super().__init__(GAME)
        self.name = 'DRONE'
        self.shot_down = False

    def move(self):
        self.plan_timestep_motion()
        self.fly()


class Comm(Communicating, Drone):
    def __init__(self, GAME=None):
        super().__init__(GAME)
        self.name = 'COMM'
        self.order = CommOrder(self)  # self is the second arg that becomes "unit" inside CommOrder.__init__(self, unit)
        self.asset_value = 1.

    def restore_unit_defaults(self):
        self.point_source_constant = self.GAME.DEFAULT_POINT_SOURCE_CONSTANT  # DEFAULT_POINT_SOURCE_CONSTANT is a global constant
        self.reception_probability_slope = self.GAME.DEFAULT_RECEPTION_PROBABILITY_SLOPE  # DEFAULT_RECEPTION_PROBABILITY_SLOPE is a global constant

    def implement_ceoi(self):
        self.add_self_to_communication_network()


class Jammer(Jamming, Drone):
    def __init__(self, GAME=None):
        super().__init__(GAME)
        self.name = 'JAMMER'
        self.order = JammerOrder(self)  # self is the second arg that becomes "unit" inside JammerOrder.__init__(self, unit)

    def restore_unit_defaults(self):
        self.point_source_constant = self.GAME.DEFAULT_POINT_SOURCE_CONSTANT  # DEFAULT_POINT_SOURCE_CONSTANT is a global constant

    def implement_ceoi(self):
        self.add_self_to_jamming_network()
        

class OccupyingTroop(Occupying, Communicating, Shooting, Unit):
    def __init__(self, GAME=None):
        super().__init__(GAME)
        self.name = 'OCCUPYING_TROOP'
        self.order = OccupyingTroopOrder(self)  # self is the second arg that becomes unit inside __init__
        self.asset_value = 10.

    def restore_unit_defaults(self):
        self.point_source_constant = self.GAME.DEFAULT_POINT_SOURCE_CONSTANT  # DEFAULT_POINT_SOURCE_CONSTANT is a global constant
        self.reception_probability_slope = self.GAME.DEFAULT_RECEPTION_PROBABILITY_SLOPE  # DEFAULT_RECEPTION_PROBABILITY_SLOPE is a global constant

    def implement_ceoi(self):
        self.add_self_to_communication_network()

    def post_timestep(self):
        self.shoot_enemy_drones()


class RoamingTroop(Roaming, Shooting, Unit):
    def __init__(self, GAME=None):
        super().__init__(GAME)
        self.name = 'ROAMING_TROOP'
        self.order = RoamingTroopOrder(self)  # self is the second arg that becomes unit inside __init__

    def move(self):
        self.plan_timestep_motion()
        self.roam()

    def post_timestep(self):
        self.shoot_enemy_drones()

