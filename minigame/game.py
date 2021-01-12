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
        self.blue = None
        self.red = None
        # CONSTANTS
        self.TIMESTEP = None
        self.DEFAULT_ROAMING_RANDOM_PERTURBATION = None
        self.DEFAULT_FLY_SPEED = None
        self.DEFAULT_POINT_SOURCE_CONSTANT = None
        self.DEFAULT_RECEPTION_PROBABILITY_SLOPE = None

    def add_blue(self, blue):
        assert blue is not self.red
        self.blue = blue
        blue.game = self
        self.restore_defaults()

    def add_red(self, red):
        assert red is not self.blue
        self.red = red
        red.game = self
        self.restore_defaults()

    def restore_defaults(self):
        if self.blue is not None:
            self.blue.restore_defaults()
        if self.red is not None:
            self.red.restore_defaults()

NoGAME = Game()
GAME1 = Game()
GAME1.TIMESTEP = .1
GAME1.DEFAULT_ROAMING_RANDOM_PERTURBATION = 2
GAME1.DEFAULT_FLY_SPEED = 3
GAME1.DEFAULT_POINT_SOURCE_CONSTANT = 1
GAME1.DEFAULT_RECEPTION_PROBABILITY_SLOPE = 10
GAME1.restore_defaults()


##########
# ORDERS #
##########

class BlankOrder():  # Default values for orders
    def __init__(self, unit):
        self.unit = unit
        self.destination_x = None
        self.destination_y = None
        self.asset_value = 0.

    def set_destination(self, dest_x, dest_y):
        self.destination_x = dest_x
        self.destination_y = dest_y

    def restore_defaults(self):
        pass

class CommOrder(BlankOrder):
    def __init__(self, unit):
        super().__init__(unit)
        self.asset_value = 1.

class JammerOrder(BlankOrder):
    def __init__(self, unit):
        super().__init__(unit)

class OccupyingTroopOrder(BlankOrder):
    def __init__(self, unit):
        super().__init__(unit)
        self.asset_value = 10.
        self.occupy_roof = True

class RoamingTroopOrder(BlankOrder):
    def __init__(self, unit):
        super().__init__(unit)
        self.occupy_roof = False

    def restore_defaults(self):
        self.roaming_random_perturbation = self.unit.GAME.DEFAULT_ROAMING_RANDOM_PERTURBATION
############
# FACTIONS #
############

class Faction():  # BLUE OR RED
    def __init__(self, name, GAME):
        self.name = name
        self.GAME = GAME
        self.units = []
        self.headquarters = None
        self.communication_network = []
        self.jamming_network = []

    def add_headquarters(self, unit):
        self.add_unit(unit)
        self.headquarters = unit

    def add_unit(self, unit):
        unit.faction = self
        unit.regame(self.GAME)
        self.units.append(unit)

    def add_unit_to_communication_network(self, unit):
        self.communication_network.append(unit)

    def add_unit_to_jamming_network(self, unit):
        self.jamming_network.append(unit)

    def restore_defaults(self):
        for u in self.units:
            u.restore_defaults()

    def pop_unit(self):
        unit = self.units.pop()
        unit.faction = None

    def clear_units(self):
        for u in self.units:
            u.faction = None
        self.units = []

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
            self.vx = ideal_delta_x * self.max_speed/ideal_speed
            self.vy = ideal_delta_y * self.max_speed/ideal_speed

    def restore_capability_defaults(self):
        self.max_speed = self.GAME.DEFAULT_FLY_SPEED

class Flying(Moving):
    def fly(self): # overload this method
        self.x_ += self.delta_x
        self.y_ += self.delta_y

    def distance_to_target(self):  # l2 distance with flying, shouldn't have both Roaming and Flying capabilities
        return np.sqrt((self.order.destination_x - self.x_)**2 + (self.order.destination_y - self.y_)**2)

class Roaming(Moving):
    def traverse_roads_to_random_spot(self):
        #TODO Add Randomization
        self.x_ += self.delta_x
        self.y_ += self.delta_y

    def distance_to_target(self):  # l1 because must traverse regular road network aligned NS/EW, shouldn't have both Roaming and Flying capabilities
        return np.abs(self.order.destination_x - self.x_) + np.abs(self.order.destination_y - self.y_)

class Occupying():
    def place_on_target(self):
        self.x_ = self.order.destination_x
        self.y_ = self.order.destination_y

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
    def __init__(self):
        self.order = BlankOrder(self)  # self is the second arg that becomes unit inside __init__
        self.regame(NoGAME)  # defines self.GAME as NoGAME and calls self.restore_defaults()
        self.faction = None
        self.name = 'GHOST'
        self.on_roof = False
        self.x_ = 0.1
        self.y_ = 0.1
        self.delta_x = 0
        self.delta_y = 0
        self.vx = 0.
        self.vy = 0.

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


class Drone(Flying, Unit):  # Parent class of Comm and Jammer
    def __init__(self):
        super().__init__()
        self.name = 'DRONE'
        self.shot_down = False

    def move(self):
        self.plan_timestep_motion()
        self.fly()


class Comm(Communicating, Drone):
    def __init__(self):
        super().__init__()
        self.name = 'COMM'
        self.order = CommOrder(self)  # self is the second arg that becomes "unit" inside CommOrder.__init__(self, unit)

    def restore_unit_defaults(self):
        self.point_source_constant = self.GAME.DEFAULT_POINT_SOURCE_CONSTANT  # DEFAULT_POINT_SOURCE_CONSTANT is a global constant
        self.reception_probability_slope = self.GAME.DEFAULT_RECEPTION_PROBABILITY_SLOPE  # DEFAULT_RECEPTION_PROBABILITY_SLOPE is a global constant

    def implement_ceoi(self):
        self.add_self_to_communication_network()


class Jammer(Flying, Jamming, Unit):
    def __init__(self):
        super().__init__()
        self.name = 'JAMMER'
        self.order = JammerOrder(self)  # self is the second arg that becomes "unit" inside JammerOrder.__init__(self, unit)

    def restore_unit_defaults(self):
        self.point_source_constant = self.GAME.DEFAULT_POINT_SOURCE_CONSTANT  # DEFAULT_POINT_SOURCE_CONSTANT is a global constant

    def implement_ceoi(self):
        self.add_self_to_jamming_network()
        

class OccupyingTroop(Occupying, Communicating, Shooting, Unit):
    def __init__(self):
        super().__init__()
        self.name = 'OCCUPYING_TROOP'
        self.order = OccupyingTroopOrder(self)  # self is the second arg that becomes unit inside __init__

        def restore_unit_defaults(self):
            self.point_source_constant = self.GAME.DEFAULT_POINT_SOURCE_CONSTANT  # DEFAULT_POINT_SOURCE_CONSTANT is a global constant
            self.reception_probability_slope = self.GAME.DEFAULT_RECEPTION_PROBABILITY_SLOPE  # DEFAULT_RECEPTION_PROBABILITY_SLOPE is a global constant

        def initialize(self):
            self.place_on_target()

        def implement_ceoi(self):
            self.add_self_to_communication_network()

        def post_timestep(self):
            self.shoot_enemy_drones()


class RoamingTroop(Roaming, Shooting, Unit):
    def __init__(self):
        super().__init__()
        self.name = 'ROAMING_TROOP'
        self.order = RoamingTroopOrder(self)  # self is the second arg that becomes unit inside __init__

        def move(self):
            self.plan_timestep_motion, self.unit.traverse_roads_to_random_spot()

        def post_timestep(self):
            self.shoot_enemy_drones()

