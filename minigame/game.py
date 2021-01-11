import numpy as np
import torch

# A Game() class will hold these constants, eventually
# Plus each faction and more.

class Game():
    def __init__(self):
        self.blue = None
        self.red = None
        # CONSTANTS
        self.TIMESTEP = .1
        self.DEFAULT_ROAMING_RANDOM_PERTURBATION = 2
        self.DEFAULT_FLY_SPEED = 3
        self.DEFAULT_POINT_SOURCE_CONSTANT = 1
        self.DEFAULT_RECEPTION_PROBABILITY_SLOPE = 10

    def add_factions(self, blue, red):
        assert blue is not red
        self.blue = blue
        self.red = red
        blue.game = self
        red.game = self

NoGame = Game()


########################################################################################
# CLASSES:                                                                             #
#  There are classes for                                                               #
#     * Orders for each unit type                                                      #
#     * Faction (eg Blue or Red)                                                       #
#     * Capabilities (Communicating, Jamming, Flying, Roaming, Occupying, Shooting)    #
#     * Units (Comm, Jammer, Occupying_Troop, Roaming_Troop)                           #
#     * Parent classes (Unit, Drone, Moving, BlankOrder)                               #
########################################################################################

##########
# ORDERS #
##########

class BlankOrder():  # Default values for orders
    def __init__(self, unit, G):
        self.G = G
        self.unit = unit
        self.destination_x = None
        self.destination_y = None
        self.asset_value = 0.

    def set_destination(self, dest_x, dest_y):
        self.destination_x = dest_x
        self.destination_y = dest_y

class CommOrder(BlankOrder):
    def __init__(self, unit, G):
        super().__init__(unit, G)
        self.G = G
        self.asset_value = 1.

class JammerOrder(BlankOrder):
    def __init__(self, unit, G):
        super().__init__(unit, G)
        self.G = G

class OccupyingTroopOrder(BlankOrder):
    def __init__(self, unit, G):
        super().__init__(unit, G)
        self.G = G
        self.asset_value = 10.
        self.occupy_roof = True

class RoamingTroopOrder(BlankOrder):
    def __init__(self, unit, G):
        super().__init__(unit, G)
        self.G = G
        self.roaming_random_perturbation = self.G.DEFAULT_ROAMING_RANDOM_PERTURBATION
        self.occupy_roof = False

############
# FACTIONS #
############

class Faction():  # BLUE OR RED
    def __init__(self, name, G):
        self.name = name
        self.G = G
        self.game = None
        self.units = []
        self.headquarters = None
        self.communication_network = []
        self.jamming_network = []

    def add_headquarters(self, unit):
        self.add_unit_to_faction(unit)
        self.headquarters = unit

    def add_unit_to_faction(self, unit):
        self.units.append(unit)
        self.units[-1].faction = self

    def add_unit_to_communication_network(self, unit):
        self.communication_network.append(unit)

    def add_unit_to_jamming_network(self, unit):
        self.jamming_network.append(unit)

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
        ideal_speed = self.distance_to_target() / self.G.TIMESTEP  # distance to target l2 for Flying, l1 for Roaming
        if ideal_speed <= self.max_speed:  # not too fast
            self.delta_x = ideal_delta_x
            self.delta_y = ideal_delta_y
        else:  # too fast
            self.vx = ideal_delta_x * self.max_speed/ideal_speed
            self.vy = ideal_delta_y * self.max_speed/ideal_speed

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
    def __init__(self, G):
        self.G = G
        self.faction = None
        self.regame(NoGame)  # defines self.G as NoGame and calls self.restore_defaults()
        self.name = 'GHOST'
        self.order = BlankOrder(self, G)  # self is the second arg that becomes unit inside __init__
        self.on_roof = False
        self.max_speed = self.G.DEFAULT_FLY_SPEED
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

    def regame(self, G):
        self.G = G
        self.restore_defaults()

    def restore_defaults(self):
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


class Drone(Unit, Flying):  # Parent class of Comm and Jammer
    def __init__(self, G):
        super().__init__(G)
        self.name = 'DRONE'
        self.shot_down = False

    def move(self):
        self.plan_timestep_motion()
        self.fly()


class Comm(Drone, Communicating):
    def __init__(self, G):
        super().__init__(G)
        self.name = 'COMM'
        self.order = CommOrder(self, G)  # self is the second arg that becomes "unit" inside CommOrder.__init__(self, unit)

    def restore_defaults(self):
        self.point_source_constant = self.G.DEFAULT_POINT_SOURCE_CONSTANT  # DEFAULT_POINT_SOURCE_CONSTANT is a global constant
        self.reception_probability_slope = self.G.DEFAULT_RECEPTION_PROBABILITY_SLOPE  # DEFAULT_RECEPTION_PROBABILITY_SLOPE is a global constant

    def implement_ceoi(self):
        self.add_self_to_communication_network()


class Jammer(Unit, Flying, Jamming):
    def __init__(self, G):
        super().__init__(G)
        self.name = 'JAMMER'
        self.order = JammerOrder(self, G)  # self is the second arg that becomes "unit" inside JammerOrder.__init__(self, unit)

    def restore_defaults(self):
        self.point_source_constant = self.G.DEFAULT_POINT_SOURCE_CONSTANT  # DEFAULT_POINT_SOURCE_CONSTANT is a global constant

    def implement_ceoi(self):
        self.add_self_to_jamming_network()
        

class OccupyingTroop(Unit, Occupying, Communicating, Shooting):
    def __init__(self, G):
        super().__init__(G)
        self.name = 'OCCUPYING_TROOP'
        self.order = OccupyingTroopOrder(self, G)  # self is the second arg that becomes unit inside __init__

        def restore_defaults(self):
            self.point_source_constant = self.G.DEFAULT_POINT_SOURCE_CONSTANT  # DEFAULT_POINT_SOURCE_CONSTANT is a global constant
            self.reception_probability_slope = self.G.DEFAULT_RECEPTION_PROBABILITY_SLOPE  # DEFAULT_RECEPTION_PROBABILITY_SLOPE is a global constant

        def initialize(self):
            self.place_on_target()

        def implement_ceoi(self):
            self.add_self_to_communication_network()

        def post_timestep(self):
            self.shoot_enemy_drones()


class RoamingTroop(Unit, Roaming, Shooting):
    def __init__(self, G):
        super().__init__(G)
        self.name = 'ROAMING_TROOP'
        self.order = RoamingTroopOrder(self, G)  # self is the second arg that becomes unit inside __init__

        def move(self):
            self.plan_timestep_motion, self.unit.traverse_roads_to_random_spot()

        def post_timestep(self):
            self.shoot_enemy_drones()

