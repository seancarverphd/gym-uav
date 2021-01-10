import numpy as np
import torch

TIME_STEP = 1
DEFAULT_ROAMING_RANDOM_PERTURBATION = 2
DEFAULT_FLY_SPEED = 3
DEFAULT_POINT_SOURCE_CONSTANT = 1
DEFAULT_RECEPTION_PROBABILITY_SLOPE = 10


class BlankOrder():  # Default values for orders
    def __init__(self, unit):
        self.unit = unit
        self.destination_x = None
        self.destination_y = None
        self.initial_commands = [self.unit.stay]
        self.ceoi = [unit.stay]
        self.move_commands = [self.unit.stay]
        self.post_timestep_commands = [self.unit.stay]
        self.asset_value = 0.

    def set_destination(self, dest_x, dest_y):
        self.destination_x = dest_x
        self.destination_y = dest_y
        


class CommOrder(BlankOrder):
    def __init__(self, unit):
        super(CommOrder, self).__init__(unit)
        self.ceoi = [self.unit.add_self_to_communication_network]
        self.move_commands = [self.unit.plan, unit.fly]
        self.asset_value = 1.


class JammerOrder(BlankOrder):
    def __init__(self, unit):
        super(JammerOrder, self).__init__(unit)
        self.ceoi = [self.unit.add_self_to_jamming_network]
        self.move_commands = [self.unit.plan, self.unit.fly]


class OccupyingTroopOrder(BlankOrder):
    def __init__(self, unit):
        super(OccupyingTroopOrder, self).__init__(unit)
        self.initial_commands = [self.unit.place_on_target]
        self.ceoi = [self.unit.add_self_to_communication_network]
        self.post_timestep_commands = [self.unit.shoot_enemy_drones]
        self.occupy_roof = True
        self.asset_value = 10.


class RoamingTroopOrder(BlankOrder):
    def __init__(self, unit):
        super(RoamingTroopOrder, self).__init__(unit)
        self.move_commands = [self.unit.traverse_roads_to_random_spot]
        self.post_timestep_commands = [self.unit.shoot_enemy_drones]
        self.roaming_random_perturbation = DEFAULT_ROAMING_RANDOM_PERTURBATION
        self.occupy_roof = False


class Faction():  # BLUE OR RED
    def __init__(self, name):
        self.name = name
        self.units = []
        self.headquarters = None
        self.communication_network = []
        self.jamming_network = []

    def add_headquarters(self, unit):
        self.add_unit(unit)
        self.headquarters = unit

    def add_unit(self, unit):
        self.units.append(unit)
        self.units[-1].faction = self

    def pop_unit(self):
        unit = self.units.pop()
        unit.faction = None

    def clear_units(self):
        for u in self.units:
            u.faction = None
        self.units = []


class Unit():
    def __init__(self):
        self.faction = None
        self.name = 'GHOST'
        self.order = BlankOrder(self)  # self is the second arg that becomes unit inside __init__
        self.on_roof = False

    def set_name(self, name):
        self.name = name

    def set_initial(self, init_x, init_y):
        self.init_x = init_x
        self.init_y = init_y

    def set_destination(self, dest_x, dest_y):
        self.order.set_destination(dest_x, dest_y)

    def reset_xy(self, init_x, init_y):
        self.x_ = init_x
        self.y_ = init_y

    def place_on_target(self):
        self.x_ = order.destination_x
        self.y_ = order.destination_y

    def initialize(self):
        for command in self.order.initial_commands:
            command()

    def implement_ceoi(self):
        for command in self.order.ceoi:
            command()

    def move(self):
        for command in self.order.move_commands:
            command()

    def post_timestep(self):
        for command in self.order.post_time_step_commands:
            command()

    def stay(self):
        pass

    def x(self):
        return self.x_

    def y(self):
        return self.y_

    def xy(self):
        return (self.x_, self.y_)

# Drone, OccupyingTroop, RoamingTroop Inherit From Unit
# Drone's child objects are COMMS and JAMMERS; first communicates, second jams, defined by children
# receiver, sender defined by children

# faction added later

class Drone(Unit):  # UAV
    def __init__(self):
        super(Drone, self).__init__()
        self.name = 'DRONE'
        self.max_speed = DEFAULT_FLY_SPEED 
        self.vx = 0.
        self.vy = 0.

    def plan(self):
        '''
        plan(): defines vx and vy in direction of destination but magnitude not greater than self.max_speed
        '''
        delta_x = self.order.destination_x - self.x_
        delta_y = self.order.destination_y - self.y_
        self.vx = delta_x/TIME_STEP
        self.vy = delta_y/TIME_STEP
        over = np.sqrt(self.vx**2 + self.vy**2) / self.max_speed
        if over > 1:
            self.vx /= over
            self.vy /= over

    def fly(self): # overload this method
        self.x_ = self.x_ + self.vx * TIME_STEP  # TIME_STEP is a global constant
        self.y_ = self.y_ + self.vy * TIME_STEP  # TIME_STEP is a global constant


    def l2_distance_to_target(self):
        return np.sqrt((self.order.destination_x - self.x_)**2 + (self.order.destination_y - self_y)**2)

class Comm(Drone):
    def __init__(self):
        super(Comm, self).__init__()
        self.name = 'COMM'
        self.order = CommOrder(self)  # self is the second arg that becomes unit inside __init__
        self.point_source_constant = DEFAULT_POINT_SOURCE_CONSTANT  # DEFAULT_POINT_SOURCE_CONSTANT is a global constant
        self.reception_probability_slope = DEFAULT_RECEPTION_PROBABILITY_SLOPE  # DEFAULT_RECEPTION_PROBABILITY_SLOPE is a global constant

    # inherits plan()
    # inherits fly()
    def add_self_to_communication_network(self):
        self.faction.add_unit_to_communication_network(self)


class Jammer(Drone):
    def __init__(self):
        super(Jammer, self).__init__()
        self.name = 'JAMMER'
        self.order = JammerOrder(self)  # self is the second arg that becomes unit inside __init__
        self.point_source_constant = DEFAULT_POINT_SOURCE_CONSTANT  # DEFAULT_POINT_SOURCE_CONSTANT is a global constant

    # inherits plan() from Drone
    # inherits fly() from Drone
    def add_self_to_jamming_network(self):
        self.faction.add_unit_to_jamming_network(self)


class GroundTroop(Unit):
    def __init__(self):
        super(GroundTroop, self).__init__()
        self.name = 'GROUND_TROOP'

    def shoot_enemy_drones(self):
        pass  #TODO Add this function

class OccupyingTroop(GroundTroop):
    def __init__(self):
        super(OccupyingTroop, self).__init__()
        self.name = 'OCCUPYING_TROOP'
        self.order = OccupyingTroopOrder(self)  # self is the second arg that becomes unit inside __init__
        self.point_source_constant = DEFAULT_POINT_SOURCE_CONSTANT  # DEFAULT_POINT_SOURCE_CONSTANT is a global constant
        self.reception_probability_slope = DEFAULT_RECEPTION_PROBABILITY_SLOPE  # DEFAULT_RECEPTION_PROBABILITY_SLOPE is a global constant

    def add_self_to_communication_network(self):
        self.faction.add_unit_to_communication_network(self)


class RoamingTroop(GroundTroop):
    def __init__(self):
        super(RoamingTroop, self).__init__()
        self.name = 'ROAMING_TROOP'
        self.order = RoamingTroopOrder(self)  # self is the second arg that becomes unit inside __init__

    def traverse_roads_to_random_spot(self):
        pass  #TODO Add this function

    def l1_distance_to_target(self):
        return np.abs(self.order.destination_x - self.x_) + np.abs(self.order.destination_y - self_y)

