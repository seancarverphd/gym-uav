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
        super().__init__(unit)
        self.ceoi = [self.unit.add_self_to_communication_network]
        self.move_commands = [self.unit.plan_timestep_motion, unit.fly]
        self.asset_value = 1.


class JammerOrder(BlankOrder):
    def __init__(self, unit):
        super().__init__(unit)
        self.ceoi = [self.unit.add_self_to_jamming_network]
        self.move_commands = [self.unit.plan_timestep_motion, self.unit.fly]


class OccupyingTroopOrder(BlankOrder):
    def __init__(self, unit):
        super().__init__(unit)
        self.initial_commands = [self.unit.place_on_target]
        self.ceoi = [self.unit.add_self_to_communication_network]
        self.post_timestep_commands = [self.unit.shoot_enemy_drones]
        self.occupy_roof = True
        self.asset_value = 10.


class RoamingTroopOrder(BlankOrder):
    def __init__(self, unit):
        super().__init__(unit)
        self.move_commands = [self.unit.plan_timestep_motion, self.unit.traverse_roads_to_random_spot]
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
        # Below might be defined in a Movable_Unit class but doing so would create a "diamond" problem of multiple inheritance, difficult to debug
        self.max_speed = DEFAULT_FLY_SPEED
        self.delta_x = 0
        self.delta_y = 0
        self.vx = 0.
        self.vy = 0.

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

    def initialize(self):
        for command in self.order.initial_commands:
            command()

    def implement_ceoi(self):
        for command in self.order.ceoi:
            command()

    def move(self):
        for command in self.order.move_commands:
            command()

    # Below might be defined in a Movable_Unit class but doing so would create a "diamond" problem of multiple inheritance, difficult to debug

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


class Moving():
    def plan_timestep_motion(self):   # Used for Comm, Jammer & RoamingTroop but NOT Occupying Troop
        '''
        plan_timestep_motion(): defines delta_x, delta_y, vx, vy in direction of destination but magnitude not greater than self.max_speed
        '''
        desired_speed = self.distance_to_target() / TIMESTEP  # distance to target l2 for Drone, l1 for RoamingTroop
        if desired_speed > max_speed:
            reduction = max_speed/desired_speed
        else:
            reduction = 1
        self.delta_x = reduction*(self.order.destination_x - self.x_)
        self.delta_y = reduction*(self.order.destination_y - self.y_)
        self.vx = delta_x / TIMESTEP # TIME_STEP is a global constant
        self.vy = delta_y / TIMESTEP  # TIME_STEP is a global constant


class Flying(Moving):
    def fly(self): # overload this method
        self.x_ += self.delta_x
        self.y_ += self.delta_y

    def distance_to_target(self):
        return np.sqrt((self.order.destination_x - self.x_)**2 + (self.order.destination_y - self_y)**2)

class Roaming(Moving):
    def distance_to_target(self):  # l1 because must traverse regular road network aligned NS/EW
        return np.abs(self.order.destination_x - self.x_) + np.abs(self.order.destination_y - self_y)

class Occupying():
    def place_on_target(self):
        self.x_ = order.destination_x
        self.y_ = order.destination_y


class Communicating():
    def add_self_to_communication_network(self):
        self.faction.add_unit_to_communication_network(self)

class Jamming():
    def add_self_to_jamming_network(self):
        self.faction.add_unit_to_jamming_network(self)

class Shooting():
    def shoot_enemy_drones(self):
        pass  #TODO Add this function


class Comm(Unit, Flying, Communicating):
    def __init__(self):
        super(Comm, self).__init__()
        self.name = 'COMM'
        self.order = CommOrder(self)  # self is the second arg that becomes unit inside __init__
        self.point_source_constant = DEFAULT_POINT_SOURCE_CONSTANT  # DEFAULT_POINT_SOURCE_CONSTANT is a global constant
        self.reception_probability_slope = DEFAULT_RECEPTION_PROBABILITY_SLOPE  # DEFAULT_RECEPTION_PROBABILITY_SLOPE is a global constant


class Jammer(Unit, Flying, Jamming):
    def __init__(self):
        super().__init__()
        self.name = 'JAMMER'
        self.order = JammerOrder(self)  # self is the second arg that becomes unit inside __init__
        self.point_source_constant = DEFAULT_POINT_SOURCE_CONSTANT  # DEFAULT_POINT_SOURCE_CONSTANT is a global constant

class OccupyingTroop(Unit, Occupying, Communicating, Shooting):
    def __init__(self):
        super().__init__()
        self.name = 'OCCUPYING_TROOP'
        self.order = OccupyingTroopOrder(self)  # self is the second arg that becomes unit inside __init__
        self.point_source_constant = DEFAULT_POINT_SOURCE_CONSTANT  # DEFAULT_POINT_SOURCE_CONSTANT is a global constant
        self.reception_probability_slope = DEFAULT_RECEPTION_PROBABILITY_SLOPE  # DEFAULT_RECEPTION_PROBABILITY_SLOPE is a global constant

class RoamingTroop(Unit, Roaming, Shooting):
    def __init__(self):
        super().__init__()
        self.name = 'ROAMING_TROOP'
        self.order = RoamingTroopOrder(self)  # self is the second arg that becomes unit inside __init__

    def traverse_roads_to_random_spot(self):
        self.x_ += self.delta_x
        self.y_ += self.delta_y
        #TODO Add Randomization


