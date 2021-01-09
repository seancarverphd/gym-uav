import numpy as np
import torch

TIME_STEP = 1
DEFAULT_ROAMING_RANDOM_PERTURBATION = 2
DEFAULT_FLY_SPEED = 3
DEFAULT_POINT_SOURCE_CONSTANT = 1
DEFAULT_RECEPTION_PROBABILITY_SLOPE = 10


class BlankOrder():
    def __init__(self, unit, dest_x=None, dest_y=None):
        self.unit = unit
        self.destination_x = dest_x
        self.destination_y = dest_y
        self.asset_value = 0.
        self.initial_commands = [unit.stay]
        self.ceoi = [unit.stay]
        self.move_commands = [unit.stay]
        self.post_timestep_commands =  [unit.stay]

    def set_destination(self, dest_x, dest_y):
        self.destination_x = dest_x
        self.destination_y = dest_y
        


class CommsOrder(BlankOrder):
    def __init__(self, unit, dest_x, dest_y):
        super(CommsOrder, self).__init__(unit, dest_x, dest_y)
        self.initial_commands = [unit.stay]
        self.ceoi = [unit.add_self_to_communication_network]
        self.move_commands = [unit.plan, unit.fly]
        self.post_timestep_commands = [unit.stay]
        self.asset_value = 1.


class JammersOrder(BlankOrder):
    def __init__(self, unit, dest_x, dest_y):
        super(JammersOrder, self).__init__(unit, dest_x, dest_y)
        self.ceoi = [unit.add_self_to_jamming_network]
        self.move_commands = [unit.plan, unit.fly]


class OccupyingTroopOrder(BlankOrder):
    def __init__(self, unit, dest_x, dest_y):
        super(OccupyingTroopOrder, self).__init__(unit, dest_x, dest_y)
        self.initial_commands = [unit.place_on_target]
        self.ceoi = [unit.add_self_to_communication_network]
        self.post_timestep_commands = [unit.shoot_enemy_drones]
        self.occupy_roof = True
        self.asset_value = 10.


class RoamingTroopsOrder(BlankOrder):
    def __init__(self, unit, dest_x, dest_y):
        super(RoamingTroopsOrder, self).__init__(unit, dest_x, dest_y)
        self.move_commands = [unit.traverse_roads_to_random_spot]
        self.post_timestep_commands = [unit.shoot_enemy_drones]
        self.random_perturbation = DEFAULT_ROAMING_RANDOM_PERTURBATION 
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
    def __init__(self, init_x=0.1, init_y=0.1, name='GHOST'):
        self.faction = None
        self.init_x = init_x
        self.init_y = init_y
        self.name = name
        self.order = None
        self.reset_xy(self.init_x, self.init_y)
        self.on_roof = False

    def give_order(self, order=None):

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
    def __init__(self, init_x=0.1, init_y=0.1, name='DRONE'):
        super(Drone, self).__init__(init_x, init_y, name)
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

class Jammer(Drone):
    def __init__(self, init_x=0.1, init_y=0.1, name='JAMMER'):
        super(Jammer, self).__init__(init_x, init_y, name)
        self.point_source_constant = DEFAULT_POINT_SOURCE_CONSTANT  # DEFAULT_POINT_SOURCE_CONSTANT is a global constant
        self.implement_ceoi()

    # inherits plan() from Drone
    # inherits fly() from Drone
    def add_self_to_jamming_network(self):
        self.faction.add_unit_to_jamming_network(self)


class Comm(Drone):
    def __init__(self, init_x=0.1, init_y=0.1, name='COMM'):
        super(Comm, self).__init__(init_x, init_y, name)
        self.point_source_constant = DEFAULT_POINT_SOURCE_CONSTANT  # DEFAULT_POINT_SOURCE_CONSTANT is a global constant
        self.reception_probability_slope = DEFAULT_RECEPTION_PROBABILITY_SLOPE  # DEFAULT_RECEPTION_PROBABILITY_SLOPE is a global constant
        self.implement_ceoi()

    # inherits plan()
    # inherits fly()
    def add_self_to_communication_network(self):
        self.faction.add_unit_to_communication_network(self)


class GroundTroop(Unit):
    def __init__(self, init_x=0.1, init_y=0.1, name='GROUND_TROOP'):
        super(GroundTroop, self).__init__(init_x, init_y, name)


class OccupyingTroop(GroundTroop):
    def __init__(self, init_x=None, init_y=None, name='OCCUPYING_TROOP'):
        super(OccupyingTroop, self).__init__(init_x, init_y, name)
        self.point_source_constant = DEFAULT_POINT_SOURCE_CONSTANT  # DEFAULT_POINT_SOURCE_CONSTANT is a global constant
        self.reception_probability_slope = DEFAULT_RECEPTION_PROBABILITY_SLOPE  # DEFAULT_RECEPTION_PROBABILITY_SLOPE is a global constant
        self.no_init_xy_passed()
        self.implement_ceoi()

    def add_self_to_communication_network(self):
        self.faction.add_unit_to_communication_network(self)

    def no_init_xy_passed(self):
        assert self.init_x is None  # instead xy defined by orders
        assert self.init_y is None  # instead xy defined by orders

class RoamingTroop(GroundTroop):
    def __init__(self, init_x=0.1, init_y=0.1, name='ROAMING_TROOP'):
        super(RoamingTroop, self).__init__(init_x, init_y, name)

