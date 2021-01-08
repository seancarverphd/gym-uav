import numpy as np
import torch

TIME_STEP = 1
DEFAULT_FLY_SPEED = 2

class Orders():
    def __init__(self, unit):
        self.unit = unit
        self.destination_x = None
        self.destination_y = None
        self.occupy_roof = False
        self.random_perturbation = 0.
        self.asset_value = 0.
        self.move_commands = None
        self.post_timestep_commands = None
        self.ceoi = None

    def set_destination(self, d):
        self.destination_x = d[0]
        self.destination_y = d[1]
        self.occupy_roof = d[3]
        self.random_perturbation = d[4]

    def set_asseet_value(self, d):
        self.asset_value = d

class CommsOrder(Orders):
    def __init__(self, unit):
        super(CommsOrder, self).__init__(unit)
        self.initial_commands = [unit.stay]
        self.ceoi = [unit.add_self_to_communication_network]
        self.move_commands = [unit.plan, unit.fly]
        self.post_timestep_commands = [unit.stay]

class JammersOrder(Orders):
    def __init__(self, unit):
        super(JammersOrder, self).__init__(unit)
        self.initial_commands = [unit.stay]
        self.ceoi = [unit.add_self_to_jamming_network]
        self.move_commands = [unit.plan, unit.fly]
        self.post_timestep_commands = [unit.stay]

class OccupyingTroopOrder(Orders):
    def __init__(self, unit):
        super(OccupyingTroopOrder, self).__init__(unit)
        self.initial_commands = [unit.on_target]
        self.ceoi = [unit.add_self_to_communication_network]
        self.move_commands = [unit.stay]
        self.post_timestep_commands = [unit.shoot_enemy_drones]

class RoamingTroopsOrder(Orders):
    def __init__(self, unit):
        super(RoamingTroopsOrder, self).__init__(unit)
        self.initial_commands = [unit.stay]
        self.ceoi = [unit.stay]  # Roaming Units dont communicate
        self.move_commands = [unit.traverse_roads_to_random_spot]
        self.post_timestep_commands = [unit.shoot_enemy_drones]


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
    def __init__(self, init_x=0.1, init_y=0.1, name='GHOST_X'):
        self.faction = None
        self.name = name
        self.receiver = None
        self.transmitter = None
        self.initialize_xy(init_x, init_y)
        self.on_roof = False

    def initialize_xy(self, init_x, init_y):
        self.x_ = init_x
        self.y_ = init_y

    def execute_initial_commands(self):
        for command in self.order.initial_commands:
            command()

    def implement_ceoi(self):
        for command in self.order.ceoi:
            command()

    def execute_move_commands(self):
        for command in self.order.move_commands:
            command()

    def execute_post_timestep_commands(self):
        for command in self.order.post_time_step_commands:
            command()

    def x(self):
        return self.x_

    def y(self):
        return self.y_

    def xy(self):
        return (self.x_, self.y_)

    def stay(self):
        pass

# Drone, OccupyingTroop, RoamingTroop Inherit From Unit
# Drone's child objects are COMMS and JAMMERS; first communicates, second jams, defined by children
# receiver, sender defined by children

# faction added later

class Drone(Unit):  # UAV
    def __init__(self, init_x=0.1, init_y=0.1, name='DRONE_X'):
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
    def __init__(self, init_x=0.1, init_y=0.1), name='JAMMER_X'):
        super(Jammer, self).__init__(init_x, init_y, name)
        self.point_source_constant = POINT_SOURCE_CONSTANT  # POINT_SOURCE_CONSTANT is a global constant
        for command in self.ceoi:
            command()  # self.add_self_to_jamming_network()

    # inherits plan()
    # inherits fly()
    def add_self_to_jamming_network(self):
        self.faction.add_unit_to_jamming_network(self)


class Comm(Drone):
    def __init__(self, init_x=0.1, init_y=0.1), name='COMM_X', max_speed=1., jamming_antenna=None):
        super(Comm, self).__init__(init_x, init_y, name)
        self.point_source_constant = POINT_SOURCE_CONSTANT  # POINT_SOURCE_CONSTANT is a global constant
        self.reception_probability_slope = RECEPTION_PROBABILITY_SLOPE  # RECEPTION_PROBABILITY_SLOPE is a global constant
        for command in self.ceoi:
            command()  # self.add_self_to_communication_network()

    # inherits plan()
    # inherits fly()
    def add_self_to_communication_network(self):
        self.faction.add_unit_to_communication_network(self)


class OccupyingTroop(Unit):
    def __init__(self, init_x=0.1, init_y=0.1), name='OCCUPYING_TROOP_X', max_speed=1., jamming_antenna=None):
        super(OccupyingTroop, self).__init__(init_x, init_y, name)
        self.point_source_constant = POINT_SOURCE_CONSTANT  # POINT_SOURCE_CONSTANT is a global constant
        self.reception_probability_slope = RECEPTION_PROBABILITY_SLOPE  # RECEPTION_PROBABILITY_SLOPE is a global constant
        for command in self.ceoi:
            command()  # self.add_to_communication_network()
        self.on_target()

    def on_target(self)
        assert self.x_ == order.destination_x
        assert self.y_ == order.destination_y

    def add_self_to_communication_network(self):
        self.faction.add_unit_to_communication_network(self)

class RoamingTroop(Unit):
    def __init__(self, init_x=0.1, init_y=0.1), name='ROAMING_TROOP_X', max_speed=1., jamming_antenna=None):
        super(OccupyingTroop, self).__init__(init_x, init_y, name)


