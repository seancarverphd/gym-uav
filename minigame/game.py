import numpy as np
import torch

class Orders():
    def __init__(self):
        self.unit = None
        self.destination_x = None
        self.destination_y = None
        self.acceptable_delta = 0.
        self.asset_value = 0.
        self.move_commands = None
        self.after_timestep_commands = None
        self.ceoi = None

    def set_destination(self):
        self.

class HeadquarterOrder(Orders):
    def __init__(self, unit):
        super(HeadquarterOrder, self).__init__()
        self.unit = unit
        self.move_commands = [unit.stay]
        self.after_timestep_commands = [unit.shoot_enemy_drones]
        self.ceoi = None

class OccupyingTroopOrder(Orders):
    def __init__(self, unit):
        super(OccupyingTroopOrder, self).__init__()
        self.unit = unit
        self.move_commands = [unit.stay]
        self.after_timestep_commands = [unit.shoot_enemy_drones]
        self.ceoi = None

class CommsOrder(Orders):
    def __init__(self, unit):
        super(CommsOrder, self).__init__()
        self.unit = unit
        self.move_commands = [unit.fly]
        self.after_timestep_commands = [unit.stay]
        self.ceoi = None

class JammersOrder(Orders):
    def __init__(self, unit):
        super(JammersOrder, self).__init__()
        self.unit = unit
        self.move_commands = [unit.fly]
        self.after_timestep_commands = [unit.stay]
        self.ceoi = None

class RoamingTroopsOrder(Orders):
    def __init__(self, unit):
        super(RoamingTroopsOrder, self).__init__()
        self.unit = unit
        move_commands = [unit.traverse_roads_to_random_spot]
        after_timestep_commands = [unit.shoot_enemy_drones]

class Faction():
    def __init__(self, name):
        self.name = name
        self.units = []

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
        self.name = name
        self.receiver = None
        self.transmitter = None
        # self.can_do = Capabilities()
        self.headquarters = False
        self.x_ = init_x
        self.y_ = init_y
        self.on_roof = False

    def x(self):
        return x_

    def y(self):
        return y_

    def xy(self):
        return (x_, y_)

    def stay(self):
        pass

    def after(self):
        pass

# Individual Units
# Drone's child objects are COMMS and JAMMERS; first communicates, second jams, defined by children
# receiver, sender defined by children

# faction added later

class Drone(Unit):  # UAV
    def __init__(self, init_x=0.1, init_y=0.1, name='DRONE', max_speed=1.):
        super(Drone, self).__init__(init_x, init_y, name)
        self.max_speed = max_speed
        self.vx = 0.
        self.vy = 0.

    def fly(self, time_step=1): # overload this method
        over = np.sqrt(self.vx**2 + self.vy**2) / self.max_speed
        if over > 1:
            self.vx /= over
            self.vy /= over
        self.x_ = self.x_ + self.vx * time_step
        self.y_ = self.y_ + self.vy * time_step

class Jammer(Drone):
    def __init__(self, init_x=0.1, init_y=0.1), name='JAMMER', max_speed=1., jamming_antenna=None):
        super(Jammer, self).__init__(init_x, init_y, name, max_speed)


class Comm(UAV):
    pass


class OccupyingTroop(Unit):
    def at_destination(self):
        assert self.x() == self.order.destination_x
        assert self.y() == self.order.destination_y


class RoamingTroop(Unit):
    def at_destination(self):
        assert self.x() == self.order.destination_x
        assert self.y() == self.order.destination_y


class Headquarters(Unit):
    def __init__(self, init_x=0.1, init_y=0.1, name='HEADQUARTERS'):
        super(Headquarters, self).__init__(init_x, init_y, name)
        self.receiver = None  #TODO
        self.sender = None  #TODO
        self.order = None

# class Building(): pass # For now, buildings at every location




