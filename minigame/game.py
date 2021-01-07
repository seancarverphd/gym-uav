import numpy as np
import torch

class Orders():
    def __init__(self):
        self.unit = None
        self.destination_x = None
        self.destination_y = None
        self.occupy_roof = False
        self.random_perturbation = 0.
        self.asset_value = 0.
        self.move_commands = None
        self.after_timestep_commands = None
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
        super(CommsOrder, self).__init__()
        self.unit = unit
        self.move_commands = [unit.fly]
        self.after_timestep_commands = [unit.stay]
        self.ceoi = None  #TODO Add CEOI

class JammersOrder(Orders):
    def __init__(self, unit):
        super(JammersOrder, self).__init__()
        self.unit = unit
        self.move_commands = [unit.fly]
        self.after_timestep_commands = [unit.stay]
        self.ceoi = None  #TODO Add CEOI

class OccupyingTroopOrder(Orders):
    def __init__(self, unit):
        super(OccupyingTroopOrder, self).__init__()
        self.unit = unit
        self.move_commands = [unit.stay]
        self.after_timestep_commands = [unit.shoot_enemy_drones]
        self.ceoi = None  #TODO Add CEOI

class RoamingTroopsOrder(Orders):
    def __init__(self, unit):
        super(RoamingTroopsOrder, self).__init__()
        self.unit = unit
        self.move_commands = [unit.traverse_roads_to_random_spot]
        self.after_timestep_commands = [unit.shoot_enemy_drones]
        # CEOI: Red Roaming Units don't communicate self.ceoi=None


class Faction():
    def __init__(self, name):
        self.name = name
        self.units = []
        self.headquarters = None

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
        self.name = name
        self.receiver = None
        self.transmitter = None
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




