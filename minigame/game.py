import numpy as np
import torch

class Capabilities():
    def __init__(self):
        self.capability = {}
        self.capability['communicate'] = False
        self.capability['jam'] = False
        self.capability['shoot'] = False
        self.capability['fly'] = False
        self.capability['traverse_roads'] = False
        self.capability['occupy_rooftop'] = False

    def add_capability(self, key):
        self.capability[key] = True

    def capable(self, order):
        for key in order.demands:
            if order.demands[key] and not self.capability[key]:
                return False
        return True

    def assert_capable(self, order)
        assert self.capable(order)

headquarters_can = Capabilities()
headquarters_can.add_capability('communicate')
headquarters_can.add_capability('shoot')
headquarters_can.add_capability('occupy_rooftop')

occupying_troops_can = Capabilities()
occupying_troops_can.add_capability('communicate')
occupying_troops_can.add_capability('shoot')
occupying_troops_can.add_capability('occupy_rooftop')

comms_can = Capabilities()
comms_can.add_capability('fly')
comms_can.add_capability('communicate')

jammers_can = Capabilities()
jammers_can.add_capability('fly')
jammers_can.add_capability('jam')

roaming_troops_can = Capabilities()
roaming_troops_can.add_capability('shoot')
roaming_troops_can.add_capability('traverse_roads')

headquarters_mission_demands = copy.deepcopy(headquarters_can)
occupying_troops_mission_demands = copy.deepcopy(occupying_troops_can)
comms_mission_demands = copy.deepcopy(comms_can)
jammers_mission_demands = copy.deepcopy(jammers_can)
roaming_troops_mission_demands = copy.deepcopy(roaming_troops_can)


class Orders():
    def __init__(self):
        destination_x = None
        destination_y = None
        acceptable_delta = 0.
        occupy_roof = False
        shoot_enemy_drones = False
        asset_value = 0.
        demands = Capabilities()


class Faction():
    def __init__(self, name):
        self.name = name
        self.units = []

    def add_unit(self, unit):
        self.units.append(unit)
        self.units[-1].faction = self

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
        self.can_do = Capabilities()
        self.on_roof = False
        self.headquarters = False
        self.x_ = init_x
        self.y_ = init_y

    def x(self):
        return x_

    def y(self):
        return y_

    def xy(self):
        return (x_, y_)

    def move(self):
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

    def move(): # overload this method
        pass  #TODO Add this method


class Jammer(UAV):
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




