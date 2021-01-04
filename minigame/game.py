import numpy as np
import torch

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
    def __init__(self, init_x, init_y):
        self.faction = None
        self.receiver = None
        self.transmitter = None
        self.name = 'GHOST'
        self.communicates = False
        self.jams = False
        self.shoots = False
        self.flies = False
        self.traverses_roads = False
        self.occupies_roofs = False
        self.on_roof = False
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


class Order():
    def __init__(self):
        destination = None
        occupy_roof = False
        shoot_enemy_drones = False

