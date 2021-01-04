import numpy as np
import torch

class Faction():
    def __init__(self, name):
        self.name = name

class Unit():
    def __init__(self):
        self.faction = None
        self.receiver = None
        self.transmitter = None
        self.name = 'GHOST'
        self.communicates = False
        self.jams = False
        self.flies = False
        self.traverses_roads = False
        self.occupying_building = False
        self.shoots_drones = False
        self.x_ = 0.
        self.y_ = 0.

    def x(self):
        return x_

    def y(self):
        return y_

    def xy(self):
        return (x_, y_)

    def move(self):
        pass



         
