import numpy as np

class Jams():
    def __init__(self, ngrid):
        self.ngrid = ngrid
        self.hq = (-1,-1)
        self.asset = (ngrid,ngrid)
        self.teleport_comm()
        self.teleport_jammer()
        Pjammer = np.array((ngrid, ngrid))
        PjamGjammer = np.array((ngrid,ngrid))

    def teleport_comm(self):
        self.comm = self.teleport(self.ngrid)

    def teleport_jammer(self)
        self.jammer = self.teleport(self.grid) 

    def teleport(self, ngrid):
        return (np.random.choice(ngrid), np.random.choice(ngrid))
