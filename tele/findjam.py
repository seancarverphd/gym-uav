import numpy as np

class Jams():
    def __init__(self):
        self.hq = np.array((0,0))
        self.asset = np.array((4,4))
        self.comm = self.teleport()
        self.jammer = self.teleport() 

    def teleport(self):
        return np.array((np.random.choice(3)+1, np.random.choice(3)+1)) 
