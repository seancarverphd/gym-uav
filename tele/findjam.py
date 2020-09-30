import numpy as np
import scipy.special

class Jams():
    def __init__(self, ngrid=5, slope=1.):
        self.ngrid = ngrid
        self.slope = slope
        self.hq = (0,0)
        self.asset = (ngrid-1,ngrid-1)
        self.teleport_comm()
        self.teleport_jammer()
        self.jx = np.arange(5).reshape(1,5)*np.ones((5,1))
        self.jy = np.arange(5).reshape(5,1)*np.ones((1,5))
        # All distributions are represented as logs for stability
        self.logPjammer_prior = np.ones((ngrid, ngrid))*(-2.)*np.log(ngrid) # logProb(jammer@loc); init to uniform

    def teleport_comm(self):
        self.comm = self.teleport(self.ngrid)

    def teleport_jammer(self):
        self.jammer = self.teleport(self.grid) 

    def teleport(self, ngrid):
        return (np.random.choice(ngrid), np.random.choice(ngrid))

    def logsig(self, x):
        return np.log(scipy.special.expit(2*self.slope*x))

    def loglikelihood(self):
        return self.logsig(self.sig(np.sqrt((self.jx - self.comm[0])**2 + (self.jy - self.comm[1])**2)))

    def run(self, steps=3):
        for _ in range(steps):
            logPjammer_unnormalized = self.loglikelihood() + self.logPjammer_prior
            self.logPjammer_prior = scipy.special.log_softmax(logPjammer_unnormalized)  # Prior updated to Posterior
            self.render()
            self.teleport_comm()

    def render(self):
        pass
