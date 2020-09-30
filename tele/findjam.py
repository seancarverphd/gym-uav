import numpy as np
import scipy.special

class Jams():
    def __init__(self, ngrid=5, slope=1.):
        self.ngrid = ngrid
        self.slope = slope
        self.hq = (0,0)
        self.asset = (ngrid-1,ngrid-1)
        self.comm = None
        self.jammer = None
        self.teleport_comm()
        self.teleport_jammer()
        self.jx = np.arange(5).reshape(1,5)*np.ones((5,1))
        self.jy = np.arange(5).reshape(5,1)*np.ones((1,5))
        # All distributions are represented as logs for stability
        self.logPjammer_prior = np.ones((ngrid, ngrid))*(-2.)*np.log(ngrid) # logProb(jammer@loc); init to uniform

    def teleport_comm(self):
        self.comm = self.teleport(self.ngrid)

    def teleport_jammer(self):
        self.jammer = self.teleport(self.ngrid) 

    def teleport(self, ngrid):
        return (np.random.choice(ngrid), np.random.choice(ngrid))

    def sig(self, x):
        return scipy.special.expit(2*self.slope*x)

    def likelihood(self, target, jam=None):
        if jam is None:
            jx = self.jx  # numpy array, all locations
            jy = self.jy  # numpy array, all locations
        else:
            jx = jam[0]   # single float, one location
            jy = jam[1]   # single float, one location
        return self.sig(np.sqrt((jx - self.comm[0])**2 + (jy - self.comm[1])**2) -
                        np.sqrt((target[0] - self.comm[0])**2 + (target[1] - self.comm[1])**2))

    def contact(self, target):
        jam = self.jammer
        p = self.likelihood(target, jam)
        return np.random.choice([True, False],p=(p, 1.-p)) 

    def likelihood_obs(self, target, obs):
            self.p_success = self.likelihood(target)              # Prob at all locations
            self.p_obs = self.p_success if obs else 1-self.p_success
            return self.p_obs

    def run(self, steps=3):
        for _ in range(steps):
            self.asset_contacted = self.contact(self.asset)       # True/False at veridical jammer location
            self.hq_contacted = self.contact(self.hq)          # True/False at veridical jammer location
            self.p_obs_asset = self.likelihood_obs(self.asset, self.asset_contacted)
            self.p_obs_hq = self.likelihood_obs(self.hq, self.hq_contacted)
            self.logPjammer_unnormalized = np.log(self.p_obs_asset) + np.log(self.p_obs_hq) + self.logPjammer_prior
            self.logPjammer_prior = scipy.special.log_softmax(self.logPjammer_unnormalized)  # Prior updated to Posterior
            self.teleport_comm()

    def render(self):
        pass
