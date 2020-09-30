import numpy as np
import scipy.special
import matplotlib.pylab as plt

class Jams():
    def __init__(self, ngrid=5, ncomms=1, njams=1, slope=1.):
        self.step = 0
        self.smallest = 1e-323
        self.ngrid = ngrid
        self.ncomms = ncomms
        self.njams = njams
        self.slope = slope
        self.hq = (0,0)
        self.asset = (ngrid-1,ngrid-1)
        self.comm = None
        self.jammer = None
        self.teleport_comm()
        self.teleport_jammer()
        self.jx = np.arange(ngrid).reshape(1,ngrid)*np.ones((ngrid,1))
        self.jy = np.arange(ngrid).reshape(ngrid,1)*np.ones((1,ngrid))
        # All distributions are represented as logs for stability
        self.logPjammer_prior = np.ones((ngrid, ngrid))*(-2.)*np.log(ngrid) # logProb(jammer@loc); init to uniform

    def teleport_comm(self):
        self.comm = []
        for _ in range(self.ncomms):
            self.comm.append(self.teleport(self.ngrid))

    def teleport_jammer(self):
        self.jammer = []
        for _ in range(self.njams):
            self.jammer.append(self.teleport(self.ngrid))

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
        kc = 0  # need to make this a loop
        return self.sig(np.sqrt((jx - self.comm[kc][0])**2 + (jy - self.comm[kc][1])**2) -
                        np.sqrt((target[0] - self.comm[kc][0])**2 + (target[1] - self.comm[kc][1])**2))

    def contact(self, target):
        kj = 0  # need to make this a loop
        jam = self.jammer[kj]
        p = self.likelihood(target, jam)
        return np.random.choice([True, False],p=(p, 1.-p)) 

    def likelihood_obs(self, target, obs):
            self.p_success = self.likelihood(target)              # Prob at all locations
            self.p_obs = self.p_success if obs else 1-self.p_success
            return self.p_obs

    def run(self, steps=1):
        for _ in range(steps):
            # need to loop these over comms
            self.asset_contacted = self.contact(self.asset)       # True/False at veridical jammer location
            self.hq_contacted = self.contact(self.hq)          # True/False at veridical jammer location
            # also need to contact comms <--> comms
            self.p_obs_asset = self.likelihood_obs(self.asset, self.asset_contacted)
            self.p_obs_hq = self.likelihood_obs(self.hq, self.hq_contacted)
            self.logPjammer_unnormalized = np.log(self.p_obs_asset + self.smallest) + np.log(self.p_obs_hq + self.smallest) + self.logPjammer_prior
            self.logPjammer_prior = scipy.special.log_softmax(self.logPjammer_unnormalized)  # Prior updated to Posterior
            self.teleport_comm()
            self.step += 1

    def render(self):
        l = plt.imshow(self.logPjammer_prior, cmap='hot', interpolation='nearest')
        plt.text(self.asset[0], self.asset[1],"Asset")
        plt.text(self.hq[0], self.hq[1],"Headquarters")
        for kc in range(self.ncomms):
            plt.text(self.comm[kc][0], self.comm[kc][1],"Comm")
        for kj in range(self.njams):
            plt.text(self.jammer[kj][0], self.jammer[kj][1],"Jammer")
        plt.title("Steps = " + str(self.step))
        return(l)
