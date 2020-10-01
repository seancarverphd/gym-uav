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

    def logsig(self, x):
        return np.log(scipy.special.expit(2*self.slope*x))

    def logsumexp_listofarrays(self, loa):
        # logsumexp across the array-elements of a list of arrays and return a new array of same shape
        s = loa[0].shape
        flats = [loa[k].flatten() for k in range(len(loa))]
        logsumexpflats = np.array([scipy.special.logsumexp([flats[k][i] for k in range(len(loa))]) for i in range(np.prod(s))])
        return logsumexpflats.reshape(s)

    def loglikelihood(self, target, jam=None, kc=0):
        if jam is None:   # need njams jx's and jy's then AIC with njams unknown
            jx = [self.jx]*self.njams  # numpy array, all locations
            jy = [self.jy]*self.njams  # numpy array, all locations
        else:
            jx = [jam[kj][0] for kj in range(self.njams)]  # single float, one location
            jy = [jam[kj][1] for kj in range(self.njams)]  # single float, one location
        self.logsigs = [self.logsig(np.sqrt((jx[kj] - self.comm[kc][0])**2 + (jy[kj] - self.comm[kc][1])**2) -
                        np.sqrt((target[0] - self.comm[kc][0])**2 + (target[1] - self.comm[kc][1])**2)) for kj in range(self.njams)]
        return self.logsumexp_listofarrays(self.logsigs)

    def contact(self, target):
        p = np.exp(self.loglikelihood(target, self.jammer))
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
