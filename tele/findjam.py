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
        l = np.log(scipy.special.expit(2*self.slope*x))
        assert (l<=0).all()
        return l

    def logsumexp_listofarrays(self, loa):
        # logsumexp across the array-elements of a list of arrays and return a new array of same shape
        s = loa[0].shape
        flats = [loa[k].flatten() for k in range(len(loa))]
        logsumexpflats = np.array([scipy.special.logsumexp([flats[k][i] for k in range(len(loa))]) for i in range(int(np.prod(s)))])
        return logsumexpflats.reshape(s)

    def sum_listofarrays(self, loa):
        # sum across the array-elements of a list of arrays and return a new array of same shape
        s = loa[0].shape
        flats = [loa[k].flatten() for k in range(len(loa))]
        sumflats = np.array([sum(np.array([flats[k][i] for k in range(len(loa))])) for i in range(int(np.prod(s)))])
        return sumflats.reshape(s)

    def itertuple(self, k):
        dims = 2*self.njams
        top = self.ngrid
        I = [0]*dims
        i = 0
        yield tuple(I), I[k]
        while True:
            I[i] += 1
            if I[i] >= top:
                assert I[i] == top
                I[i] = 0
                i += 1
            else:
                i = 0
                yield tuple(I), I[k]
            if i >= dims:
                break

    def makeJm(self, k):
        Jm = np.zeros([self.ngrid]*(2*self.njams))
        for I, ik in self.itertuple(k):
            Jm[I] = ik
        return Jm

    def makeJxy(self):
        Jx = []
        Jy = []
        for k in range(self.njams):
            Jx.append(self.makeJm(2*k))
            Jy.append(self.makeJm(2*k+1))
        return Jx, Jy

    def loglikelihood(self, target, jam=None, kc=0):
        if jam is None:   # need njams jx's and jy's then AIC with njams unknown
            jx, jy = self.makeJxy()
        else:
            jx = [jam[kj][0] for kj in range(self.njams)]  # single float, one location
            jy = [jam[kj][1] for kj in range(self.njams)]  # single float, one location
        self.logsigs = [self.logsig(np.sqrt((jx[kj] - self.comm[kc][0])**2 + (jy[kj] - self.comm[kc][1])**2) -
                        np.sqrt((target[0] - self.comm[kc][0])**2 + (target[1] - self.comm[kc][1])**2)) for kj in range(self.njams)]
        l = self.sum_listofarrays(self.logsigs)
        assert (l <= 0).all()
        return l

    def contact(self, target):
        p = np.exp(self.loglikelihood(target, self.jammer))
        return np.random.choice([True, False],p=(p, 1.-p)) 

    def loglikelihood_obs(self, target, obs):
        self.log_p_success = self.loglikelihood(target)
        self.log_p_obs = self.log_p_success if obs else np.log(1 - np.exp(self.log_p_success))
        return self.log_p_obs

    def run(self, steps=1):
        for _ in range(steps):
            # need to loop these over comms
            self.asset_contacted = self.contact(self.asset)       # True/False at veridical jammer location
            self.hq_contacted = self.contact(self.hq)             # True/False at veridical jammer location
            # also need to contact comms <--> comms
            self.log_p_obs_asset = self.loglikelihood_obs(self.asset, self.asset_contacted)
            self.log_p_obs_hq = self.loglikelihood_obs(self.hq, self.hq_contacted)
            self.logPjammer_unnormalized = self.log_p_obs_asset + self.log_p_obs_hq + self.logPjammer_prior
            self.logPjammer_prior = scipy.special.log_softmax(self.logPjammer_unnormalized)  # Prior updated to Posterior
            self.teleport_comm()
            self.step += 1

    def marginal(self, P):
        M = np.zeros((self.ngrid, self.ngrid))
        for i in range(self.ngrid):
            for j in range(self.ngrid):
                M += P[(i,j)]
        return M

    def render(self):
        l = plt.imshow(self.marginal(self.logPjammer_prior), cmap='hot', interpolation='nearest')
        plt.text(self.asset[0], self.asset[1], "Asset")
        plt.text(self.hq[0], self.hq[1], "Headquarters")
        # plt.text(self.comm[0][0], self.comm[0][1], "Comm")
        for kc in range(self.ncomms):
            plt.text(self.comm[kc][0], self.comm[kc][1], "Comm")
        for kj in range(self.njams):
            plt.text(self.jammer[kj][0], self.jammer[kj][1],"Jammer")
        plt.title("Steps = " + str(self.step))
        return(l)
