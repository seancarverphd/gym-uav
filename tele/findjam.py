import torch
import numpy as np
import scipy.special
import matplotlib.pylab as plt

class Jams():
    def __init__(self, ngrid=5, ncomms=1, njams=1, slope=10., seed=None):
        self.seed = seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed+1)
        self.step = 0  # initialize counter for number of steps
        self.ngrid = ngrid  # grid points on map in 1D
        self.ncomms = ncomms
        self.njams = njams
        self.slope = slope
        self.hq = (0,0)
        self.asset = (ngrid-1,ngrid-1)
        self.comm = None
        self.jammers = None
        self.teleport_comm()
        self.teleport_jammers()
        self.Jx, self.Jy = self.makeJxy()
        self.Jx1 = torch.tensor([self.jammers[kj][0] for kj in range(self.njams)], dtype=float)  # single float, one location
        self.Jy1 = torch.tensor([self.jammers[kj][1] for kj in range(self.njams)], dtype=float)  # single float, one location
        self.ddiff = torch.tensor([self.njams] + [self.ngrid, self.ngrid]*self.njams, dtype=float)
        self.ddiff1 = torch.zeros(self.njams)
        # All distributions are represented as logs for stability
        self.logPjammers_prior = torch.ones([self.ngrid]*2*self.njams)*(-2*self.njams)*np.log(self.ngrid) # logProb(jammers@loc); init to uniform
        self.priorshape = self.logPjammers_prior.shape

    def teleport_comm(self):
        self.comm = []
        for _ in range(self.ncomms):
            self.comm.append(self.teleport(self.ngrid))

    def teleport_jammers(self):
        self.jammers = []
        for _ in range(self.njams):
            self.jammers.append(self.teleport(self.ngrid))

    def teleport(self, ngrid):
        return (np.random.choice(ngrid), np.random.choice(ngrid))

    def itertuple(self, k):
        if k == -1:
            dims = 2*self.njams - 2
        else:
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
        Jrx = torch.tensor(np.array(Jx), dtype=float)
        Jry = torch.tensor(np.array(Jy), dtype=float)
        return Jrx, Jry

    def logsig(self, x):
        return torch.log(scipy.special.expit(2*self.slope*x/self.ngrid))
        # return torch.log(scipy.special.expit(2*self.slope*x))

    def distdiff(self, target, jx, jy, kc=0):
        arg1 = (jx - self.comm[kc][0])**2 + (jy - self.comm[kc][1])**2 
        arg2 = torch.tensor((target[0] - self.comm[kc][0])**2 + (target[1] - self.comm[kc][1])**2, dtype=float)
        return torch.sqrt(arg1) - torch.sqrt(arg2)

    def loglikelihood(self, target, kc=0):
        self.ddiff = self.distdiff(target, self.Jx, self.Jy, kc)
        return self.logsig(self.ddiff).sum(axis=0)  # axis 0 is jammer num, add logs because independent

    def loglikelihood_obs(self, target, obs):
        self.log_p_success = self.loglikelihood(target)
        self.log_p_obs = self.log_p_success if obs else torch.log(1 - torch.exp(self.log_p_success))
        return self.log_p_obs

    def loglikelihood_scalar(self, target, kc=0):
        self.ddiff1 = self.distdiff(target, self.Jx1, self.Jy1, kc)
        return self.logsig(self.ddiff1).sum(axis=0)  # axis 0 is jammer num, add logs because independent

    def try_to_contact(self, target):
        p = torch.exp(self.loglikelihood_scalar(target))
        return np.random.choice([True, False],p=(p, 1.-p)) 

    def normalize(self):
        flat = self.logPjammers_prior = torch.nn.functional.log_softmax(self.logPjammers_prior.flatten(), dim=0)  # New Prior equals normalized Posterior
        return flat.reshape(self.priorshape)

    def run(self, steps=1):
        for _ in range(steps):
            # need to loop these over comms, also need to contact comms <--> comms
            self.asset_contacted = self.try_to_contact(self.asset)       # Returns True/False using veridical jammer location(s)
            self.hq_contacted = self.try_to_contact(self.hq)             # Returns True/False using veridical jammer location(s)
            # The rest of the calculations should NOT use the unknown jammer location(s): instead ND-Array of all jammer locations
            self.logPjammers_prior += self.loglikelihood_obs(self.asset, self.asset_contacted)  # Returns ND-array over locations, asset
            self.logPjammers_prior += self.loglikelihood_obs(self.hq, self.hq_contacted)  # decomposes into sum by independence assumption
            self.logPjammers_prior = self.normalize()
            self.teleport_comm()
            self.step += 1

    def marginal(self, P):
        for i, I in enumerate(self.itertuple(-1)):
            pass
        Ps = torch.zeros((i+1, self.ngrid, self.ngrid))
        for j, I in enumerate(self.itertuple(-1)):
            Ps[j] = P[I[0]]
        return torch.logsumexp(Ps, dim=0).T

    def render(self):
        l = plt.imshow(self.marginal(self.logPjammers_prior), cmap='hot', interpolation='nearest')
        plt.text(self.asset[0], self.asset[1], "Asset")
        plt.text(self.hq[0], self.hq[1], "Headquarters")
        # plt.text(self.comm[0][0], self.comm[0][1], "Comm")
        for kc in range(self.ncomms):
            plt.text(self.comm[kc][0], self.comm[kc][1], "Comm")
        for kj in range(self.njams):
            plt.text(self.jammers[kj][0], self.jammers[kj][1],"Jammer")
        plt.title("Steps = " + str(self.step))
        return(l)

