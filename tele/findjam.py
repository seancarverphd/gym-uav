import copy
import itertools
import matplotlib.pylab as plt
import numpy as np
import scipy.special
import torch

class Jams():
    def __init__(self, ngrid=5, ncomms=1, nassets=1, njams=1, slope=10., nsteps=1, seed=None):
        self.ngrid = ngrid  # grid points on map in 1D
        self.ncomms = ncomms
        self.nassets = nassets
        self.njams = njams
        self.slope = slope
        self.nsteps = nsteps
        self.seed = seed
        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed+1)
        self.hq = self.headquarters()
        self.nfriendly = len(self.hq) + self.ncomms + self.nassets  # 1 for headquarters
        self.adjacency = torch.zeros((self.nfriendly, self.nfriendly), dtype=bool)
        self.assets0 = ((self.ngrid-1.1,self.ngrid-1.1),)
        self.assign_assets(self.assets0)
        self.friendly_initialize()
        self.jammer_initialize()


    def headquarters(self):
        '''
        headquarters: return coordinates of headquarters on grid
        '''
        return [(0.1, 0.1)]

    
    def assign_assets(self, a0):
        '''
        assign_assets: returns a list of asset locations, copying assets0, then using randomly assigned assets up to self.nassets
        '''
        assets = []
        for a in range(self.nassets):
            if a < len(a0):
                assets.append(a0[a])
            else:
                assets.append(self.teleport_ongrid())
        self.assets0 = a0  # update assets0 with what is passed in
        self.assets = assets


    def teleport_ongrid(self):
        '''
        teleport_ongrid: select a random location on grid
        '''
        return (np.random.choice(self.ngrid), np.random.choice(self.ngrid))


    def teleport_offgrid(self):
        '''
        teleport_offgrid: select a random location within the bounds of the grid, but with probability 1, not on a gridpoint
        '''
        return tuple(np.random.uniform(low=0.0, high=self.ngrid-1, size=2))

    
    def teleport_comms(self):
        '''
        teleport_comms: select random locations on grid for comm(s)
                           done once during every step of the loop
        '''
        comms = []
        for comm in self.comms_set: 
            comms.append(self.teleport_offgrid())
        return comms


    def friendly_flatten(self, hq, comms, assets):
        '''
        friendly_flatten: combine all friendly units into one tuple, in specified order
        '''
        ff = []
        ff.extend(hq)  # just one headquarters
        ff.extend(comms)  # ncomm comm units
        ff.extend(assets) # nassets assets including len(assets0) predefined, others, if any, random
        return tuple(ff)


    def friendly_move(self):
        '''
        friendly_move: right now a wrapper for teleport_comm but will be generalized
        '''
        self.comms = self.teleport_comms()
        self.friendly = self.friendly_flatten(self.hq, self.comms, self.assets)

    
    def friendly_initialize(self):
        '''
        friendly_initialize: Initialize the friendly's
        '''
        self.comms_set = {(i + 1) for i in range(self.ncomms)}
        self.friendly_move()


    def teleport_jammers(self):
        '''
        teleport_jammers: select random locations on grid for jammers
                          done once on class initialization
        '''
        self.jammers = []
        for _ in range(self.njams):
            self.jammers.append(self.teleport_offgrid())


    def jammer_initialize(self):
        '''
        jammers_initialize: initialize jammers
        '''
        self.teleport_jammers()


class JamsPoint(Jams):
    def __init__(self):
        super().__init__(**kwargs)


class JamsGrid(Jams):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 0  # initialize counter for number of steps
        self.Jx, self.Jy = self.makeJxy()
        self.Jx1, self.Jy1 = self.makeJxy1()
        self.ddiff = torch.tensor([self.njams] + [self.ngrid, self.ngrid]*self.njams, dtype=float)
        self.ddiff1 = torch.zeros(self.njams)
        self.ambient_noise_power = 0
        self.makeMj()
        self.makeMf1()
        # All distributions are represented as logs for stability
        self.logPjammers_prior = torch.ones([self.ngrid]*2*self.njams)*(-2*self.njams)*np.log(self.ngrid)  # logProb(jammers@loc); init to discrete-uniform on grid
        self.priorshape = self.logPjammers_prior.shape


    def itertuple(self, dims):
        '''
        itertuple is a generator producing the multiindexes of all elements of a "square" tensor
                  where the the square tensor has dim dimensions of all length self.ngrid
                  usually dims = 2*self.njams (for calculations) or 2*self.njams-2 (for plotting)
        '''
        if dims == 0:
            yield tuple()
            return
        top = self.ngrid
        I = [0]*dims
        yield tuple(I)
        i = 0
        while True:
            I[i] += 1
            if I[i] >= top:
                assert I[i] == top
                I[i] = 0
                i += 1
            else:
                i = 0
                yield tuple(I)
            if i >= dims:
                break


    def makeJm(self, k):
        '''
        makeJm creates an np.array of size [self.ngrid]*(2*self.njams)
               the array returned projects the index onto the kth coordinate
               Jm[i1,i2,...,ik,...] = ik
               This is usefull with doing vectorized tensor arithmetic on ik
               Question: could I do the same thing with broadcasting?
        '''
        Jm = np.zeros([self.ngrid]*(2*self.njams))
        for I in self.itertuple(2*self.njams):
            Jm[I] = I[k] 
        return Jm


    def makeJxy(self):
        '''
        makeJxy creates and returns 2 tensors Jrx, Jry.  Each of these tensors has shape [self.njams] + [self.ngrid]*(2*self.njams)
                the zeroth dimension is j indexing jammers the rest of the dimensions are the size of the joint grid. 
                The tensor components are Jrx[x1,y1,x2,y2,...,xj,...] = xj; for Jry[...] = yj; the x's and y's alternate
                Note xj and yj are integer indicies as well as xy-coordinates representing longitude and lattitude.
                This works because we put our xy-grid on integer range(self.ngrid) values of x and y.
                To generalize, transform Jx --> longitude(Jx) and Jy --> latitude(Jy) or whatever xy values you are using
                The Jrx is [x1, x2, x3, ..., xnjams].  Likewise for Jry.
        makeJxy is called once on object initialization and the values returned are stored as self.Jx and self.Jy
        '''
        Jx = []
        Jy = []
        for k in range(self.njams):
            Jx.append(self.makeJm(2*k))
            Jy.append(self.makeJm(2*k+1))
        # Update for non-integer x, y here
        Jrx = torch.tensor(np.array(Jx), dtype=float)
        Jry = torch.tensor(np.array(Jy), dtype=float)
        return Jrx, Jry


    def makeJxy1(self):
        '''
        makeJrxy1 creates and returns 2 tensors Jrx1, and Jry1.  Each of these has shape [self.njams] -- 1D-tensor
           The jth component of Jrx1 is the veridical x-value of the jth jammer
        '''
        Jrx1 = torch.tensor([self.jammers[kj][0] for kj in range(self.njams)], dtype=float)  # each component single float, veridical x-location 
        Jry1 = torch.tensor([self.jammers[kj][1] for kj in range(self.njams)], dtype=float)  # each component single float, veridical y-location
        return Jrx1, Jry1


    def adjacent_grid_coord(self, old_coord):
        new_coord = old_coord + np.random.choice([-1., 0., 1.])
        if new_coord < 0.:
            new_coord += 1.
        elif new_coord > self.ngrid-1:
            new_coord -= 1.
        return new_coord


    def jammers_move(self):
        '''
        jammers_move:
        '''
        for kj in range(self.njams):
            newx = self.adjacent_grid_coord(self.jammers[kj][0])
            newy = self.adjacent_grid_coord(self.jammers[kj][1])
            self.jammers[kj] = (newx, newy)

 
    def list_of_neighbors(self, idx):
        if idx[0] < 1:
            lowest = idx[0] % 1
            list1 = [lowest, lowest+1]
        elif idx[0] > self.ngrid - 2:
            highest = (idx[0] % 1) + self.ngrid - 1
            list1 = [highest-1, highest]
        else:
            list1 = [idx[0]-1, idx[0], idx[0]+1]
        if len(idx) == 1:
            return list1
        elif len(idx) == 2:
            list2 = self.list_of_neighbors(idx[1:])
            return [a for a in itertools.product(list1, list2)]
        else:
            list2 = self.list_of_neighbors(idx[1:])
            return [(a, *b) for (a, (*b,)) in itertools.product(list1, list2)]


    def number_of_boundaries(self, idx):
        nbounds = 0
        for i in idx:
            if i < 0:
                nbounds = -np.inf
                print("Warning: number_of_boundaries outside of grid!")
            elif i == 0:
                nbounds += 1
            elif i == self.ngrid-1:
                nbounds += 1
            elif i > self.ngrid-1:
                nbounds = -np.inf
                print("Warning: number_of_boundaries outside of grid!")
        return nbounds


    def weight_of_index(self, neighbor, idx):
        assert len(idx) == 2*self.njams
        centerweight = torch.tensor((2**self.number_of_boundaries(idx))/(3**(2*self.njams)))
        neighborweight = torch.tensor((2**self.number_of_boundaries(neighbor))/(3**(2*self.njams)))
        return torch.tensor([centerweight, neighborweight]).min()

    def jam_convolve(self, idx, logP):
        terms = torch.tensor([logP[neighbor] + torch.log(self.weight_of_index(neighbor, idx)) for neighbor in self.list_of_neighbors(idx)])
        return torch.logsumexp(terms, dim=0)


    def jammers_predict_args(self, logP):
        '''
        jammers_predict_args: version of function with calling and returning arguments
        '''
        newP = copy.deepcopy(logP)
        for idx in self.itertuple(2*self.njams):
            newP[idx] = self.jam_convolve(idx, logP)
        return newP


    def jammers_predict(self):
        '''
        jamemrs_predict: wrapper for jammers_predict_args that doesn't use arguments, takes them from self
        '''
        self.logPjammers_prior = self.jammers_predict_args(self.logPjammers_prior)


    def dist_jxy_to_friendly(self, jx, jy, kf=0):
        '''
        dist_to_comm computes the Euclidean distance from (cx, cy) (the comm) to (jx, jy) in the Cartesian plane
                     jx, jy could be grid tensors (shape: [njams] + Grid.shape) for jammers or 
                                                  (shape: [njams]) for 1D veridical jammer locations in x and y
                     kc^th comm
        Might generalize Euclidean distance to distance on globe, but that probably isn't necessary
        '''
        fx = torch.tensor(self.friendly[kf][0], dtype=float)
        fy = torch.tensor(self.friendly[kf][1], dtype=float)
        return torch.sqrt((jx - fx)**2 + (jy - fy)**2)


    def power_friendly_at_friendly(self):
        return torch.tensor([[# 0 if f1 == f2 else 
                (#self.Mf1[f1]
                    1/self.dist_jxy_to_friendly(self.friendly[f1][0], self.friendly[f1][1], f2)**2) for f1 in range(self.nfriendly)] for f2 in range(self.nfriendly)])
    

    def power_jammers_at_friendly_grid(self):
        return torch.stack([(1./(self.dist_jxy_to_friendly(self.Jx, self.Jy, kf)**2)) for kf in range(self.nfriendly)], dim=0).sum(dim=1) # Mj  # friendly at 0th position


    def power_jammers_at_friendly_veridical(self):
        return torch.stack([(1./(self.dist_jxy_to_friendly(self.Jx1, self.Jy1, kf)**2)) for kf in range(self.nfriendly)], dim=0).sum(dim=1)  # Mj


    def power_ambient(self):
        return self.ambient_noise_power


    def power_background_at_friendly_veridical(self):
        return self.power_jammers_at_friendly_veridical() + self.power_ambient()


    def power_background_at_friendly_grid(self):
        return self.power_jammers_at_friendly_grid() + self.power_ambient()


    def sjr_db_veridical(self):
        return 10*torch.log10(self.power_friendly_at_friendly()/self.power_background_at_friendly_veridical())


    def prepare_background(self):
        j = self.njams
        perm = [k+1 for k in range(2*j)]
        perm.append(0)
        return self.power_background_at_friendly_grid().permute(perm).unsqueeze(2*j)


    def prepare_db(self, db):
        j = self.njams
        perm = [2*j, 2*j+1]
        perm.extend([k for k in range(2*j)])
        return db.permute(perm)


    def sjr_db_grid(self):
        S = self.power_friendly_at_friendly()
        B = self.prepare_background()
        db = 10*torch.log10(S/B)
        return self.prepare_db(db)


    def makeMj(self):
        self.Mj = torch.ones((self.njams))  # Will make this more general later


    def makeMf1(self):
        self.Mf1 = torch.ones((self.nfriendly))  # Will make this more general later


    # def distdiff(self, target, jx, jy, kc=0):
    #     '''
    #     distdiff computes the difference between the distances: comm <--> jammer minus comm <--> target
    #              if distance comm <--> jammer is larger than distance comm <--> target, then
    #              difference is positive and probability of successful transmission is closer to one
    #     '''
    #     targetx = torch.tensor(target[0], dtype=float)
    #     targety = torch.tensor(target[1], dtype=float)
    #     dist_c2j = self.dist_to_comm(jx, jy, kc)
    #     dist_c2t = self.dist_to_comm(targetx, targety, kc)
    #     return dist_c2j - dist_c2t


    def logsig(self, x):
        '''
        logsig returns the log of the sigmoid function (expit) of its argument x
               scaled so that result is independent of grid size assuming the same self.slope
        before adjusting for grid size: returned "torch.log(scipy.special.expit(2*self.slope*x))" in this case
               self.slope was slope of sigmoid (before log) specifed at x=0 where there is 
               equal distance between comm-jammer and comm-target, where value of sigmoid (before log) is expit=1/2
        '''
        return torch.log(scipy.special.expit(2*self.slope*x/self.ngrid))


    # def loglikelihood_ddiff(self, target, jx, jy, kc=0):
    #    '''
    #    loglikelihood: log-likelihood of successful communication between comm and target with specified jammer location(s)
    #                   Depending on jammer_x and jammer_y will compute for whole grid or just veridical locations (see wrappers below)
    #    '''
    #    ddiff = self.distdiff(target, jx, jy, kc)
    #    return self.logsig(ddiff).sum(axis=0)  # axis=0 is jammer num, add logs because jamming from different jammers independent


    def loglikelihood_grid(self):
        '''
        loglikelihood_grid: wrapper for loglikelihood passing object's whole grid for possible jammer locations
        '''
        return self.logsig(self.sjr_db_grid())


    def loglikelihood_veridical(self):
        '''
        loglikelihood_veridical: wrapper for loglikelihood passing veridical jammer locations instead of whole grid for possible jammer locations
        '''
        return self.logsig(self.sjr_db_veridical())


    def loglikelihood_obs(self, adjacency):
        '''
        loglikelihood_obs: loglikelihood_grid of actual communication between comm and target
                           pass obs=True for likelihood of success communication
                           pass obs=False for likelihood of unsuccessful communication
        '''
        log_p_success = self.loglikelihood_grid()
        log_p_obs = torch.zeros(log_p_success.shape)
        for f1 in range(self.nfriendly):
            for f2 in range(self.nfriendly):
                log_p_obs[f1,f2] = log_p_success[f1,f2] if adjacency[f1,f2] else torch.log(1 - torch.exp(log_p_success[f1,f2]))
        return log_p_obs


    def update_jammers(self, adjacency):
        # Does account for missing information from out of reach friendlys
        return self.loglikelihood_obs(adjacency).sum(dim=0).sum(dim=0)  # First two slots are for to from friendly; add them up as independent


    def try_to_contact(self, sender, receiver):
        '''
        try_to_contact flips a "coin" (usually unfair) to simulate success or failure to communicate based on likelihood derived from veridical jammer locations
        '''
        p = torch.exp(self.loglikelihood_veridical())
        return torch.tensor(np.random.choice([True, False],p=(p[sender, receiver], 1.-p[sender, receiver])))


    def all_try(self):
        adjacency = torch.zeros((self.nfriendly, self.nfriendly), dtype=bool)
        for sender in range(self.nfriendly):
            for receiver in range(self.nfriendly):
                adjacency[sender, receiver] = self.try_to_contact(sender, receiver)
        return adjacency


    def normalize(self, distrib):
        '''
        normalize is required after a Bayesian update to make the distributions sum to 1.
                  However this is not necessary to do within the run() loop or to get maximum aposteriori estimates of Jammer location
                  normalize is called at the end of the loop.
        '''
        priorshape = distrib.shape
        flat = torch.nn.functional.log_softmax(distrib.flatten(), dim=0)  # New Prior equals normalized Posterior
        return flat.reshape(priorshape)


    def normalize_prior(self):
        '''
        normalize_prior: wrapper for normalize that normalizes prior (not used in code but might be on command line)
        '''
        return self.normalize(self.logPjammers_prior)


    def run(self, steps=None):
        '''
        run take steps or self.nsteps of moving comm (teleport for now) and Bayesian update of prior
        '''
        if steps is None:
            steps = self.nsteps
        for _ in range(steps):
            self.friendly_move()  # teleports comms to new locations
            self.jammers_move()
            self.logPjammers_predict = self.jammers_predict_args(self.logPjammers_prior)
            self.adjacency = self.all_try()
            self.logPjammers_prior = self.logPjammers_predict + self.update_jammers(self.adjacency)
            self.step += 1
            # self.hq_contacted = self.try_to_contact(self.hq)             # Returns True/False using veridical jammer location(s)
            # self.logPjammers_prior += self.loglikelihood_obs(self.hq, self.hq_contacted)  # decomposes into sum by independence assumption
            # self.asset_contacted = self.try_to_contact(self.asset)       # Returns True/False using veridical jammer location(s)
            # self.logPjammers_prior += self.loglikelihood_obs(self.asset, self.asset_contacted)  # Returns ND-array over locations, asset
        self.logPjammers_prior = self.normalize(self.logPjammers_prior)


    def marginal(self, joint):
        '''
        marginal: needed for plotting in 2D if joint density has greater than 2 dimensions
                  adds probabilities for each x_njams and y_njams on 2D grid
                  where njams references the marginal of last jammer position x and y (last two coordinates of joint distribution).
                  All jammer positions are equivalent and should give same answer, but this way book keeping simplifies.
        '''
        dims_summing_across = 2*self.njams-2  # two less dimensions than number of dimensions in joint distribution
        nterms = self.ngrid**dims_summing_across  # number of terms in sum for each x and y in the computation of the marginal
        terms_rearranged_for_sum = torch.zeros((nterms, self.ngrid, self.ngrid))  # coordinates are (term, x, y)
        for term_number, term_multiindex in enumerate(self.itertuple(dims_summing_across)):  # enumerate all term_multiindecies in a tensor with two less coordinates
            terms_rearranged_for_sum[term_number] = joint[term_multiindex]  # each side of this assignment is a 2D tensor in x and y
        return torch.logsumexp(terms_rearranged_for_sum, dim=0)  # convert to probabilities, add across term, then retake log


    def annotations(self):
        '''
        annotations: add annotations to plot
        '''
        col = 'lightblue' if self.step == 0 else 'black'
        for kf in range(self.nfriendly):
            if kf==0:  # headquarters
                plt.text(self.friendly[kf][0], self.friendly[kf][1], "Headquarters", color=col)
            elif kf in self.comms_set:  # comms
                plt.text(self.friendly[kf][0], self.friendly[kf][1], "Comm", color=col)
            else:  # assets
                plt.text(self.friendly[kf][0], self.friendly[kf][1], "Asset", color=col)
        for kj in range(self.njams):
            plt.text(self.jammers[kj][0], self.jammers[kj][1],"Jammer", color=col)
        plt.title("Steps = " + str(self.step))
        plt.show()


    def render(self):
        '''
        render: plots the marginal
        '''
        plt.clf()
        plt.imshow(self.marginal(self.logPjammers_prior).T, cmap='hot', interpolation='nearest')  # transpose to get plot right
        self.annotations()


    def render_prediction(self):
        '''
        render: plots the marginal
        '''
        plt.clf()
        plt.imshow(self.marginal(self.logPjammers_predict).T, cmap='hot', interpolation='nearest')  # transpose to get plot right
        self.annotations()


    def unravel_index(self, index, shape):
        '''
        unravel_index: copied from the web, converts an integer index to a multiindex for a tensor of specified shape
                       needed for finding the grid indicies from argmax's output that point to the maximum aposteriori grid location for jammers
        '''
        out = []
        for dim in reversed(shape):
            out.append(index % dim)
            index = index // dim
        return tuple(reversed(out))


    def estimates(self):
        '''
        estimates: produces the maximum aposteriori estimates of the jammer locations based on the information currently in grid
        '''
        imax = self.logPjammers_prior.argmax()
        return self.unravel_index(imax, tuple([self.ngrid]*(2*self.njams)))


    def logjoint_iid_from_logmarginal(self, logmarginal):
        '''
        logjoint_iid_from_logmarginal:  Construct a joint distribution from a marginal under the assumption that coordinates are iid
                                        iid means independent and identically distributed
                                        In this case, it's pairs of coordinates x and y
        '''
        logjoint_iid = torch.zeros([self.ngrid]*(2*self.njams))  # same shape as logjoint
        for index_joint in self.itertuple(2*self.njams):
            for kj in range(self.njams):
                logjoint_iid[index_joint] += logmarginal[index_joint[2*kj], index_joint[2*kj+1]]
        return logjoint_iid


    def conditional(self, joint, freeze):
        '''
        conditional: create the conditional distribution from the joint distribution
                     if the joint distribution is P(x1, y1, x2, y2, x3, y3)
                     then conditional([4,5,3,1]) is the distribution P(x3, y3|x1=4, y1=5, x2=3, y1=1)
        '''
        return self.normalize(joint[freeze])


    def show_conditional(self, freeze):
        assert len(freeze) + 2 == len(self.logPjammers_prior.shape)
        plt.imshow(self.conditional(self.logPjammers_prior, freeze).T, cmap='hot', interpolation='nearest')  # transpose to get plot right
        self.annotations()


    def test_independence(self):
        logjoint = self.logPjammers_prior
        logmargin = self.marginal(logjoint)
        logjoint_iid = self.logjoint_iid_from_logmarginal(logmargin)
        return logjoint_iid.allclose(logjoint)


class test1D():
    '''
    Unfinished classed for doing 1 dimensional grids: Jammers and Comms move in 1D
    '''
    def __init__(self, ngrid=64, slope=1):
        self.ngrid = ngrid
        self.slope = 1


    def run(self):
        self.grid = torch.zeros((self.ngrid, self.ngrid))
        for j in range(self.ngrid):
            for c in range(self.ngrid):
                self.grid[j, c] = torch.abs(torch.tensor(c) - torch.tensor(j)) - torch.abs(torch.tensor(c))


    def render0(self):
        l = plt.imshow(self.grid, cmap='hot', interpolation='nearest')


    def logsig(self, x):
        return torch.log(scipy.special.expit(torch.true_divide(2*self.slope*x, self.ngrid)))


    def render2(self):
        l = plt.imshow(self.logsig(self.grid), cmap='hot', interpolation='nearest')

