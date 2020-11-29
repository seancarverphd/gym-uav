import copy
import itertools
import matplotlib.pylab as plt
import numpy as np
import scipy.special
import torch


class JamData():
    def __init__(self):
        self.alldata = False
        self.step = None
        self.friendly_pre = None
        self.friendly = None
        self.comms = None
        self.jammers_pre = None
        self.jammers = None
        self.torchstate = None
        self.numpystate = None
        self.adjacency = None
        self.logPjammers_prior = None
        self.logPjammers_predict = None
        self.update = None
        self.logPjammers_unnormalized = None
        self.logPjammers_posterior = None


class Jams():
    def __init__(self, ngrid=5, ncomms=1, nassets=1, njams=1, slope=10., nsteps=1, move=True, misspecified=False, delta=None, seed=None, push=True):
        self.ngrid = ngrid  # grid points on map in 1D
        self.ncomms = ncomms
        self.nassets = nassets
        self.njams = njams
        self.slope = slope
        self.nsteps = nsteps
        self.move = move
        self.assume_move = move if not misspecified else not move
        self.delta = delta
        self.seed = seed
        self.push = push
        self.current = JamData()
        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed+1)
        self.hq = self.headquarters()
        self.nfriendly = len(self.hq) + self.ncomms + self.nassets  # 1 for headquarters
        self.current.adjacency = torch.zeros((self.nfriendly, self.nfriendly), dtype=bool)
        self.assets0 = ((self.ngrid-1.1,self.ngrid-1.1),)
        self.assign_assets(self.assets0)
        self.friendly_initialize()
        self.jammer_initialize()
        self.stack = []
        self.currect_on_stack = False
        if self.delta == 'known':
            self.delta = tuple(int(round(k)) for k in self.tuple_of_all_jammers())
        if self.delta is None:
            self.current.logPjammers_unnormalized = torch.ones([self.ngrid]*2*self.njams)*(-2*self.njams)*np.log(self.ngrid)  # logProb(jammers@loc); init to discrete-uniform on grid
        else:
            assert len(self.delta) == 2*self.njams
            self.current.logPjammers_unnormalized = torch.ones([self.ngrid]*2*self.njams)*(-np.inf)
            self.current.logPjammers_unnormalized[self.delta] = 0

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
        self.current.comms = self.teleport_comms()
        return self.friendly_flatten(self.hq, self.current.comms, self.assets)
        # self.current.friendly = 

    
    def friendly_initialize(self):
        '''
        friendly_initialize: Initialize the friendly's
        '''
        self.comms_set = {(i + 1) for i in range(self.ncomms)}
        self.current.friendly = self.friendly_move()


    def teleport_jammers(self):
        '''
        teleport_jammers: select random locations on grid for jammers
                          done once on class initialization
        '''
        self.current.jammers = []
        for _ in range(self.njams):
            self.current.jammers.append(self.teleport_offgrid())


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
        self.ddiff = torch.tensor([self.njams] + [self.ngrid, self.ngrid]*self.njams, dtype=float)
        self.ddiff1 = torch.zeros(self.njams)
        self.ambient_noise_power = 0
        self.makeMj()
        self.makeMf1()
        # All distributions are represented as logs for stability
        self.priorshape = self.current.logPjammers_unnormalized.shape


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
        Jx1 = torch.tensor([self.current.jammers[kj][0] for kj in range(self.njams)], dtype=float)  # each component single float, veridical x-location 
        Jy1 = torch.tensor([self.current.jammers[kj][1] for kj in range(self.njams)], dtype=float)  # each component single float, veridical y-location
        return Jx1, Jy1


    # def adjacent_grid_coord(self, old_coord):
    #     new_coord = old_coord + np.random.choice([-1., 0., 1.])
    #     if new_coord < 0.:
    #         new_coord += 1.
    #     elif new_coord > self.ngrid-1:
    #         new_coord -= 1.
    #     return new_coord


    def tuple_of_all_jammers(self):
        '''
        tuple_of_all_jammers(): takes veridical jammer locations and produces a tuple of all jammers suitable for an idx for list_of_neighbors
                                and other functions
        '''
        idx = []  # idx is an index when indicates position on grid---but doesn't have to, and usually doesn't
        for kj in range(self.njams):
            idx.extend(list(self.current.jammers[kj]))
        return tuple(idx)


    def list_of_tuples_for_each_jammer(self, idx):
        '''
        list_of_tuples_for_each_jammer: given an index into joint space, break tuple into list of tuples one for each jammer (x, y)
                                        inverts: list_of_tuples_for_each_jammer(tuple_of_all_jammers) = self.current.jammers
        '''
        listofjammers = []
        for kj in range(self.njams):
            listofjammers.append((idx[2*kj], idx[2*kj+1]))
        return listofjammers


    def jammers_move(self):
        '''
        jammers_move: returns new position of jammers, randomly selected from neighbors with equal weight
        '''
        # for kj in range(self.njams):
        #     newx = self.adjacent_grid_coord(self.jammers[kj][0])
        #     newy = self.adjacent_grid_coord(self.jammers[kj][1])
        #     self.jammers[kj] = (newx, newy)
        if not self.move:
            return self.current.jammers
        old = self.tuple_of_all_jammers()
        neighbors = self.list_of_neighbors(old)
        new = np.random.choice(len(neighbors))
        return self.list_of_tuples_for_each_jammer(neighbors[new])


    def list_of_neighbors(self, idx):
        '''
        list_of_neighbors: given an index idx (real valued) into joint space, returns list of neighbors in the space; 
                           involves edge effexts, neighbors are one or none different from idx in each dimension
        '''
        assert idx[0] >= 0
        assert idx[0] <= self.ngrid - 1
        if idx[0] < 1:
            lowest = (idx[0] % 1)
            list1 = [lowest, lowest+1]
        elif idx[0] == self.ngrid - 1:
            highest = self.ngrid - 1
            list1 = [highest-1, highest]
        elif idx[0] > self.ngrid - 2:  # > 8.0 when ngrid==10
            highest = (idx[0] % 1) + self.ngrid - 2
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


    # def number_of_boundaries(self, idx):
    #     nbounds = 0
    #     for i in idx:
    #         if i < 0:
    #             nbounds = -np.inf
    #             print("Warning: number_of_boundaries outside of grid!")
    #         elif i == 0:
    #             nbounds += 1
    #         elif i == self.ngrid-1:
    #             nbounds += 1
    #         elif i > self.ngrid-1:
    #             nbounds = -np.inf
    #             print("Warning: number_of_boundaries outside of grid!")
    #     return nbounds


    # def weight_of_index(self, neighbor, idx):
    #     assert len(idx) == 2*self.njams
    #     centerweight = torch.tensor((2**self.number_of_boundaries(idx))/(3**(2*self.njams)))
    #     neighborweight = torch.tensor((2**self.number_of_boundaries(neighbor))/(3**(2*self.njams)))
    #     return torch.tensor([centerweight, neighborweight]).min()


    def jam_convolve(self, idx1, logP):  # sum over idx0, value at idx1
        neighbors = self.list_of_neighbors(idx1)
        terms = torch.tensor([logP[idx0] - torch.log(torch.tensor(len(self.list_of_neighbors(idx0)), dtype=float)) for idx0 in neighbors])
        return torch.logsumexp(terms, dim=0)


    def jammers_predict_args(self, logP):
        '''
        jammers_predict_args: version of function with calling and returning arguments
        '''
        newP = copy.deepcopy(logP)
        if not self.assume_move:
            return newP
        for idx1 in self.itertuple(2*self.njams):
            newP[idx1] = self.jam_convolve(idx1, logP)
        return newP


    def jammers_predict(self):
        '''
        jammers_predict: wrapper for jammers_predict_args that doesn't use arguments, takes them from self
        '''
        self.current.logPjammers_unnormalized = self.jammers_predict_args(self.current.logPjammers_unnormalized)


    def dist_jxy_to_friendly(self, jx, jy, kf=0):
        '''
        dist_jxy_to_friendly computes the Euclidean distance from jammer(s) (jx, jy) to a specified friendly (fx, fy) in the Cartesian plane
                     jx, jy could be grid tensors (shape: [njams] + [GRIDSHAPE]) for grid of jammers or 
                                                  (shape: [njams]) for 1D veridical jammer locations in x and y
                     kf specifies kf^th comm
        Might generalize Euclidean distance to distance on globe, but that probably isn't necessary
        '''
        #TODO see if you can let friendlies be a tensor to do all friendlies at once, faster
        fx = torch.tensor(self.current.friendly[kf][0], dtype=float)
        fy = torch.tensor(self.current.friendly[kf][1], dtype=float)
        return torch.sqrt((jx - fx)**2 + (jy - fy)**2)


    def dist_jxy_to_point(self, jx, jy, xy):
        '''
        dist_jxy_to_point computes the Euclidean distance from jammer(s) (jx, jy) to a specified point (px, py) in the Cartesian plane
        Might generalize Euclidean distance to distance on globe, but that probably isn't necessary
        '''
        px = torch.tensor(xy[0], dtype=float)
        py = torch.tensor(xy[1], dtype=float)
        return torch.sqrt((jx - px)**2 + (jy - py)**2)


    def power_friendly_at_friendly(self):
        '''
        power_friendly_at_friendly(): returns a tensor of shape [nfriendly, nfreindly] showing at [f1,f2] power at friendly f2 of friendly f1
        '''
        return torch.tensor([[# 0 if f1 == f2 else 
                (#self.Mf1[f1]
                    1/self.dist_jxy_to_friendly(self.current.friendly[f1][0], 
                        self.current.friendly[f1][1], f2)**2) for f1 in range(self.nfriendly)] for f2 in range(self.nfriendly)])
    

    def power_jammer_at_friendly_grid(self):
        '''
        power_jammer_at_friendly_grid(): returns an ND-tensor of shape [self.nfriendly, self.njams]+GRID_SHAPE
                                         component (kf,kj,grid1...grid2j) is power at friendly kf of jammer kj when the jammers are located at grid1...grid2j
        '''
        #TODO There seems to be some redundant calulations here.  Need just a two-dimensional grid not 2J-dims.
        return torch.stack([(1./(self.dist_jxy_to_friendly(self.Jx, self.Jy, kf)**2)) for kf in range(self.nfriendly)], dim=0) # Mj  # friendly at 0th position


    def power_jammer_at_friendly_veridical(self):
        '''
        power_jammer_at_friendly_veridical(): returns a 2D-tensor of shape [self.nfriendly, self.njams]
                                               component (kf,kj) is power at friendly kf of jammer kj
        '''
        Jx1, Jy1 = self.makeJxy1()
        #TODO May be running above function an unnecesary number of times
        return torch.stack([(1./(self.dist_jxy_to_friendly(Jx1, Jy1, kf)**2)) for kf in range(self.nfriendly)], dim=0)  # Mj


    def power_jammer_at_point_veridical(self, xy):
        '''
        power_jammer_at_point_veridical(xy): returns a 1D-tensor of shape [self.njams]
                                             component kj is power of jammer kj at point with coordinates xy (real-valued 2-tuple)
        '''
        Jx1, Jy1 = self.makeJxy1()
        #TODO May be running above function an unnecesary number of times
        return (1./(self.dist_jxy_to_point(Jx1, Jy1, xy)**2))


    def power_ambient(self):
        '''
        power_ambient(): Returns 0 for now
        '''
        return self.ambient_noise_power


    def power_background_at_friendly_veridical(self):
        return self.power_jammer_at_friendly_veridical().sum(dim=1) + self.power_ambient()


    def power_background_at_friendly_grid(self):
        return self.power_jammer_at_friendly_grid().sum(dim=1) + self.power_ambient()


    def power_background_at_point_veridical(self, xy):
        return self.power_jammer_at_point_veridical(xy).sum(dim=0) + self.power_ambient()


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


    def normalize_unnormalized(self):
        '''
        normalize_unnormalized: wrapper for normalize that normalizes posterior (not used in code but might be on command line)
        '''
        return self.normalize(self.current.logPjammers_unnormalized)


    def run(self, steps=1, record=False):
        self.alldata = False
        self.current_on_stack = False
        for s in range(steps):
            if record and s==steps-1:
                self.current.friendly_pre = copy.deepcopy(self.current.friendly)
                self.current.jammers_pre = copy.deepcopy(self.current.jammers)
            self.step += 1
            self.current.step = copy.deepcopy(self.step)
            self.current.friendly = self.friendly_move()  # teleports comms to new locations stored in self.friendly
            self.current.jammers = self.jammers_move()
            self.current.adjacency = self.all_try()  # Next line uses random number generatation and depends on random state
            if record and s==steps-1:
                self.current.logPjammers_prior = self.current.logPjammers_unnormalized
                self.current.logPjammers_predict = self.jammers_predict_args(self.current.logPjammers_prior)
                self.current.update = self.update_jammers(self.current.adjacency)
                self.current.logPjammers_unnormalized = self.current.logPjammers_predict + self.current.update
                self.current.logPjammers_posterior = self.normalize(self.current.logPjammers_unnormalized)
                self.current.torchstate = torch.get_rng_state()
                self.current.numpystate = np.random.get_state()
                self.current.alldata = True
            else:
                self.current.logPjammers_unnormalized = self.jammers_predict_args(self.current.logPjammers_unnormalized) + self.update_jammers(self.current.adjacency)


    def pushstack(self):
        if self.current.alldata is True:
            self.stack.append(copy.deepcopy(self.current))
            self.current_on_stack = True


    def popstack(self):
        if self.current_on_stack:
            self.stack.pop()
        if len(self.stack) == 0:
            print("Bottom of stack reached!")
            return
        self.current = copy.deepcopy(self.stack[-1])
        self.step = copy.deepcopy(self.current.step)
        self.alldata = True
        self.current_on_stack = True
        torch.set_rng_state(self.current.torchstate)
        np.random.set_state(self.current.numpystate)


    def advance(self, steps=1):
        self.run(steps, record=True)
        if self.push:
            self.pushstack()
        self.render()


    def retreat(self, stacksteps=1):
        for _ in range(stacksteps):
            self.popstack()
        self.render()


    def logcumsumexp(self, x, dim):
        # slow implementation (taken from web), but ok for now
        if (dim != -1) or (dim != x.ndimension() - 1):
            x = x.transpose(dim, -1)
        out = []
        for i in range(1, x.size(-1) + 1):
            out.append(torch.logsumexp(x[..., :i], dim=-1, keepdim=True))
        out = torch.cat(out, dim=-1)
        if (dim != -1) or (dim != x.ndimension() - 1):
            out = out.transpose(-1, dim)
        return out


    def credible(self, logP, C=0.95):
        the_sort = self.normalize(logP).flatten().sort(descending=True)
        included = self.logcumsumexp(the_sort.values, dim=0) < np.log(C)
        idx = np.where(np.diff(included))[0][0]
        inside = the_sort.indices[:idx]
        #boundary = the_sort.indices[idx+1]
        credible_set = torch.zeros(included.shape)
        credible_set[inside] = True
        credible_set = credible_set.reshape(logP.shape)
        return credible_set


    def video(self, nframes):
        for f in range(nframes):
            self.advance()
            plt.savefig('video/frame'+str(f)+'.png')
            plt.clf()


    def normalizer(self, logP):
        return torch.logsumexp(logP.flatten(), dim=0)


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


    def annotations(self, titleprefix=''):
        '''
        annotations: add annotations to plot
        '''
        col = 'lightblue' if self.current.step == 0 else 'black'
        for kf in range(self.nfriendly):
            if kf==0:  # headquarters
                plt.text(self.current.friendly[kf][0], self.current.friendly[kf][1], "Headquarters", color=col)
            elif kf in self.comms_set:  # comms
                plt.text(self.current.friendly[kf][0], self.current.friendly[kf][1], "Comm", color=col)
            else:  # assets
                plt.text(self.current.friendly[kf][0], self.current.friendly[kf][1], "Asset", color=col)
        for kj in range(self.njams):
            plt.text(self.current.jammers[kj][0], self.current.jammers[kj][1],"Jammer", color=col)
        estimates = [e.item() for e in self.estimates()]
        for ej in self.list_of_tuples_for_each_jammer(estimates):
            plt.text(ej[0], ej[1],"Estimate", color=col)
        plt.title(titleprefix + "Steps = " + str(self.current.step))
        plt.show()


    def connections(self):
        # return  # Not tested and does not seem to work completely and properly
        ax = plt.gca()
        for f2 in range(self.nfriendly):
            for f1 in range(f2):
                x = self.current.friendly[f1][0]
                y = self.current.friendly[f1][1]
                dx = self.current.friendly[f2][0] - x
                dy = self.current.friendly[f2][1] - y
                assert f1 < f2
                if self.current.adjacency[f1,f2] and not self.current.adjacency[f2, f1]:
                    style = '->'
                elif self.current.adjacency[f2,f1] and not self.current.adjacency[f1, f2]:
                    style = '<-'
                elif self.current.adjacency[f1,f2] and self.current.adjacency[f2, f1]:
                    style = '<->'
                else:
                    continue
                ax.annotate("", xy=(x+dx,y+dy), xytext=(x, y), arrowprops=dict(arrowstyle=style, color='lightgreen'))
                    # plt.arrow(x,y,dx,dy, width=0.002, head_width=0.006, head_length=0.003)
                # else:
                    # pass
                    #plt.arrow(x,y,dx,dy, width=0.002, head_width=0.006, head_length=0.003, linestyle=':', color='red')


    def render(self):
        '''
        render: plots the marginal of the unnormalized posterior
        '''
        plt.clf()
        plt.imshow(self.marginal(self.current.logPjammers_unnormalized).T, cmap='hot', interpolation='nearest')  # transpose to get plot right
        self.annotations()
        self.connections()


    def render_background(self):
        '''
        render_background: plots the power at background
        '''
        BG = torch.zeros((self.ngrid, self.ngrid), dtype=float)
        for g1 in range(self.ngrid):
            for g2 in range(self.ngrid):
                BG[g1, g2] = self.power_background_at_point_veridical((g1, g2))
        plt.clf()
        plt.imshow(BG.T, cmap='hot', interpolation='nearest')
        self.annotations()
        self.connections()


    def render_posterior(self):
        '''
        render: plots the marginal of the posterior
        '''
        assert self.current.alldata is True
        plt.clf()
        plt.imshow(self.marginal(self.current.logPjammers_posterior).T, cmap='hot', interpolation='nearest')  # transpose to get plot right
        self.annotations()
        self.connections()


    def render_update(self):
        '''
        render_update: draws I don't know what; update is not a distribution so marginal might not mean anything for njams>1
        '''
        assert self.current.alldata is True
        plt.clf()
        plt.imshow(self.marginal(self.current.update).T, cmap='hot', interpolation='nearest')  # transpose to get plot right
        self.annotations("Update Before: ")


    def render_prediction(self):
        '''
        render: plots the marginal
        '''
        assert self.current.alldata is True
        plt.clf()
        plt.imshow(self.marginal(self.current.logPjammers_predict).T, cmap='hot', interpolation='nearest')  # transpose to get plot right
        self.annotations("Prediction Before: ")


    def render_prior(self):
        '''
        render: plots the marginal
        '''
        assert self.current.alldata is True
        plt.clf()
        plt.imshow(self.marginal(self.current.logPjammers_prior).T, cmap='hot', interpolation='nearest')  # transpose to get plot right
        self.annotations("Prior Before: ")


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
        imax = self.current.logPjammers_unnormalized.argmax()
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
        assert len(freeze) + 2 == len(self.current.logPjammers_unnormalized.shape)
        plt.imshow(self.conditional(self.current.logPjammers_unnormalized, freeze).T, cmap='hot', interpolation='nearest')  # transpose to get plot right
        self.annotations()


    def test_independence(self):
        logjoint = self.current.logPjammers_unnormalized
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

