import copy
import gym
from gym import spaces, utils
from gym.utils import seeding
import numpy as np
import findjam

DIR_TO_IDX = {
    'stop'  : 0,
    'north' : 1,
    'south' : 2,
    'east'  : 3,
    'west'  : 4,
}

IDX_TO_DIR = dict(zip(DIR_TO_IDX.values(), DIR_TO_IDX.keys()))
 
class UavA1Env(gym.Env):  
    metadata = {'render.modes': ['human']}   
    def __init__(self, layout='large', seed=None):
        super(UavA1Env, self).__init__()
        if layout == 'large':
            pass
            # self.n_units = 4
            # self.n_streets_ns = 16
            # self.n_streets_ew = 16
            # self.start_streets_ns = 4
            # self.assets = [4, 8, 12, 8, 8, 4, 8, 12]
            # self.max_steps = 50
        # elif layout == 'small':
            # self.n_units = 2
            # self.n_streets_ns = 5
            # self.n_streets_ew = 5
            # self.start_streets_ns = 2
            # self.assets = [3, 1, 3, 3]
            # self.max_steps = 25
        else:
            assert False  # No valid layout specified!
        self.J = findjam.Jam(ngrid=ngrid, ncomms=1, njams=1, move=False, seed=seed)
        self.state = [0]*(2*self.n_units)
        self.action_space = spaces.MultiDiscrete([5]*self.n_units)  # STOP, NORTH, SOUTH, EAST, WEST
        self.observation_space = spaces.MultiDiscrete([self.n_streets_ns,self.n_streets_ew]*self.n_units) # (street & cross-street) on map for each unit
        self.seed_list = self.seed(seed)
        self.reset()
 
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        initial_state = copy.deepcopy(self.state)
        self.steps_this_epoch += 1
        dist = np.zeros([self.n_assets, self.n_units])
        # loop over units and update their positions
        for unit in range(self.n_units):
            ns = 0  # default if not set to +/- 1
            ew = 0  # default if not set to +/- 1
            if IDX_TO_DIR[action[unit]] == 'north':
                ns = 1
            elif IDX_TO_DIR[action[unit]] == 'south':
                ns = -1
            elif IDX_TO_DIR[action[unit]] == 'east':
                ew = 1
            elif IDX_TO_DIR[action[unit]] == 'west':
                ew = -1
            else:
                assert IDX_TO_DIR[action[unit]] == 'stop'
            u_ns = unit*2
            u_ew = unit*2 + 1
            self.state[u_ns] += ns
            self.state[u_ew] += ew
            if self.state[u_ns] < 0:
                self.state[u_ns] = 0
            elif self.state[u_ns] >= self.n_streets_ns:
                self.state[u_ns] = self.n_streets_ns - 1
            if self.state[u_ew] < 0:
                self.state[u_ew] = 0
            elif self.state[u_ew] >= self.n_streets_ew:
                self.state[u_ew] = self.n_streets_ew - 1
            # loop over assets and update distances to assets
            for asset in range(self.n_assets):
                a_ns = asset*2
                a_ew = asset*2 + 1
                dist[asset, unit] = np.abs(self.state[u_ns] - self.assets[a_ns]) + np.abs(self.state[u_ew] - self.assets[a_ew]) 
        observation = tuple(self.state)
        D = dist.min(axis=1).max()
        reward = 10 if D == 0 else 1./D
        done = (self.steps_this_epoch >= self.max_steps)
        info = {'initial_state': initial_state, 'state': copy.deepcopy(self.state), 'assets': self.assets}
        return observation, reward, done, info
 
    def reset(self):
        self.state = []
        for unit in range(self.n_units):
            start_ns = self.np_random.choice(self.start_streets_ns)
            start_ew = self.np_random.choice(self.n_streets_ew)
            self.state.extend([start_ns, start_ew])
        self.steps_this_epoch = 0
        return tuple(self.state)
 
    def render(self, mode='human', close=False):
        assert mode == 'human'
        grid = ('. '*self.n_streets_ew+'\n')*self.n_streets_ns
        gridlist = list(grid)
        for asset in range(self.n_assets):
            asset_ns = self.assets[asset*2]
            asset_ew = self.assets[asset*2+1]
            gridlist[(self.n_streets_ns-1-asset_ns)*(2*self.n_streets_ns+1)+2*asset_ew] = 'A'
        for unit in range(self.n_units):
            unit_ns = self.state[unit*2]
            unit_ew = self.state[unit*2+1]
            if (gridlist[(self.n_streets_ns-1-unit_ns)*(2*self.n_streets_ns+1)+2*unit_ew] == 'A'
                    or gridlist[(self.n_streets_ns-1-unit_ns)*(2*self.n_streets_ns+1)+2*unit_ew] == 'O'):
                gridlist[(self.n_streets_ns-1-unit_ns)*(2*self.n_streets_ns+1)+2*unit_ew] = 'O'
            else:
                gridlist[(self.n_streets_ns-1-unit_ns)*(2*self.n_streets_ns+1)+2*unit_ew] = 'U'
        grid = ''.join(gridlist)
        print(grid)

