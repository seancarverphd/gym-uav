import copy
import ptan
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from common import serialize

HID_SIZE = 128

class ModelA2C(nn.Module):
    def __init__(self, obs_size, act_size):
        super(ModelA2C, self).__init__()  # inherit methods from torch.nn.Module

        self.base = nn.Sequential(
            nn.Linear(obs_size, HID_SIZE),
            nn.ReLU(),
        )
        self.mean_x = nn.Sequential(
            nn.Linear(HID_SIZE, 1),
            nn.Tanh(),
        )
        self.mean_y = nn.Sequential(
            nn.Linear(HID_SIZE, 1),
            nn.Tanh(),
        )
        self.var_minor = nn.Sequential(
            nn.Linear(HID_SIZE, 1),
            nn.Softplus(),
        )
        self.var_delta = nn.Sequential(
            nn.Linear(HID_SIZE, 1),
            nn.Softplus(),
        )
        self.major_axis_angle = nn.Sequential(
            nn.Linear(HID_SIZE, 1),
            nn.Tanh(),
        )
        self.value = nn.Linear(HID_SIZE, 1)

    def forward(self, x):
        base_out = self.base(x)
        return self.mean_x(base_out), self.mean_y(base_out), \
                self.var_minor(base_out), self.var_delta(base_out), \
                self.major_axis_angle(base_out), \
                self.value(base_out)  # Here can translate variables to in range


class AgentA2C(ptan.agent.BaseAgent):
    def __init__(self, net=None, device="cpu", env=None):
        self.env = env
        if net is not None:
            self.net = net
        else:
            n_obs = len(serialize(self.env.reset(), self.env.observation_space))
            n_actions = len(serialize(self.env.example_action(), self.env.action_space))
            self.net = ModelA2C(n_obs, n_actions).to(device)
        self.device = device

    def load(self, fname):
        n_obs = len(serialize(self.env.reset(), self.env.observation_space))
        n_actions = len(serialize(self.env.example_action(), self.env.action_space))
        self.net = ModelA2C(n_obs, n_actions).to(self.device)
        self.net.load_state_dict(torch.load(fname))
        self.net.eval()

    def next_action(self, obs):
        return self([obs], [])[0][0]

    def critic(self, observation, unit='COMM', x=None, y=None):
        obs = copy.deepcopy(observation)
        if x is not None:
            obs[unit]['posx'] = x
        if y is not None:
            obs[unit]['posy'] = y
        obs_v = ptan.agent.float32_preprocessor(serialize(obs, self.env.observation_space))
        obs_v = obs_v.to(self.device)
        _, _, _, _, _, value = self.net(obs_v)
        return value.data.cpu().numpy()[0]

    def visualize_value(self, observation, unit='COMM'):
        xr = np.arange(0, self.env.map.n_streets_ew-1, .05)
        yr = np.arange(0, self.env.map.n_streets_ns-1, .05)
        H = np.zeros([len(xr), len(yr)])
        for i, x in enumerate(xr):
            for j, y in enumerate(yr):
                H[i,j] = self.critic(observation, x=x, y=y)
        plt.imshow(H.T, cmap='hot', interpolation='nearest')  # transpose to get plot right

    def center(self, obs, agent_states=[]):  # states are really observations
        return self([obs], [False])[0][0]

    def reseed(self, seed=None):
        np.random.seed(seed)

    def __call__(self, obs, agent_states=[]):  # states are really observations
        # DRY THIS OUT
        # cpx = obs[0]['COMM']['posx']/7.
        # cpy = obs[0]['COMM']['posy']/7.
        # chh = float(obs[0]['COMM']['hears']['HQ'])
        # cha = float(obs[0]['COMM']['hears']['ASSET'])  # Need more hears for Jammers
        # hpx = obs[0]['HQ']['posx']/7.
        # hpy = obs[0]['HQ']['posy']/7.
        # apx = obs[0]['ASSET']['posx']/7.
        # apy = obs[0]['ASSET']['posy']/7.
        # eg:  {'COMM': {'posx': 1, 'posy': 1, 'hears': {'HQ': True, 'ASSET': False}},
        #         'HQ': {'posx': 0, 'posy': 0, 'hears': {'COMM': True, 'ASSET': False}},
        #      'ASSET': {'posx': 7, 'posy': 7, 'hears': {'COMM': False, 'HQ': False}}}
        # obs_v = ptan.agent.float32_preprocessor([cpx, cpy, chh, cha, hpx, hpy, apx, apy])
        obs_v = ptan.agent.float32_preprocessor(serialize(obs[0], self.env.observation_space))
        obs_v = obs_v.to(self.device)

        mean_x_v, mean_y_v, var_minor_v, var_delta_v, major_axis_angle_v, _ = self.net(obs_v)

        midpoint_x = (self.env.map.n_streets_ew-1.)/2
        midpoint_y = (self.env.map.n_streets_ns-1.)/2

        mean_x = midpoint_x*(1 + mean_x_v.data.cpu().numpy())
        mean_y = midpoint_y*(1 + mean_y_v.data.cpu().numpy())
        var_minor = var_minor_v.data.cpu().numpy()
        var_delta = var_delta_v.data.cpu().numpy()
        cov_theta = np.pi*major_axis_angle_v.data.cpu().numpy()

        var_major = var_minor + var_delta
        var_x = var_major*np.sin(cov_theta)**2 + var_minor*np.cos(cov_theta)**2
        var_y = var_major*np.cos(cov_theta)**2 + var_minor*np.sin(cov_theta)**2
        cov_xy = (var_major - var_minor)*np.sin(cov_theta)*np.cos(cov_theta)
        cov = np.array([[var_x, cov_xy], [cov_xy, var_y]]).reshape([2,2])
        U, D, Vh = np.linalg.svd(cov)  # U and Vh are always orthogonal; will differ if cov is not symmetric; forcing U=Vh below
        d0 = np.clip(D[0], 1e-3, 1e3) # np.inf)
        d1 = np.clip(D[1], 1e-3, 1e3) # np.inf)
        cov2 = U@np.diagflat([d0, d1])@U.T # U.T = Vh unless cov is not symmetric
        mu = np.array([mean_x, mean_y])

        # _ = np.linalg.cholesky(cov2)
        if len(agent_states) > 0 and agent_states[0] is False:
            actions_xy = mu.reshape(2)
        else:
            actions_xy = np.random.multivariate_normal(mu.reshape(2), cov2)
            #            np.array([1,1]), np.array([[1, 0], [0, 1]]))
        actions_x = np.clip(actions_xy[0], 0, self.env.map.n_streets_ew-1.)
        actions_y = np.clip(actions_xy[1], 0, self.env.map.n_streets_ns-1.)
        actions = {'COMM': {'destx': actions_x, 'desty': actions_y, 'speed': 1}}
        return [actions], agent_states

