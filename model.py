import ptan
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

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
    def __init__(self, net, device="cpu"):
        self.net = net
        self.device = device

    def __call__(self, obs, agent_states):  # states are really observations
        # DRY THIS OUT
        cpx = obs[0]['COMM']['posx']/7.
        cpy = obs[0]['COMM']['posy']/7.
        chh = float(obs[0]['COMM']['hears']['HQ'])
        cha = float(obs[0]['COMM']['hears']['ASSET'])  # TODO Need more hears for Jammers
        hpx = obs[0]['HQ']['posx']/7.
        hpy = obs[0]['HQ']['posy']/7.
        apx = obs[0]['ASSET']['posx']/7.
        apy = obs[0]['ASSET']['posy']/7.
        # eg:  {'COMM': {'posx': 1, 'posy': 1, 'hears': {'HQ': True, 'ASSET': False}},
        #         'HQ': {'posx': 0, 'posy': 0, 'hears': {'COMM': True, 'ASSET': False}},
        #      'ASSET': {'posx': 7, 'posy': 7, 'hears': {'COMM': False, 'HQ': False}}}
        obs_v = ptan.agent.float32_preprocessor([cpx, cpy, chh, cha, hpx, hpy, apx, apy])
        obs_v = obs_v.to(self.device)

        mean_x_v, mean_y_v, var_minor_v, var_delta_v, major_axis_angle_v, _ = self.net(obs_v)

        mean_x = 4. + 4.*mean_x_v.data.cpu().numpy()  # Based on an 8x8 grid TODO Generalize
        mean_y = 4. + 4.*mean_y_v.data.cpu().numpy()  # Based on an 8x8 grid TODO Generalize
        var_minor = var_minor_v.data.cpu().numpy()
        var_delta = var_delta_v.data.cpu().numpy()
        cov_theta = np.pi*major_axis_angle_v.data.cpu().numpy()

        var_major = var_minor + var_delta
        var_x = var_major*np.sin(cov_theta)**2 + var_minor*np.cos(cov_theta)**2
        var_y = var_major*np.cos(cov_theta)**2 + var_minor*np.sin(cov_theta)**2
        cov_xy = (var_major - var_minor)*np.sin(cov_theta)*np.cos(cov_theta)
        cov = np.array([[var_x, cov_xy], [cov_xy, var_y]]).reshape([2,2])
        U, D, Vh = np.linalg.svd(cov) 
        d0 = np.clip(D[0], 1e-3, np.inf)
        d1 = np.clip(D[1], 1e-3, np.inf)
        cov2 = U@np.diagflat([d0, d1])@Vh
        mu = np.array([mean_x, mean_y])

        # _ = np.linalg.cholesky(cov2)
        actions = np.random.multivariate_normal(mu.reshape(2), cov2)
        #        np.array([1,1]), np.array([[1, 0], [0, 1]])) # 
        print("actions", str(actions))
        actions = np.clip(actions, 0, 7)  # Based on an 8x8 grid TODO Generalize
        return actions, agent_states

