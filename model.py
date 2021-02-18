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
            nn.Linear(HID_SIZE, act_size),
            nn.Tanh(),
        )
        self.mean_y = nn.Sequential(
            nn.Linear(HID_SIZE, act_size),
            nn.Tanh(),
        )
        self.var_minor = nn.Sequential(
            nn.Linear(HID_SIZE, act_size),
            nn.Softplus(),
        )
        self.var_delta = nn.Sequential(
            nn.Linear(HID_SIZE, act_size),
            nn.Softplus(),
        )
        self.major_axis_angle = nn.Sequential(
            nn.Linear(HID_SIZE, act_size),
            nn.Tanh(),
        )
        self.value = nn.Linear(HID_SIZE, 1)

    def forward(self, x):
        base_out = self.base(x)
        return self.mean_x(base_out), self.mean_y(base_out), \
                self.var_minor(base_out), self.var_delta(base_out), \
                self.major_axis_angle(base_out)
                self.value(base_out)  # Here can translate variables to in range


class AgentA2C(ptan.agent.BaseAgent):
    def __init__(self, net, device="cpu"):
        self.net = net
        self.device = device

    def __call__(self, states, agent_states):  # states are really observations
        states_v = ptan.agent.float32_preprocessor(states)  # states are really observations
        states_v = states_v.to(self.device)  # states are really observations

        mean_x_v, mean_y_v, var_minor_v, var_delta_v, major_axis_angle_v, _ = self.net(states_v)  # states are really observations

        mean_x = 4. + 4.*mean_x_v.data.cpu().numpy()  # Based on an 8x8 grid TODO Generalize
        mean_y = 4. + 4.*mean_y_v.data.cpu().numpy()  # Based on an 8x8 grid TODO Generalize
        var_minor = var_minor_v.data.cpu().numpy()
        var_delta = var_delta_v.data.cpu().numpy()
        cov_theta = np.pi*major_axis_angle_x_v.data.cpu().numpy()

        var_major = var_minor + var_delta
        var_x = var_major*sin(cov_theta)**2 + var_minor*cos(cov_theta)**2
        var_y = var_major*cos(cov_theta)**2 + var_minor*sin(cov_theta)**2
        cov_xy = (var_major - var_minor)*sin(cov_theta)*cos(cov_theta)
        cov = numpy.array([[var_x, cov_xy], [cov_xy, var_y]])
        mu = numpy.array([mean_x, mean_y])

        actions = np.random.multivariate_normal(mu, cov)
        actions = np.clip(actions, 0, 7)  # Based on an 8x8 grid TODO Generalize
        return actions, agent_states

