#!/usr/bin/env python3
import os
import time
import math
import ptan
import gym
import argparse
from tensorboardX import SummaryWriter

from lib import model, common

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F


ENV_ID = "uav-v0"
GAMMA = 0.99
REWARD_STEPS = 2
BATCH_SIZE = 32
LEARNING_RATE = 5e-5
ENTROPY_BETA = 1e-4

TEST_ITERS = 1000


def test_net(net, env, count=40, device="cpu"):
    rewards = 0.0
    steps = 0
    for _ in range(count):
        obs = env.reset()
        while True:
            obs_v = ptan.agent.float32_preprocessor([obs])
            obs_v = obs_v.to(device)
            net_obs_v = net(obs_v)
            mean_x_v = net_obs_v[0]
            mean_x_v = net_obs_v[1]
            action = {'COMM': {'destx': mean_x_v.squeeze(dim=0).data.cpu().numpy().clip(0, 7), \ # TODO Generalize beyond 8x8
                'desty': mean_y_v.squeeze(dim=0).data.cpu().numpy().clip(0, 7), 'speed': 1}}  # TODO squeeze? Check dims
            obs, reward, done, _ = env.step(action)  #TODO Check OK
            rewards += reward
            steps += 1
            if done:
                break
    return rewards / count, steps / count

