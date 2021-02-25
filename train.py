#!/usr/bin/env python3
import os
import time
import math
import ptan
import gym
import argparse
from tensorboardX import SummaryWriter

# from lib import model, common
import model
import math
import common

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

import gym_uav
from common import serialize


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
            # Convert the obs here
            # cpx = obs['COMM']['posx']/7.
            # cpy = obs['COMM']['posy']/7.
            # chh = float(obs['COMM']['hears']['HQ'])
            # cha = float(obs['COMM']['hears']['ASSET'])  # TODO Need more hears for Jammers
            # hpx = obs['HQ']['posx']/7.
            # hpy = obs['HQ']['posy']/7.
            # apx = obs['ASSET']['posx']/7.
            # apy = obs['ASSET']['posy']/7.
            # eg:  {'COMM': {'posx': 1, 'posy': 1, 'hears': {'HQ': True, 'ASSET': False}},
            #         'HQ': {'posx': 0, 'posy': 0, 'hears': {'COMM': True, 'ASSET': False}},
            #      'ASSET': {'posx': 7, 'posy': 7, 'hears': {'COMM': False, 'HQ': False}}}
            # obs_v = ptan.agent.float32_preprocessor([cpx, cpy, chh, cha, hpx, hpy, apx, apy])
            obs_v = ptan.agent.float32_preprocessor(serialize(obs, env.observation_space))
            obs_v = obs_v.to(device)
            net_obs_v = net(obs_v)
            mean_x_v = net_obs_v[0]  # if not in test_net here and next I would sample from the multi-variate normal to get destx, desty, instead of using means
            mean_y_v = net_obs_v[1]
            # eg: {'COMM': {'destx': 0, 'desty': 7, 'speed': 1}}
            action = {'COMM': {'destx': mean_x_v.squeeze(dim=0).data.cpu().numpy().clip(0, 7),  # TODO Generalize beyond 8x8
                'desty': mean_y_v.squeeze(dim=0).data.cpu().numpy().clip(0, 7), 'speed': 1}}  # TODO squeeze? Check dims
            obs, reward, done, _ = env.step(action)  #TODO Check OK
            rewards += reward
            steps += 1
            if done:
                break
    return rewards / count, steps / count

def calc_logprob(mean_x_v, mean_y_v, var_minor_v, var_delta_v, major_axis_angle_v, actions_v):
    pi = torch.Tensor([math.pi])
    mu_v = torch.stack([mean_x_v, mean_y_v], dim=1)
    var_major_v = var_minor_v + var_delta_v
    var_x_v = var_major_v*torch.sin(pi*major_axis_angle_v)**2 + var_minor_v*torch.cos(pi*major_axis_angle_v)**2
    var_y_v = var_major_v*torch.cos(pi*major_axis_angle_v)**2 + var_minor_v*torch.sin(pi*major_axis_angle_v)**2
    cov_xy_v = (var_major_v - var_minor_v)*torch.sin(pi*major_axis_angle_v)*torch.cos(pi*major_axis_angle_v)
    p1 = []
    p2 = []
    p3 = []
    entropy = []
    for idx in range(len(mean_x_v)): 
        cov = torch.tensor([[var_x_v[idx], cov_xy_v[idx]], [cov_xy_v[idx], var_y_v[idx]]])
        U, D, V = torch.svd(cov)
        inverse_singular_values = torch.diagflat(torch.Tensor([1./D[0].clamp(min=1e-3), 1./D[1].clamp(min=1e-3)]))
        singular_values = torch.diagflat(torch.Tensor([D[0].clamp(min=1e-3), D[1].clamp(min=1e3)]))
        p1.append(-1./2.*(actions_v[idx][0:1].T-mu_v[idx]).T@V@inverse_singular_values@U.T@(actions_v[idx][0:1].T-mu_v[idx]))
        p2.append(-torch.log(2.*pi))
        p3.append(-1./2.*torch.logdet(U@singular_values@V.T))
        entropy.append(torch.logdet(2*math.pi*math.e*U@singular_values@V.T))
    p1_v = torch.tensor(p1)
    p2_v = torch.tensor(p2)
    p3_v = torch.tensor(p3)
    ent_v = torch.tensor(entropy)
    return p1_v + p2_v + p3_v, ent_v

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action='store_true', help='Enable CUDA')
    parser.add_argument("-n", "--name", required=True, help="Name of the run")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    save_path = os.path.join("saves", "a2c-" + args.name)
    os.makedirs(save_path, exist_ok=True)

    env = gym.make(ENV_ID)
    test_env = gym.make(ENV_ID)

    net = model.ModelA2C(12, 3).to(device) #TODO make general env.observation_space.shape[0], env.action_space.shape[0]).to(device)
    print(net)

    writer = SummaryWriter(comment="-a2c_" + args.name)
    agent = model.AgentA2C(net, device=device, env=env)
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, GAMMA, steps_count=REWARD_STEPS)

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    batch = []
    best_reward = None
    with ptan.common.utils.RewardTracker(writer) as tracker:
        with ptan.common.utils.TBMeanTracker(writer, batch_size=10) as tb_tracker:
            for step_idx, exp in enumerate(exp_source):
                rewards_steps = exp_source.pop_rewards_steps()
                if rewards_steps:
                    rewards, steps = zip(*rewards_steps)
                    tb_tracker.track("episode_steps", steps[0], step_idx)
                    tracker.reward(rewards[0], step_idx)

                if step_idx % TEST_ITERS == 0:
                    ts = time.time()
                    rewards, steps = test_net(net, test_env, device=device)
                    print("Test done is %.2f sec, reward %.3f, steps %d" % (
                        time.time() - ts, rewards, steps))
                    writer.add_scalar("test_reward", rewards, step_idx)
                    writer.add_scalar("test_steps", steps, step_idx)
                    if best_reward is None or best_reward < rewards:
                        if best_reward is not None:
                            print("Best reward updated: %.3f -> %.3f" % (best_reward, rewards))
                            name = "best_%+.3f_%d.dat" % (rewards, step_idx)
                            fname = os.path.join(save_path, name)
                            torch.save(net.state_dict(), fname)
                        best_reward = rewards

                batch.append(exp)
                if len(batch) < BATCH_SIZE:
                    continue

                states_v, actions_v, vals_ref_v = \
                    common.unpack_batch_a2c(
                        batch, net, device=device,
                        last_val_gamma=GAMMA ** REWARD_STEPS,
                        env=env)
                batch.clear()

                optimizer.zero_grad()
                # mu_v, var_v, value_v = net(states_v)
                mean_x_v, mean_y_v, var_minor_v, var_delta_v, major_axis_angle_v, value_v = net(states_v)

                loss_value_v = F.mse_loss(
                    value_v.squeeze(-1), vals_ref_v)

                adv_v = vals_ref_v.unsqueeze(dim=-1) - value_v.detach()
                log_prob_v, ent_v = calc_logprob(mean_x_v, mean_y_v, var_minor_v, var_delta_v, major_axis_angle_v, actions_v)
                log_prob_v  = adv_v * log_prob_v
                #    mu_v, var_v, actions_v)
                #    def calc_logprob(mu_v, cov_v, actions_v):
                loss_policy_v = -log_prob_v.mean()
                entropy_loss_v = ENTROPY_BETA * ent_v.mean()

                loss_v = loss_policy_v + entropy_loss_v + \
                         loss_value_v
                loss_v.backward()
                optimizer.step()

                tb_tracker.track("advantage", adv_v, step_idx)
                tb_tracker.track("values", value_v, step_idx)
                tb_tracker.track("batch_rewards", vals_ref_v, step_idx)
                tb_tracker.track("loss_entropy", entropy_loss_v, step_idx)
                tb_tracker.track("loss_policy", loss_policy_v, step_idx)
                tb_tracker.track("loss_value", loss_value_v, step_idx)
                tb_tracker.track("loss_total", loss_v, step_idx)

