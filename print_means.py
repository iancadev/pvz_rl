import numpy as np

pg_score = np.load('gp_reprod_rewards.npy')
pg_iter = np.load('gp_reprod_iterations.npy')

print("PG   final mean scor", pg_score[-1])
print("PG   final mean iter", pg_iter[-1])

dqn_score = np.load('dqn_reprod_real_rewards.npy')
dqn_iter = np.load('dqn_reprod_real_iterations.npy')

print("DQN  final mean scor", dqn_score[-1])
print("DQN  final mean iter", dqn_iter[-1])

ddqn_score = np.load('ddqn_reprod_real_rewards.npy')
ddqn_iter = np.load('ddqn_reprod_real_iterations.npy')

print("DDQN final mean scor", ddqn_score[-1])
print("DDQN final mean iter", ddqn_iter[-1])


# EVALUATE AC AGENT
from agents import evaluate
import gym
import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="gym")
from itertools import count
import numpy as np
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from pvz import config
import matplotlib.pyplot as plt
np.bool8 = np.bool


from agents import ACAgent3, TrainerAC3, evaluate

if __name__ == "__main__":

    env = TrainerAC3(render=False,max_frames = 400)
    agent = ACAgent3(
        input_size = env.num_observations(),
        possible_actions=env.get_actions()
    )
    agent.load("ac_reprod(policy)", "ac_reprod(valuen)")
    avg_score, avg_iter = evaluate(env, agent, n_iter=1000, verbose=False)
    print("AC mean score", avg_score)
    print("AC mean iters", avg_iter)




