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


def train(env, agent, n_iter=None, n_record=500, n_save=1000):
    # Get iteration count from environment variable, default to 200 for quick demo
    if n_iter is None:
        n_iter = int(os.environ.get('PVZ_EPISODES', 200))
    sum_score = 0
    sum_iter = 0
    score_plt = []
    iter_plt = []
    eval_score_plt = []
    eval_iter_plt = []
    save = False
    best_score = None

    for episode_idx in range(n_iter):

        # play episodes
        summary = env.play(agent)
        summary['score'] = np.sum(summary["rewards"])

        sum_score += summary['score']
        sum_iter += summary.get('episode_length', len(summary['rewards']))

        # Update agent
        agent.update(summary["observations"],summary["actions"],summary["rewards"])

        if (episode_idx%n_record == n_record-1):
            if save:
                if sum_score >= best_score:
                    agent.save(nn_name1, nn_name2)
                    best_score = sum_score
            print("---Episode {}, mean score {}".format(episode_idx,sum_score/n_record))
            print("---n_iter {}".format(sum_iter/n_record))
            score_plt.append(sum_score/n_record)
            iter_plt.append(sum_iter/n_record)
            sum_iter = 0
            sum_score = 0
            # input()
        if not save:
            if (episode_idx%n_save == n_save-1):
                s = input("Save? (y/n): ")
                if (s=='y'):
                    save = True
                    best_score = 0
                    nn_name1 = input("Save name for policy net: ")
                    nn_name2 = input("Save name for value net: ")

    # Create results directory if it doesn't exist
    os.makedirs('results/figures', exist_ok=True)

    # Score plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(n_record, n_iter+1, n_record), score_plt, label='Actor-Critic Score')
    plt.xlabel('Training Episodes')
    plt.ylabel('Average Score')
    plt.title('Actor-Critic Training Progress - Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('results/figures/actor_critic_training_score.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Iterations plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(n_record, n_iter+1, n_record), iter_plt, label='Actor-Critic Iterations', color='green')
    plt.xlabel('Training Episodes')
    plt.ylabel('Average Game Length')
    plt.title('Actor-Critic Training Progress - Game Length')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('results/figures/actor_critic_training_iterations.png', dpi=300, bbox_inches='tight')
    plt.close()


# Import your agent
from agents import ACAgent3, TrainerAC3

if __name__ == "__main__":

    env = TrainerAC3(render=False,max_frames = 400)
    agent = ACAgent3(
        input_size = env.num_observations(),
        possible_actions=env.get_actions()
    )
    train(env, agent)




