import gym
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
# from game_render import render


def evaluate(env, agent, n_iter=1000, verbose = True):
    sum_score = 0
    sum_iter = 0
    score_hist = []
    iter_hist = []
    n_iter = n_iter
    actions = []

    for episode_idx in range(n_iter):
        if verbose:
            print("\r{}/{}".format(episode_idx, n_iter), end="")
        
        # play episodes
        summary = env.play(agent)
        summary['score'] = np.sum(summary["rewards"])

        score_hist.append(summary['score'])
        iter_hist.append(summary.get('episode_length', len(summary['rewards'])))

        sum_score += summary['score']
        sum_iter += summary.get('episode_length', len(summary['rewards']))
        
        # if env.env._scene._chrono >= 1000:
        #    render_info = env.env._scene._render_info
        #    render(render_info)
        #    input()
        actions.append(summary['actions'])

    actions = np.concatenate(actions)
    plant_action = np.mod(actions - 1, 4)
    if verbose:
        import os
        import time

        # Create unique timestamp for filenames
        timestamp = int(time.time())

        # Create results directory if it doesn't exist
        os.makedirs('results/figures', exist_ok=True)

        # Plot of the score
        plt.figure(figsize=(10, 6))
        plt.hist(score_hist, bins=30, alpha=0.7, edgecolor='black')
        plt.title("Score per play over {} plays".format(n_iter))
        plt.xlabel('Score')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        plt.savefig(f'results/figures/score_distribution_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Plot of the iterations
        plt.figure(figsize=(10, 6))
        plt.hist(iter_hist, bins=30, alpha=0.7, edgecolor='black', color='orange')
        plt.title("Survived frames per play over {} plays".format(n_iter))
        plt.xlabel('Frames Survived')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        plt.savefig(f'results/figures/survival_distribution_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Plot of the action
        plt.figure(figsize=(12, 6))
        plt.hist(np.concatenate(actions), np.arange(0, config.N_LANES * config.LANE_LENGTH * 4 + 2) -0.5, density=True, alpha=0.7, edgecolor='black')
        plt.title("Action usage density over {} plays".format(n_iter))
        plt.xlabel('Action ID')
        plt.ylabel('Density')
        plt.grid(True, alpha=0.3)
        plt.savefig(f'results/figures/action_distribution_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Plot of plant usage
        plt.figure(figsize=(8, 6))
        plt.hist(plant_action, np.arange(0,5) - 0.5, density=True, alpha=0.7, edgecolor='black', color='green')
        plt.title("Plant usage density over {} plays".format(n_iter))
        plt.xlabel('Plant Type')
        plt.ylabel('Density')
        plt.xticks([0, 1, 2, 3, 4], ['Sunflower', 'Peashooter', 'Wallnut', 'Potatomine', 'Other'])
        plt.grid(True, alpha=0.3)
        plt.savefig(f'results/figures/plant_distribution_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()

    return sum_score/n_iter, sum_iter/n_iter
