import matplotlib.pyplot as plt
import numpy as np
import os

DDQN='DDQN'

if __name__ == '__main__':
    
    os.makedirs('results/figures', exist_ok=True)
    NUM_ITER = 100000

    # Score plot
    iter_plt = np.load("dqn_quick_trash_real_iterations.npy")
    score_plt = np.load("dqn_quick_trash_real_rewards.npy")
    increments = int(NUM_ITER/len(iter_plt))
    ranges = range(increments, NUM_ITER+1, increments)
    
    plt.figure(figsize=(10, 6))
    plt.plot(ranges, score_plt, label=f'{DDQN} Score')
    plt.xlabel('Training Episodes')
    plt.ylabel('Average Score')
    plt.title(f'{DDQN} Training Progress - Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'results/figures/{DDQN}_training_score.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Iterations plot
    plt.figure(figsize=(10, 6))
    plt.plot(ranges, iter_plt, label=f'{DDQN} Iterations', color='orange')
    plt.xlabel('Training Episodes')
    plt.ylabel('Average Game Length')
    plt.title(f'{DDQN} Training Progress - Game Length')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'results/figures/{DDQN}_training_iterations}.png', dpi=300, bbox_inches='tight')
    plt.close()