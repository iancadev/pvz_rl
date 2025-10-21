#!/usr/bin/env python3
"""
Figure 1: Training performance comparison using available training data
"""

import os
import numpy as np
import matplotlib.pyplot as plt

def setup_matplotlib():
    """Configure matplotlib for paper-quality figures"""
    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 12,
        'figure.figsize': (12, 8),
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight'
    })

def generate_figure_1():
    """Generate Figure 1: Training performance comparison"""
    print("📈 Generating Figure 1: Training performance comparison...")

    setup_matplotlib()
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # Load available training data
    data_sources = []

    # DDQN data
    if os.path.exists("ddqn_reproduction_rewards.npy"):
        ddqn_rewards = np.load("ddqn_reproduction_rewards.npy")
        ddqn_iterations = np.load("ddqn_reproduction_iterations.npy")

        # Apply moving average for cleaner visualization
        window = 50
        if len(ddqn_rewards) >= window:
            ddqn_rewards_smooth = np.convolve(ddqn_rewards, np.ones(window)/window, mode='valid')
            ddqn_iterations_smooth = np.convolve(ddqn_iterations, np.ones(window)/window, mode='valid')
            x_ddqn = np.arange(len(ddqn_rewards_smooth))

            ax1.plot(x_ddqn, ddqn_rewards_smooth, label='DDQN', linewidth=2, color='blue')
            ax2.plot(x_ddqn, ddqn_iterations_smooth, label='DDQN', linewidth=2, color='blue')
            data_sources.append("DDQN")

    # DQN data
    if os.path.exists("dqn_reproduction_rewards.npy"):
        dqn_rewards = np.load("dqn_reproduction_rewards.npy")
        dqn_iterations = np.load("dqn_reproduction_iterations.npy")

        # Apply moving average for cleaner visualization
        window = 50
        if len(dqn_rewards) >= window:
            dqn_rewards_smooth = np.convolve(dqn_rewards, np.ones(window)/window, mode='valid')
            dqn_iterations_smooth = np.convolve(dqn_iterations, np.ones(window)/window, mode='valid')
            x_dqn = np.arange(len(dqn_rewards_smooth))

            ax1.plot(x_dqn, dqn_rewards_smooth, label='DQN', linewidth=2, color='red')
            ax2.plot(x_dqn, dqn_iterations_smooth, label='DQN', linewidth=2, color='red')
            data_sources.append("DQN")

    # Quick training data comparison
    if os.path.exists("ddqn_quick_rewards.npy"):
        ddqn_quick_rewards = np.load("ddqn_quick_rewards.npy")
        ddqn_quick_iterations = np.load("ddqn_quick_iterations.npy")
        x_quick_ddqn = np.arange(len(ddqn_quick_rewards))

        ax3.plot(x_quick_ddqn, ddqn_quick_rewards, label='DDQN Quick (200 episodes)',
                linewidth=2, color='lightblue', alpha=0.8)
        ax4.plot(x_quick_ddqn, ddqn_quick_iterations, label='DDQN Quick (200 episodes)',
                linewidth=2, color='lightblue', alpha=0.8)
        data_sources.append("DDQN Quick")

    if os.path.exists("dqn_quick_rewards.npy"):
        dqn_quick_rewards = np.load("dqn_quick_rewards.npy")
        dqn_quick_iterations = np.load("dqn_quick_iterations.npy")
        x_quick_dqn = np.arange(len(dqn_quick_rewards))

        ax3.plot(x_quick_dqn, dqn_quick_rewards, label='DQN Quick (200 episodes)',
                linewidth=2, color='lightcoral', alpha=0.8)
        ax4.plot(x_quick_dqn, dqn_quick_iterations, label='DQN Quick (200 episodes)',
                linewidth=2, color='lightcoral', alpha=0.8)
        data_sources.append("DQN Quick")

    # Configure axes
    ax1.set_xlabel('Training Episodes')
    ax1.set_ylabel('Rewards')
    ax1.set_title('Long Training: Rewards vs Episodes')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Training Episodes')
    ax2.set_ylabel('Iterations Survived')
    ax2.set_title('Long Training: Survival vs Episodes')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3.set_xlabel('Training Episodes')
    ax3.set_ylabel('Rewards')
    ax3.set_title('Quick Training: Rewards vs Episodes')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    ax4.set_xlabel('Training Episodes')
    ax4.set_ylabel('Iterations Survived')
    ax4.set_title('Quick Training: Survival vs Episodes')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    fig.suptitle('Figure 1: Training Performance Comparison',
                fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.savefig('results/figures/figure_1_training_comparison.png')
    plt.close()

    print("✅ Figure 1 generated")
    print(f"📊 Data sources used: {', '.join(data_sources)}")

    return data_sources

if __name__ == "__main__":
    os.makedirs('results/figures', exist_ok=True)
    generate_figure_1()