import numpy as np
import matplotlib.pyplot as plt
import sys
import torch
import os


if __name__ == "__main__":
    name = sys.argv[1]
    rewards = np.load(name+"_rewards.npy")
    iterations = np.load(name+"_iterations.npy")
    loss = torch.load(name+"_loss", weights_only=False)
    real_rewards = np.load(name+"_real_rewards.npy")
    real_iterations = np.load(name+"_real_iterations.npy")

    n_iter = rewards.shape[0]
    n_record = real_rewards.shape[0]
    record_period = n_iter//n_record
    slice_size = 500

    rewards = np.reshape(rewards, (n_iter//slice_size, slice_size)).mean(axis=1)
    iterations = np.reshape(iterations, (n_iter//slice_size, slice_size)).mean(axis=1)
    loss = np.reshape(loss, (n_iter//slice_size, slice_size)).mean(axis=1)

    x = list(range(0, n_iter, slice_size))
    xx = list(range(1, n_iter, record_period))

    # Create results directory if it doesn't exist
    os.makedirs('results/figures', exist_ok=True)

    # Rewards plot
    plt.figure(figsize=(10, 6))
    plt.plot(x, rewards, label='Training Rewards', alpha=0.7)
    plt.plot(xx, real_rewards, color='red', label='Evaluation Rewards', linewidth=2)
    plt.xlabel('Training Episodes')
    plt.ylabel('Average Reward')
    plt.title(f'{name.upper()} Training Progress - Rewards')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'results/figures/{name}_training_rewards.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Iterations plot
    plt.figure(figsize=(10, 6))
    plt.plot(x, iterations, label='Training Iterations', alpha=0.7)
    plt.plot(xx, real_iterations, color='red', label='Evaluation Iterations', linewidth=2)
    plt.xlabel('Training Episodes')
    plt.ylabel('Average Game Length')
    plt.title(f'{name.upper()} Training Progress - Game Length')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'results/figures/{name}_training_iterations.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Loss plot
    plt.figure(figsize=(10, 6))
    plt.plot(x, loss, label='Training Loss', alpha=0.7)
    plt.xlabel('Training Episodes')
    plt.ylabel('Loss')
    plt.title(f'{name.upper()} Training Progress - Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'results/figures/{name}_training_loss.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Training plots saved to results/figures/ for {name}")