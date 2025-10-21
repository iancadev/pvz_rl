#!/usr/bin/env python3
"""
Figure 7: Convergence analysis using real training data
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def setup_matplotlib():
    """Configure matplotlib for paper-quality figures"""
    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 12,
        'figure.figsize': (15, 10),
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight'
    })

def calculate_convergence_metrics(data, window_size=100):
    """Calculate convergence metrics from training data"""
    if len(data) < window_size * 2:
        return None, None, None

    # Calculate rolling statistics
    rolling_mean = []
    rolling_var = []
    rolling_std = []

    for i in range(window_size, len(data)):
        window_data = data[i-window_size:i]
        rolling_mean.append(np.mean(window_data))
        rolling_var.append(np.var(window_data))
        rolling_std.append(np.std(window_data))

    return np.array(rolling_mean), np.array(rolling_var), np.array(rolling_std)

def calculate_convergence_trend(data):
    """Calculate convergence trend using linear regression"""
    if len(data) < 10:
        return None, None

    x = np.arange(len(data))
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, data)
    return slope, r_value

def generate_figure_7():
    """Generate Figure 7: Convergence analysis"""
    print("📈 Generating Figure 7: Convergence analysis...")

    setup_matplotlib()
    fig = plt.figure(figsize=(16, 12))

    # Create a grid for subplots
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    analysis_results = []

    # DDQN Analysis
    if os.path.exists("ddqn_reproduction_rewards.npy"):
        ddqn_rewards = np.load("ddqn_reproduction_rewards.npy")
        ddqn_iterations = np.load("ddqn_reproduction_iterations.npy")

        # Calculate convergence metrics for rewards
        mean_rewards, var_rewards, std_rewards = calculate_convergence_metrics(ddqn_rewards)

        if mean_rewards is not None:
            ax1 = fig.add_subplot(gs[0, 0])
            x = np.arange(100, len(ddqn_rewards))

            # Plot rolling variance (convergence indicator)
            ax1.plot(x, var_rewards, color='blue', linewidth=2, alpha=0.8)
            ax1.set_xlabel('Training Episodes')
            ax1.set_ylabel('Rolling Variance (window=100)')
            ax1.set_title('DDQN Rewards: Variance Reduction')
            ax1.grid(True, alpha=0.3)

            # Calculate trend
            slope, r_value = calculate_convergence_trend(var_rewards[-500:])  # Last 500 episodes
            if slope is not None:
                ax1.text(0.05, 0.95, f'Trend slope: {slope:.2e}\nR²: {r_value**2:.3f}',
                        transform=ax1.transAxes, verticalalignment='top',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))

            # Plot rolling mean (performance stabilization)
            ax2 = fig.add_subplot(gs[0, 1])
            ax2.plot(x, mean_rewards, color='blue', linewidth=2, alpha=0.8)
            ax2.set_xlabel('Training Episodes')
            ax2.set_ylabel('Rolling Mean Reward (window=100)')
            ax2.set_title('DDQN Rewards: Mean Performance')
            ax2.grid(True, alpha=0.3)

            analysis_results.append(f"DDQN: Variance trend slope = {slope:.2e}")

    # DQN Analysis
    if os.path.exists("dqn_reproduction_rewards.npy"):
        dqn_rewards = np.load("dqn_reproduction_rewards.npy")
        dqn_iterations = np.load("dqn_reproduction_iterations.npy")

        # Calculate convergence metrics for rewards
        mean_rewards, var_rewards, std_rewards = calculate_convergence_metrics(dqn_rewards)

        if mean_rewards is not None:
            ax3 = fig.add_subplot(gs[1, 0])
            x = np.arange(100, len(dqn_rewards))

            # Plot rolling variance
            ax3.plot(x, var_rewards, color='red', linewidth=2, alpha=0.8)
            ax3.set_xlabel('Training Episodes')
            ax3.set_ylabel('Rolling Variance (window=100)')
            ax3.set_title('DQN Rewards: Variance Reduction')
            ax3.grid(True, alpha=0.3)

            # Calculate trend
            slope, r_value = calculate_convergence_trend(var_rewards[-500:])
            if slope is not None:
                ax3.text(0.05, 0.95, f'Trend slope: {slope:.2e}\nR²: {r_value**2:.3f}',
                        transform=ax3.transAxes, verticalalignment='top',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.8))

            # Plot rolling mean
            ax4 = fig.add_subplot(gs[1, 1])
            ax4.plot(x, mean_rewards, color='red', linewidth=2, alpha=0.8)
            ax4.set_xlabel('Training Episodes')
            ax4.set_ylabel('Rolling Mean Reward (window=100)')
            ax4.set_title('DQN Rewards: Mean Performance')
            ax4.grid(True, alpha=0.3)

            analysis_results.append(f"DQN: Variance trend slope = {slope:.2e}")

    # Comparison Analysis
    ax5 = fig.add_subplot(gs[2, :])

    if os.path.exists("ddqn_reproduction_rewards.npy") and os.path.exists("dqn_reproduction_rewards.npy"):
        ddqn_rewards = np.load("ddqn_reproduction_rewards.npy")
        dqn_rewards = np.load("dqn_reproduction_rewards.npy")

        # Calculate final performance statistics
        ddqn_final_mean = np.mean(ddqn_rewards[-100:])  # Last 100 episodes
        ddqn_final_std = np.std(ddqn_rewards[-100:])
        dqn_final_mean = np.mean(dqn_rewards[-100:])
        dqn_final_std = np.std(dqn_rewards[-100:])

        # Create comparison plot
        agents = ['DDQN', 'DQN']
        means = [ddqn_final_mean, dqn_final_mean]
        stds = [ddqn_final_std, dqn_final_std]

        bars = ax5.bar(agents, means, yerr=stds, capsize=10, alpha=0.7,
                      color=['blue', 'red'], edgecolor='black')
        ax5.set_ylabel('Final Performance (Mean ± Std)')
        ax5.set_title('Convergence Comparison: Final 100 Episodes')
        ax5.grid(True, alpha=0.3)

        # Add value labels
        for bar, mean, std in zip(bars, means, stds):
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 10,
                    f'{mean:.1f}±{std:.1f}',
                    ha='center', va='bottom', fontweight='bold')

        # Statistical test
        t_stat, p_value = stats.ttest_ind(ddqn_rewards[-100:], dqn_rewards[-100:])
        ax5.text(0.5, 0.95, f'T-test: t={t_stat:.3f}, p={p_value:.3f}',
                transform=ax5.transAxes, ha='center', va='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

        analysis_results.append(f"DDQN vs DQN t-test: p={p_value:.3f}")

    # Add overall title
    fig.suptitle('Figure 7: Convergence Analysis',
                fontsize=16, fontweight='bold')

    plt.savefig('results/figures/figure_7_convergence_analysis.png')
    plt.close()

    print("✅ Figure 7 generated")
    for result in analysis_results:
        print(f"📊 {result}")

    return analysis_results

if __name__ == "__main__":
    os.makedirs('results/figures', exist_ok=True)
    generate_figure_7()