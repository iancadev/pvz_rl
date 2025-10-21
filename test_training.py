#!/usr/bin/env python3
"""
Quick test of DDQN training
"""

import gym
from agents import experienceReplayBuffer, DDQNAgent, QNetwork

def test_training():
    print("🧪 Testing DDQN training...")
    try:
        # Create environment and agent
        env = gym.make('gym_pvz:pvz-env-v2')
        buffer = experienceReplayBuffer(memory_size=1000, burn_in=100)  # Smaller for testing
        net = QNetwork(env, device='cpu', use_zombienet=False, use_gridnet=False)
        agent = DDQNAgent(env, net, buffer, n_iter=10, batch_size=32)

        print("✅ Agent created successfully")

        # Test training for just a few steps
        agent.train(max_episodes=5, evaluate_frequency=10, evaluate_n_iter=2)

        print("✅ Training completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Training test failed: {e}")
        return False

if __name__ == "__main__":
    test_training()