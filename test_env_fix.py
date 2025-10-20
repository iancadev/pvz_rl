#!/usr/bin/env python3
"""
Test script to verify the environment fix works
"""

import gym
import gym_pvz

def test_env():
    """Test that the environment works correctly"""
    print("🧪 Testing PVZ environment fix...")

    try:
        # Create environment
        env = gym.make('gym_pvz:pvz-env-v2')
        print("✅ Environment created successfully")

        # Test reset
        obs = env.reset()
        print(f"✅ Reset successful, observation shape: {obs.shape}")
        print(f"✅ Observation space: {env.observation_space}")
        print(f"✅ Action space: {env.action_space}")

        # Test step
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print(f"✅ Step successful, observation shape: {obs.shape}")

        print("🎉 Environment fix verified! Training should now work.")
        return True

    except Exception as e:
        print(f"❌ Environment test failed: {e}")
        return False

if __name__ == "__main__":
    test_env()