#!/usr/bin/env python3
"""
Quick evaluation script using pre-trained agents for testing
"""

import os
import sys
import json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from agents import evaluate, PlayerQ

def main():
    """Quick evaluation with pre-trained DDQN agent"""
    print("🎮 Running quick evaluation with pre-trained DDQN agent...")

    try:
        env = PlayerQ(render=False)
        agent = torch.load("agents/agent_zoo/dfq5_epsexp", weights_only=False)

        # Quick evaluation with fewer episodes
        avg_score, avg_iter = evaluate(env, agent, n_iter=10)

        result = {
            'agent_type': 'DDQN (pre-trained)',
            'avg_score': avg_score,
            'avg_iterations': avg_iter,
            'n_episodes': 10
        }

        os.makedirs('results/tables', exist_ok=True)
        with open('results/tables/quick_test.json', 'w') as f:
            json.dump(result, f, indent=2)

        print(f"✅ Quick test complete!")
        print(f"   Score: {avg_score:.2f}")
        print(f"   Survival: {avg_iter:.2f} frames")

        return True

    except Exception as e:
        print(f"❌ Quick evaluation failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)