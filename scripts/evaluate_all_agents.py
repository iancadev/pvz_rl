#!/usr/bin/env python3
"""
Evaluate all trained agents and generate performance metrics
"""

import os
import sys
import json
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import gym
import torch
from agents import evaluate, PlayerV2, ReinforceAgentV2, PlayerQ, PlayerQ_DQN, ACAgent3, TrainerAC3

def evaluate_agent(agent_type, agent_path=None, n_episodes=1000):
    """Evaluate a single agent"""
    print(f"Evaluating {agent_type} agent...")

    if agent_type == "DDQN":
        env = PlayerQ(render=False)
        if agent_path:
            agent = torch.load(agent_path, weights_only=False)
        else:
            agent = torch.load("agents/agent_zoo/dfq5_epsexp", weights_only=False)

    elif agent_type == "DQN":
        env = PlayerQ_DQN(render=False)
        if agent_path:
            agent = torch.load(agent_path, weights_only=False)
        else:
            agent = torch.load("agents/agent_zoo/dfq5_dqn", weights_only=False)

    elif agent_type == "Reinforce":
        from pvz import config
        env = PlayerV2(render=False, max_frames=500 * config.FPS)
        agent = ReinforceAgentV2(
            input_size=env.num_observations(),
            possible_actions=env.get_actions()
        )
        if agent_path:
            agent.load(agent_path)
        else:
            agent.load("agents/agent_zoo/dfp5")

    elif agent_type == "ActorCritic":
        from pvz import config
        env = TrainerAC3(render=False, max_frames=500 * config.FPS)
        agent = ACAgent3(
            input_size=env.num_observations(),
            possible_actions=env.get_actions()
        )
        if agent_path:
            agent.load(agent_path + "_policy", agent_path + "_value")
        else:
            agent.load("agents/agent_zoo/ac_policy_v1", "agents/agent_zoo/ac_value_v1")

    # Evaluate agent
    avg_score, avg_iter = evaluate(env, agent, n_iter=n_episodes)

    return {
        'agent_type': agent_type,
        'avg_score': avg_score,
        'avg_iterations': avg_iter,
        'n_episodes': n_episodes
    }

def main():
    """Evaluate all agents and save results"""
    os.makedirs('results/tables', exist_ok=True)

    # Agent configurations
    agents_to_evaluate = [
        ("DDQN", None),
        ("DQN", None),
        ("Reinforce", None),
        ("ActorCritic", None)
    ]

    # Check for newly trained agents
    if os.path.exists("ddqn_reproduction"):
        agents_to_evaluate.append(("DDQN", "ddqn_reproduction"))
    if os.path.exists("dqn_reproduction"):
        agents_to_evaluate.append(("DQN", "dqn_reproduction"))

    results = []

    for agent_type, agent_path in agents_to_evaluate:
        try:
            result = evaluate_agent(agent_type, agent_path, n_episodes=100)  # Reduced for speed
            results.append(result)
            print(f"✅ {agent_type}: Score={result['avg_score']:.2f}, Iterations={result['avg_iterations']:.2f}")
        except Exception as e:
            print(f"❌ Failed to evaluate {agent_type}: {e}")

    # Save results
    df = pd.DataFrame(results)
    df.to_csv('results/tables/agent_performance.csv', index=False)

    with open('results/tables/agent_performance.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n📊 Evaluation complete! Results saved to results/tables/")
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()