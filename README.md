# pvz_rl
Plants vs. Zombies remake and RL agents

## Requirements

The following python libraries are required: pytorch, gym, pygame (and shap to evaluate feature importance if wanted).

## Installation/Setup

The game engine we developed and the Open AI gym environment are both encapsulated in python libraries.
We provide a Makefile that automates the entire setup and reproduction process.

### Quick Start with Makefile

For a quick demo with pre-trained agents (recommended):
```
make all
```

For complete paper reproduction with full training:
```
make full
```

To see all available commands:
```
make help
```

### Manual Installation (Alternative)

If you prefer manual installation:

```
git clone https://github.com/inf581-pvz-anonymous/pvz_rl.git
cd pvz_rl
make setup install
```

Everything related to the game balance and FPS configuration is in the pvz library we developed. In particular, config.py, entities/zombies.WaveZombieSpawner.py and the main files containing balance data. For plant and zombie characteristics, check the files in the entities/plants/ and entities/zombies/ folders.

You can test your installation by running:
```
make test
```

## Usage Examples

### Quick Demo (Recommended)

To quickly see the system in action with pre-trained agents:
```
make quick
```

(The in-class demo likely used `python game_render.py`, if I remember correctly)


### Training Agents

For quick training (200 episodes, ~5 minutes):
```
make train-quick
```

For full paper reproduction training (100K episodes, ~30-60 minutes):
```
make train-full
```

Individual training scripts are also available:
- `python3 train_ddqn_agent.py` - Train DDQN agent
- `python3 train_dqn_agent.py` - Train DQN agent
- `python3 train_actor_critic_agent.py` - Train Actor-Critic agent
- `python3 train_policy_agent.py` - Train Policy agent

### Evaluation and Analysis

Evaluate all trained agents:
```
make evaluate
```

Generate all figures and tables:
```
make figures
make tables
```

### Visualization

**Note**: The game visualization (`python game_render.py`) currently has compatibility issues and may not work. Use the evaluation scripts instead to see agent performance.

To visualize a game with higher FPS (more fluid motions), change the FPS variable in pvz/pvz/config.py. This may have a slight impact on the behavior of some agents.

### Feature Importance

Feature importance analysis is included in the evaluation pipeline. For manual analysis:
```
python3 script_feature_importance.py
```

### Available Make Commands

Run `make help` to see all available commands:
- `make all` - Quick demo (default)
- `make full` - Complete reproduction
- `make setup` - Install dependencies
- `make install` - Install packages
- `make train-quick` - Quick training
- `make train-full` - Full training
- `make evaluate` - Evaluate agents
- `make figures` - Generate figures
- `make clean` - Clean generated files
- `make test` - Test installation
