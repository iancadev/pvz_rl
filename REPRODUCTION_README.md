# 🎮 Plants vs Zombies RL - Complete Reproduction Guide

This is a working fork of the Plants vs Zombies RL codebase that can reproduce all paper results with a single command.

## 🚀 Quick Start

```bash
git clone https://github.com/chrishwiggins/pvz_rl.git
cd pvz_rl
make
```

That's it! The `make` command will:
- Install all dependencies
- Set up the environment
- Train all agents
- Evaluate performance
- Generate all figures and tables

## 📋 Available Targets

| Target | Description | Time |
|--------|-------------|------|
| `make` | Reproduce everything | ~6-8 hours |
| `make quick` | Quick demo with pre-trained agents | ~1 minute |
| `make test` | Test installation | ~30 seconds |
| `make figures` | Generate figures only | ~5 minutes |
| `make tables` | Generate tables only | ~2 minutes |
| `make train` | Train all agents | ~6 hours |
| `make evaluate` | Evaluate agents | ~30 minutes |
| `make clean` | Clean all generated files | ~5 seconds |

## 📁 Output Structure

After running `make`, you'll find:

```
results/
├── figures/
│   ├── ddqn_training_rewards.png
│   ├── ddqn_training_iterations.png
│   ├── agent_comparison.png
│   ├── learning_comparison.png
│   ├── action_distribution.png
│   └── feature_importance.png
├── tables/
│   ├── performance_table.md
│   ├── performance_table.tex
│   ├── hyperparameters.md
│   ├── environment_specs.md
│   └── training_stats.md
└── models/
    ├── ddqn_reproduction
    ├── dqn_reproduction
    └── ...
```

## 🔧 Requirements

- Python 3.8+
- PyTorch
- OpenAI Gym
- Pygame
- Matplotlib
- NumPy
- Pandas
- SHAP (for feature importance)

All dependencies are installed automatically by `make setup`.

## 🎯 What Gets Reproduced

### Figures
1. **Training Curves**: Learning progress for all agents
2. **Agent Comparison**: Performance comparison across methods
3. **Action Analysis**: Action usage distribution
4. **Feature Importance**: SHAP analysis of DDQN decisions
5. **Learning Comparison**: Side-by-side learning curves

### Tables
1. **Performance Comparison**: Mean scores and survival times
2. **Hyperparameters**: Configuration for each agent type
3. **Environment Specs**: Game environment details
4. **Training Statistics**: Training time and convergence info

## 🐛 Troubleshooting

### Common Issues

**Import errors**: Run `make test` to verify installation
```bash
make test
```

**PyTorch loading errors**: We've fixed the PyTorch 2.6+ compatibility issues
```bash
# This is already handled in the fixed code
weights_only=False
```

**Missing pre-trained agents**: Some evaluations use pre-trained models
```bash
# Check if agent zoo exists
ls agents/agent_zoo/
```

**Environment issues**: Ensure packages are properly installed
```bash
export PYTHONPATH="./pvz:./gym-pvz:$PYTHONPATH"
```

### Performance Notes

- **Full reproduction** takes 6-8 hours (includes training from scratch)
- **Quick demo** uses pre-trained agents (1 minute)
- **Figure generation** is fast once agents are trained
- Training can be interrupted and resumed

## 📚 Paper Reference

This reproduction corresponds to the paper:
[Plants vs Zombies Reinforcement Learning](https://hanadyg.github.io/portfolio/report/INF581_report.pdf)

## 🔄 Incremental Usage

You can run parts individually:

```bash
# Just test with pre-trained agents
make quick

# Only generate figures (if you have training data)
make figures

# Only evaluate (if you have trained models)
make evaluate

# Train specific agent types
echo "my_ddqn_model" | python train_ddqn_agent.py
```

## 🤝 Contributing

This fork includes fixes for:
- ✅ PyTorch 2.6+ compatibility
- ✅ Missing imports
- ✅ Package installation issues
- ✅ Automated figure generation
- ✅ Comprehensive evaluation pipeline

## 📧 Support

If you encounter issues:
1. First try `make clean && make test`
2. Check the troubleshooting section above
3. Verify you have sufficient disk space (~5GB) and time (6-8 hours for full reproduction)

---

**Happy reproducing! 🎉**