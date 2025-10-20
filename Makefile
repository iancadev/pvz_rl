# Plants vs Zombies RL - Complete Reproduction Pipeline
# Run 'make' to reproduce all figures and tables from the paper

.PHONY: all setup install train evaluate figures tables clean help

# Python path setup
export PYTHONPATH := $(shell pwd)/pvz:$(shell pwd)/gym-pvz:$(PYTHONPATH)

# Default target - reproduce everything
all: setup install train evaluate figures tables
	@echo "🎉 All results reproduced! Check the results/ directory for figures and tables."

# Help target
help:
	@echo "Plants vs Zombies RL - Reproduction Pipeline"
	@echo ""
	@echo "Targets:"
	@echo "  all       - Reproduce all results (default)"
	@echo "  setup     - Install dependencies and setup environment"
	@echo "  install   - Install pvz and gym-pvz packages"
	@echo "  train     - Train all agents (DDQN, DQN, Actor-Critic, Policy)"
	@echo "  evaluate  - Evaluate all trained agents"
	@echo "  figures   - Generate all figures"
	@echo "  tables    - Generate all tables"
	@echo "  quick     - Quick demo with pre-trained agents"
	@echo "  clean     - Clean generated files"
	@echo "  help      - Show this help"

# Setup dependencies
setup:
	@echo "📦 Installing dependencies..."
	pip install torch torchvision pygame gym numpy matplotlib shap pandas tabulate

# Install packages
install:
	@echo "🔧 Installing pvz and gym-pvz packages..."
	cd pvz && pip install -e .
	cd gym-pvz && pip install -e .

# Create results directory
results:
	@mkdir -p results/figures results/tables results/models

# Train all agents
train: install results
	@echo "🚀 Training all agents..."
	@echo "Training DDQN agent..."
	echo "ddqn_reproduction" | python3 train_ddqn_agent.py
	@echo "Training DQN agent..."
	echo "dqn_reproduction" | python3 train_dqn_agent.py
	@echo "Training Actor-Critic agent..."
	python3 train_actor_critic_agent.py
	@echo "Training Policy agent..."
	python3 train_policy_agent.py
	@echo "✅ All agents trained!"

# Quick demo with pre-trained agents (for testing)
quick: install results
	@echo "🎮 Running quick demo with pre-trained agents..."
	python3 scripts/quick_evaluation.py

# Evaluate all agents
evaluate: results
	@echo "📊 Evaluating all agents..."
	python3 scripts/evaluate_all_agents.py

# Generate all figures
figures: results
	@echo "📈 Generating all figures..."
	python3 scripts/generate_figures.py

# Generate all tables
tables: results
	@echo "📋 Generating all tables..."
	python3 scripts/generate_tables.py

# Clean generated files
clean:
	@echo "🧹 Cleaning generated files..."
	rm -rf results/
	rm -f *.npy *_loss *_rewards.npy *_iterations.npy *_real_rewards.npy *_real_iterations.npy
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -delete
	@echo "✅ Cleaned!"

# Test installation
test: install
	@echo "🧪 Testing installation..."
	python3 -c "from pvz import config; print('✅ pvz package works')"
	python3 -c "import gym; import gym_pvz; env = gym.make('gym_pvz:pvz-env-v2'); print('✅ gym-pvz environment works')"
	python3 -c "from agents import DDQNAgent, QNetwork; print('✅ agents package works')"
	@echo "✅ All tests passed!"