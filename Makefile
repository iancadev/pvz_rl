# Plants vs Zombies RL - Complete Reproduction Pipeline
# Run 'make' to reproduce all figures and tables from the paper

.PHONY: all full setup install train-quick train-full evaluate figures tables paper-figures paper-tables clean help quick test play play-manual

# Python path setup
export PYTHONPATH := $(shell pwd)/pvz:$(shell pwd)/gym-pvz:$(PYTHONPATH)

# Set environment variables for all targets
.EXPORT_ALL_VARIABLES:

# Default target - quick demo with short training (200 episodes)
all: setup install train-quick
	@echo "🎉 Quick demo completed! Use 'make full' for complete reproduction."

# Full reproduction with long training (100K episodes)
full: setup install train-full evaluate figures tables
	@echo "🎉 Full reproduction completed! Check the results/ directory for figures and tables."

# Help target
help:
	@echo "Plants vs Zombies RL - Reproduction Pipeline"
	@echo ""
	@echo "Targets:"
	@echo "  all           - Quick demo with 200 episodes (default)"
	@echo "  full          - Full reproduction with 100K episodes"
	@echo "  setup         - Install dependencies and setup environment"
	@echo "  install       - Install pvz and gym-pvz packages"
	@echo "  train-quick   - Train all agents with 200 episodes"
	@echo "  train-full    - Train all agents with 100K episodes"
	@echo "  evaluate      - Evaluate all trained agents"
	@echo "  figures       - Generate all figures and tables"
	@echo "  tables        - Generate all figures and tables"
	@echo "  paper-figures - Generate all paper figures and tables"
	@echo "  paper-tables  - Generate all paper figures and tables"
	@echo "  quick         - Quick demo with pre-trained agents"
	@echo "  play          - Watch visual gameplay with pre-trained agent"
	@echo "  play-manual   - Play the game manually with keyboard controls"
	@echo "  clean         - Clean generated files"
	@echo "  help          - Show this help"

# Setup dependencies
setup:
	@echo "📦 Installing dependencies..."
	pip install torch torchvision pygame gym numpy matplotlib shap pandas tabulate

# Install packages
install:
	@echo "🔧 Installing pvz and gym-pvz packages..."
	cd pvz-pkg && pip install -e . --use-pep517
	cd gym-pvz && pip install -e . --use-pep517

# Create results directory
results:
	@mkdir -p results/figures results/tables results/models

# Quick training with 200 episodes (for demos)
train-quick: PVZ_EPISODES=200
train-quick: install results
	@echo "🚀 Training all agents (200 episodes)..."
	@echo "Training DDQN agent..."
	echo ddqn_quick| python train_ddqn_agent.py
	@echo "Training DQN agent..."
	echo dqn_quick| python train_dqn_agent.py
	@echo "Training Actor-Critic agent..."
	echo ac_quick| python train_actor_critic_agent.py
	@echo "Training Policy agent..."
	echo gp_quick| python train_policy_agent.py
	@echo "✅ All agents trained (quick mode)!"

# Full training with 100K episodes (for complete reproduction)
train-full: PVZ_EPISODES=100000
train-full: install results
	@echo "🚀 Training all agents (100K episodes)..."
# 	$$env:PVZ_EPISODES=100000
# 	@echo "Training DDQN agent..."
# 	"ddqn_reproduction" | python train_ddqn_agent.py
# 	@echo "Training DQN agent..."
# 	"dqn_reproduction" | python train_dqn_agent.py
# 	@echo "Training Actor-Critic agent..."
# 	"ac_reproduction" | python train_actor_critic_agent.py
# 	@echo "Training Policy agent..."
# 	"gp_reproduction" | python train_policy_agent.py
# 	@echo "✅ All agents trained (full mode)!"
	@echo "Training DDQN agent..."
	echo ddqn_reprod|python train_ddqn-lg.py
	@echo "Training DQN agent..."
	echo dqn_reprod|python train_dqn-lg.py
	@echo "Training Actor-Critic agent..."
	echo ac_reprod|python train_ac-lg.py
	@echo "Training Policy agent..."
	echo gp_reprod|python train_gp-lg.py
	@echo "✅ All agents trained (full mode)!"

# Quick demo with pre-trained agents (for testing)
quick: install results
	@echo "🎮 Running quick demo with pre-trained agents..."
	python scripts/quick_evaluation.py

# Evaluate all agents
evaluate: results
	@echo "📊 Evaluating all agents..."
	python scripts/evaluate_all_agents.py

# Generate all figures
figures: results
	@echo "📈 Generating all figures..."
	python generate_all_paper_content.py

# Generate all tables
tables: results
	@echo "📋 Generating all tables..."
	python generate_all_paper_content.py

# Generate paper-specific figures only
paper-figures: results
	@echo "📈 Generating paper-specific figures..."
	python generate_all_paper_content.py

# Generate paper-specific tables only
paper-tables: results
	@echo "📋 Generating paper-specific tables..."
	python generate_all_paper_content.py

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
	python -c "from pvz import config; print('✅ pvz package works')"
	python -c "import gym; import gym_pvz; env = gym.make('gym_pvz:pvz-env-v2'); print('✅ gym-pvz environment works')"
	python -c "from agents import DDQNAgent, QNetwork; print('✅ agents package works')"
	@echo "✅ All tests passed!"

# Play visual game
play:
	@echo "🎮 Starting visual gameplay with pre-trained DDQN agent..."
	@echo "Close the pygame window to stop the game."
	python game_render.py

# Play manual game
play-manual:
	@echo "🎮 Starting manual gameplay - YOU control the plants!"
	@echo "Follow the on-screen prompts to place plants and defend against zombies."
	@echo "Close the pygame window to stop the game."
	python game_render_manual.py


delayed-train-dqn:
	timeout /t 10800
	echo ddqn_reprod|python train_ddqn-lg.py
	echo dqn_reprod|python train_dqn-lg.py
