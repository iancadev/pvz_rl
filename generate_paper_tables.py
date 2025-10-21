#!/usr/bin/env python3
"""
Generate all tables specifically mentioned in the paper
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from tabulate import tabulate

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

def generate_table_i_performance():
    """Generate Table I: Results on the slow-paced benchmark"""
    print("📋 Generating Table I: Performance comparison...")

    # Data from the paper's Table I
    performance_data = {
        'Agent': ['REINFORCE', 'REINFORCE', 'Actor critic', 'DQN', 'DDQN'],
        'Masked': ['NO', 'YES', 'NO', 'YES', 'YES'],
        'Mean score': [324.4, 538.12, 678.68, 1651.36, 1892.04],
        'Mean iterations': [145.549, 165.217, 165.894, 306.288, 338.413]
    }

    df = pd.DataFrame(performance_data)

    # Save as CSV
    df.to_csv('results/tables/table_i_performance.csv', index=False)

    # Save as markdown
    with open('results/tables/table_i_performance.md', 'w') as f:
        f.write("# Table I: Results on the slow-paced benchmark\n\n")
        f.write(tabulate(df, headers='keys', tablefmt='pipe', showindex=False))
        f.write("\n\n")
        f.write("**Note**: Masked refers to whether illegal actions were masked during training.\n")

    # Save as LaTeX
    with open('results/tables/table_i_performance.tex', 'w') as f:
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\caption{Results on the slow-paced benchmark}\n")
        f.write("\\label{tab:performance_benchmark}\n")
        f.write("\\begin{tabular}{llcc}\n")
        f.write("\\toprule\n")
        f.write("Agent & Masked & Mean score & Mean iterations \\\\\n")
        f.write("\\midrule\n")
        for _, row in df.iterrows():
            f.write(f"{row['Agent']} & {row['Masked']} & {row['Mean score']:.2f} & {row['Mean iterations']:.2f} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    print("✅ Table I generated")

def generate_table_ii_plants():
    """Generate Table II: Plant types and characteristics"""
    print("📋 Generating Table II: Plant characteristics...")

    # Data from the paper's Table II
    plant_data = {
        'Plant': ['Sunflower', 'Peashooter', 'Potatomine', 'Wall-nut'],
        'Cost': [50, 100, 25, 50],
        'HP': [300, 300, 300, 4000],
        'Attack': [0, 20, '∞ (once)', 0],
        'Sun prod.': [25, 0, 0, 0],
        'Cooldown': [5, 5, 20, 20]
    }

    df = pd.DataFrame(plant_data)

    # Save as CSV
    df.to_csv('results/tables/table_ii_plants.csv', index=False)

    # Save as markdown
    with open('results/tables/table_ii_plants.md', 'w') as f:
        f.write("# Table II: Plant types and characteristics\n\n")
        f.write(tabulate(df, headers='keys', tablefmt='pipe', showindex=False))
        f.write("\n\n")
        f.write("**Columns**:\n")
        f.write("- **Cost**: Sun cost required to plant\n")
        f.write("- **HP**: Hit points (health)\n")
        f.write("- **Attack**: Damage per attack\n")
        f.write("- **Sun prod.**: Sun production per time unit\n")
        f.write("- **Cooldown**: Time between plantings\n")

    # Save as LaTeX
    with open('results/tables/table_ii_plants.tex', 'w') as f:
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\caption{Plant types and characteristics}\n")
        f.write("\\label{tab:plant_characteristics}\n")
        f.write("\\begin{tabular}{lccccc}\n")
        f.write("\\toprule\n")
        f.write("Plant & Cost & HP & Attack & Sun prod. & Cooldown \\\\\n")
        f.write("\\midrule\n")
        for _, row in df.iterrows():
            f.write(f"{row['Plant']} & {row['Cost']} & {row['HP']} & {row['Attack']} & {row['Sun prod.']} & {row['Cooldown']} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    print("✅ Table II generated")

def generate_table_iii_regression():
    """Generate Table III: DDQN with linear regression"""
    print("📋 Generating Table III: DDQN regression results...")

    # Data from the paper's Table III
    regression_data = {
        'Configuration': ['DDQN', 'DDQN with regression', 'DDQN with regression'],
        'Plays': ['1K', '1K', '1.5K'],
        'Mean score': [1892.04, 1057.8, 1659.08],
        'Mean iterations': [338.413, 234.401, 311.011]
    }

    df = pd.DataFrame(regression_data)

    # Save as CSV
    df.to_csv('results/tables/table_iii_regression.csv', index=False)

    # Save as markdown
    with open('results/tables/table_iii_regression.md', 'w') as f:
        f.write("# Table III: DDQN with linear regression\n\n")
        f.write(tabulate(df, headers='keys', tablefmt='pipe', showindex=False))
        f.write("\n\n")
        f.write("**Note**: Comparison of DDQN performance with and without linear regression preprocessing.\n")

    # Save as LaTeX
    with open('results/tables/table_iii_regression.tex', 'w') as f:
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\caption{DDQN with linear regression}\n")
        f.write("\\label{tab:ddqn_regression}\n")
        f.write("\\begin{tabular}{lccc}\n")
        f.write("\\toprule\n")
        f.write("Configuration & Plays & Mean score & Mean iterations \\\\\n")
        f.write("\\midrule\n")
        for _, row in df.iterrows():
            f.write(f"{row['Configuration']} & {row['Plays']} & {row['Mean score']:.2f} & {row['Mean iterations']:.2f} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    print("✅ Table III generated")

def generate_environment_specifications():
    """Generate environment specification table"""
    print("📋 Generating environment specifications table...")

    from pvz import config

    env_specs = {
        'Parameter': [
            'Grid Size',
            'Action Space',
            'Observation Space',
            'Plant Types',
            'Zombie Types',
            'Episode Length',
            'Reward Function',
            'Frame Rate',
            'Max Frames'
        ],
        'Value': [
            f"{config.N_LANES} × {config.LANE_LENGTH}",
            "181 discrete actions",
            "95-dimensional vector",
            "4 (Peashooter, Sunflower, Wallnut, Potatomine)",
            "4 (Basic, Cone, Bucket, Flag)",
            "Variable (until win/loss)",
            "Score-based with survival bonus",
            f"{config.FPS} FPS",
            f"{config.MAX_FRAMES} frames"
        ]
    }

    df = pd.DataFrame(env_specs)

    # Save as CSV
    df.to_csv('results/tables/environment_specifications.csv', index=False)

    # Save as markdown
    with open('results/tables/environment_specifications.md', 'w') as f:
        f.write("# Environment Specifications\n\n")
        f.write(tabulate(df, headers='keys', tablefmt='pipe', showindex=False))

    # Save as LaTeX
    with open('results/tables/environment_specifications.tex', 'w') as f:
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\caption{Environment Specifications}\n")
        f.write("\\label{tab:environment_specs}\n")
        f.write("\\begin{tabular}{lc}\n")
        f.write("\\toprule\n")
        f.write("Parameter & Value \\\\\n")
        f.write("\\midrule\n")
        for _, row in df.iterrows():
            f.write(f"{row['Parameter']} & {row['Value']} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    print("✅ Environment specifications table generated")

def generate_hyperparameters_table():
    """Generate hyperparameters table"""
    print("📋 Generating hyperparameters table...")

    hyperparams = {
        'Parameter': [
            'Learning Rate',
            'Batch Size',
            'Memory Size',
            'Target Update Frequency',
            'Epsilon Start',
            'Epsilon End',
            'Epsilon Decay',
            'Hidden Layer Size',
            'Gamma (Discount Factor)'
        ],
        'DDQN': [
            '1e-4',
            '200',
            '100,000',
            '2,000 steps',
            '1.0',
            '0.01',
            'Exponential',
            '512',
            '0.99'
        ],
        'DQN': [
            '1e-4',
            '200',
            '100,000',
            'N/A',
            '1.0',
            '0.01',
            'Exponential',
            '512',
            '0.99'
        ],
        'Actor-Critic': [
            '1e-3',
            'Episode',
            'N/A',
            'N/A',
            'N/A',
            'N/A',
            'N/A',
            '80',
            '0.99'
        ],
        'Policy Gradient': [
            '1e-3',
            'Episode',
            'N/A',
            'N/A',
            'N/A',
            'N/A',
            'N/A',
            '80',
            '0.99'
        ]
    }

    df = pd.DataFrame(hyperparams)

    # Save as CSV
    df.to_csv('results/tables/hyperparameters.csv', index=False)

    # Save as markdown
    with open('results/tables/hyperparameters.md', 'w') as f:
        f.write("# Hyperparameters Configuration\n\n")
        f.write(tabulate(df, headers='keys', tablefmt='pipe', showindex=False))

    print("✅ Hyperparameters table generated")

def main():
    """Generate all paper tables"""
    os.makedirs('results/tables', exist_ok=True)

    print("📊 Generating all paper-specific tables...")

    generate_table_i_performance()
    generate_table_ii_plants()
    generate_table_iii_regression()
    generate_environment_specifications()
    generate_hyperparameters_table()

    print("\n📋 All paper tables generated! Check results/tables/")
    print("Generated tables:")
    print("  - Table I: Performance comparison on slow-paced benchmark")
    print("  - Table II: Plant types and characteristics")
    print("  - Table III: DDQN with linear regression results")
    print("  - Environment specifications")
    print("  - Hyperparameters configuration")
    print("\nEach table is available in:")
    print("  - CSV format (.csv)")
    print("  - Markdown format (.md)")
    print("  - LaTeX format (.tex)")

if __name__ == "__main__":
    main()