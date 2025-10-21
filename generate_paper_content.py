#!/usr/bin/env python3
"""
Generate paper figures and tables using available training data
Creates visualizations and analysis based on experimental results where possible,
or provides clear documentation of what would be needed for reproduction.
"""

import os
import sys
import subprocess

def run_script(script_name, description):
    """Run a script and report results"""
    try:
        print(f"\n🔧 {description}...")
        result = subprocess.run([sys.executable, script_name],
                              capture_output=True, text=True, check=True)
        print(f"✅ {description} completed")
        if result.stdout:
            print(f"📊 Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error in {script_name}: {e}")
        if e.stdout:
            print(f"stdout: {e.stdout}")
        if e.stderr:
            print(f"stderr: {e.stderr}")
        return False
    except FileNotFoundError:
        print(f"❌ Script not found: {script_name}")
        return False

def main():
    """Generate all paper content using available data"""
    print("🎯 Generating paper reproduction content")
    print("="*60)

    # Ensure output directories exist
    os.makedirs('results/figures', exist_ok=True)
    os.makedirs('results/tables', exist_ok=True)

    # Track results
    completed = []
    failed = []

    # Generate figures and tables
    scripts_to_run = [
        ("generate_figure_1.py", "Figure 1: Training performance comparison"),
        ("generate_figure_4_status.py", "Figure 4: Action frequency analysis requirements"),
        ("generate_figure_7.py", "Figure 7: Convergence analysis"),
        ("generate_figure_8_status.py", "Figure 8: Feature importance analysis requirements"),
        ("generate_figure_9_status.py", "Figure 9: Policy gradient performance analysis"),
        ("generate_table_ii_status.py", "Table II: Statistical significance testing requirements"),
        ("generate_table_3_status.py", "Table 3: Hyperparameter sensitivity analysis requirements")
    ]

    # Run each script
    for script, description in scripts_to_run:
        if run_script(script, description):
            completed.append(description)
        else:
            failed.append(description)

    # Available figures that use real data
    real_data_figures = [
        "Figure 2: DQN/DDQN learning curves",
        "Figure 3: Score distribution",
        "Figure 5: Survival distribution",
        "Figure 6: Action distribution"
    ]

    # Summary report
    print("\n" + "="*60)
    print("📋 PAPER CONTENT GENERATION SUMMARY")
    print("="*60)

    print(f"\n✅ GENERATED ({len(completed)} items):")
    for item in completed:
        print(f"   • {item}")

    print(f"\n📊 AVAILABLE WITH REAL DATA ({len(real_data_figures)} items):")
    for item in real_data_figures:
        print(f"   • {item}")

    if failed:
        print(f"\n❌ FAILED ({len(failed)} items):")
        for item in failed:
            print(f"   • {item}")

    print(f"\n📈 REPRODUCTION STATUS:")
    print(f"   • Real experimental data available: {len(real_data_figures)}")
    print(f"   • Implementation requirements documented: {len(completed)}")
    print(f"   • Generation failures: {len(failed)}")

    print(f"\n🎉 Paper content generation complete!")
    print(f"📂 Results saved in results/ directory")

if __name__ == "__main__":
    main()