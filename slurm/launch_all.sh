#!/bin/bash
# ============================================================
# Launch one SLURM job per size group for given experiments.
#
# Usage:
#   bash slurm/launch_all_sizes.sh
# ============================================================

set -euo pipefail

# 1. Define the sizes you want to run
SIZES=("very_small" "small")

# 2. Define the experiments from your YAML
# (Comment out any you don't want to run right now)
EXPERIMENTS=(
    # Section 1: Baselines
    "buysell_section_one"
    "trading_section_one"
    "ultimatum_section_one"

    # Section 1: Retry Ablations
    "buysell_section_one_retry3"
    "trading_section_one_retry3"
    "ultimatum_section_one_retry3"

    # Section 2: Personas / Social Behavior
    "buysell_section_two_personas_medium"
    "trading_section_two_personas_medium"
    "ultimatum_section_two_personas_medium"
)

echo "Preparing to launch ${#EXPERIMENTS[@]} experiments across ${#SIZES[@]} sizes..."
echo "Total jobs to submit: $((${#EXPERIMENTS[@]} * ${#SIZES[@]}))"
echo "--------------------------------------------------------"

for SIZE in "${SIZES[@]}"; do
    for EXP in "${EXPERIMENTS[@]}"; do
        # Create a clean job name, e.g., "buysell_section_one_medium"
        JOB_NAME="${EXP}_${SIZE}"
        
        echo "Submitting: $JOB_NAME"
        sbatch --job-name="$JOB_NAME" --export="EXPERIMENT=$EXP,SIZE=$SIZE" slurm/run.sh
        
        # Optional: slight delay to avoid hammering the SLURM scheduler
        sleep 0.5
    done
done

echo "--------------------------------------------------------"
echo "All jobs submitted! Check your queue with: squeue -u \$(whoami)"