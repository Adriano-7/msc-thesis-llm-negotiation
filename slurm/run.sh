#!/bin/bash
#SBATCH --job-name=negotiation
#SBATCH --output=logs/slurm/%x_%j.log
#SBATCH --error=logs/slurm/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -p normal
#SBATCH --qos=gpu_batch
#SBATCH --gres=gpu:2

# ============================================================
# Parameterised SLURM launcher for NegotiationArena
#
# Usage:
#   sbatch --export=EXPERIMENT=buysell_section_one,MODEL=Qwen/Qwen2.5-7B-Instruct slurm/run.sh
#   sbatch --export=EXPERIMENT=buysell_section_one slurm/run.sh          # runs all models in config
#   sbatch --export=EXPERIMENT=trading_section_one,NUM_RUNS=5 slurm/run.sh  # quick test
# ============================================================

set -euo pipefail

# ── defaults ──────────────────────────────────────────────────
EXPERIMENT="${EXPERIMENT:?ERROR: set EXPERIMENT via --export}"
CONFIG="${CONFIG:-configs/experiments.yaml}"
MODEL="${MODEL:-}"          # empty = run all models in config
NUM_RUNS="${NUM_RUNS:-}"    # empty = use config default

# ── environment ───────────────────────────────────────────────
set -a
source .env 2>/dev/null || true
set +a

export HF_HOME="${HF_HOME:-/data/01/$(whoami)/huggingface}"
mkdir -p logs/slurm

# ── build command ─────────────────────────────────────────────
CMD="python runner/run_experiment.py --config ${CONFIG} --experiment ${EXPERIMENT}"
[ -n "$MODEL" ]    && CMD="$CMD --model \"$MODEL\""
[ -n "$NUM_RUNS" ] && CMD="$CMD --num_runs $NUM_RUNS"

echo "============================================"
echo "EXPERIMENT : $EXPERIMENT"
echo "MODEL      : ${MODEL:-<all from config>}"
echo "NUM_RUNS   : ${NUM_RUNS:-<from config>}"
echo "CMD        : $CMD"
echo "============================================"

eval $CMD