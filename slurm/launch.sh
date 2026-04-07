#!/bin/bash
# ============================================================
# Unified launcher for NegotiationArena experiments.
#
# Reads a server profile, builds sbatch flags, and submits
# one SLURM job per (experiment, size) combination.
#
# Usage:
#   SERVER=mia bash slurm/launch.sh
#   SERVER=deucalion bash slurm/launch.sh
#
# Override any SLURM parameter at launch time:
#   SERVER=deucalion PARTITION=normal-a100-80 GPUS=4 bash slurm/launch.sh
#   SERVER=mia GPUS=1 MEM=32G TIME=8:00:00 bash slurm/launch.sh
#
# Select specific experiments/sizes:
#   SERVER=deucalion EXPERIMENTS="buysell_section_one" SIZES="very_small" bash slurm/launch.sh
#
# Dry run (print commands without submitting):
#   SERVER=deucalion DRY_RUN=1 bash slurm/launch.sh
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Validate and load server profile ────────────────────────
SERVER="${SERVER:?ERROR: set SERVER=mia|deucalion|... }"
PROFILE="${SCRIPT_DIR}/servers/${SERVER}.sh"

if [ ! -f "$PROFILE" ]; then
    echo "ERROR: Server profile not found: $PROFILE"
    echo "Available servers:"
    for f in "$SCRIPT_DIR"/servers/*.sh; do
        [ -f "$f" ] && echo "  - $(basename "$f" .sh)"
    done
    exit 1
fi

echo "Loading server profile: $SERVER"
source "$PROFILE"

# ── Validate required profile variables ──────────────────────
: "${SLURM_PARTITION:?Profile must set SLURM_PARTITION}"
: "${SLURM_TIME:?Profile must set SLURM_TIME}"
: "${CONDA_INIT_SCRIPT:?Profile must set CONDA_INIT_SCRIPT}"
: "${CONDA_ENV_PATH:?Profile must set CONDA_ENV_PATH}"
: "${HF_HOME_TEMPLATE:?Profile must set HF_HOME_TEMPLATE}"
: "${SLURM_GPU_DIRECTIVE:?Profile must set SLURM_GPU_DIRECTIVE}"

# Resolve HF_HOME (expand $(whoami) etc.)
HF_HOME=$(eval echo "$HF_HOME_TEMPLATE")

# ── Build sbatch flags from profile ─────────────────────────
SBATCH_FLAGS=(
    --partition="$SLURM_PARTITION"
    --time="$SLURM_TIME"
    --cpus-per-task="${SLURM_CPUS_PER_TASK:-8}"
    $SLURM_GPU_DIRECTIVE
)

[ -n "${SLURM_QOS:-}" ]     && SBATCH_FLAGS+=(--qos="$SLURM_QOS")
[ -n "${SLURM_ACCOUNT:-}" ] && SBATCH_FLAGS+=(--account="$SLURM_ACCOUNT")
[ -n "${SLURM_MEM:-}" ]     && SBATCH_FLAGS+=(--mem="$SLURM_MEM")

# ── Experiments and sizes ────────────────────────────────────
DEFAULT_SIZES=("very_small" "small")
DEFAULT_EXPERIMENTS=(
    # Section 1: Baselines
    "buysell_section_one"
    "trading_section_one"
    "ultimatum_section_one"

    # Section 1: Retry Ablations
    "buysell_section_one_retry3"
    "trading_section_one_retry3"
    "ultimatum_section_one_retry3"

    # Section 2: Personas / Social Behavior
    "buysell_section_two_personas"
    "trading_section_two_personas"
    "ultimatum_section_two_personas"
)

# Allow override via environment
# Use SIZES=none for experiments that define their own models in the YAML
if [ "${SIZES:-}" = "none" ]; then
    SIZE_ARRAY=("none")
elif [ -n "${SIZES:-}" ]; then
    IFS=' ' read -ra SIZE_ARRAY <<< "$SIZES"
else
    SIZE_ARRAY=("${DEFAULT_SIZES[@]}")
fi

if [ -n "${EXPERIMENTS:-}" ]; then
    IFS=' ' read -ra EXP_ARRAY <<< "$EXPERIMENTS"
else
    EXP_ARRAY=("${DEFAULT_EXPERIMENTS[@]}")
fi

# ── Build the export string for env vars needed inside the job
ENV_VARS="CONDA_INIT_SCRIPT=${CONDA_INIT_SCRIPT}"
ENV_VARS+=",CONDA_ENV_PATH=${CONDA_ENV_PATH}"
ENV_VARS+=",HF_HOME=${HF_HOME}"
ENV_VARS+=",SERVER=${SERVER}"
[ -n "${MODULE_LOADS:-}" ]    && ENV_VARS+=",MODULE_LOADS=${MODULE_LOADS}"
[ -n "${EXTRA_ENV_VARS:-}" ]  && ENV_VARS+=",EXTRA_ENV_VARS=${EXTRA_ENV_VARS}"

# ── Summary ──────────────────────────────────────────────────
TOTAL=$((${#EXP_ARRAY[@]} * ${#SIZE_ARRAY[@]}))
echo "============================================"
echo "Server     : $SERVER"
echo "Partition  : $SLURM_PARTITION"
echo "GPU        : $SLURM_GPU_DIRECTIVE"
echo "CPUs/task  : ${SLURM_CPUS_PER_TASK:-8}"
echo "Memory     : ${SLURM_MEM:-<default>}"
echo "Time limit : $SLURM_TIME"
echo "QoS        : ${SLURM_QOS:-<none>}"
echo "Account    : ${SLURM_ACCOUNT:-<none>}"
echo "Sizes      : ${SIZE_ARRAY[*]}"
echo "Experiments: ${#EXP_ARRAY[@]}"
echo "Total jobs : $TOTAL"
echo "Dry run    : ${DRY_RUN:-no}"
echo "============================================"

# ── Submit jobs ──────────────────────────────────────────────
cd "$REPO_DIR"
mkdir -p logs/slurm

for SIZE in "${SIZE_ARRAY[@]}"; do
    for EXP in "${EXP_ARRAY[@]}"; do
        if [ "$SIZE" = "none" ]; then
            JOB_NAME="${EXP}"
            EXPORT_STR="ALL,${ENV_VARS},EXPERIMENT=${EXP}"
        else
            JOB_NAME="${EXP}_${SIZE}"
            EXPORT_STR="ALL,${ENV_VARS},EXPERIMENT=${EXP},SIZE=${SIZE}"
        fi

        FULL_CMD=(sbatch
            --job-name="$JOB_NAME"
            "${SBATCH_FLAGS[@]}"
            --export="$EXPORT_STR"
            slurm/run.sh
        )

        if [ "${DRY_RUN:-0}" = "1" ]; then
            echo "[DRY RUN] ${FULL_CMD[*]}"
        else
            echo "Submitting: $JOB_NAME"
            "${FULL_CMD[@]}"
            sleep 0.5
        fi
    done
done

echo "────────────────────────────────────────────"
echo "Done. Check queue: squeue -u \$(whoami)"
