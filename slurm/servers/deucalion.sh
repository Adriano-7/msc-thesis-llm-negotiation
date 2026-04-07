# ============================================================
# Deucalion HPC Server Profile — Portuguese EuroHPC supercomputer
#
# Hardware: 4× NVIDIA A100 (40 GB or 80 GB) per GPU node
# Partitions: dev-a100-{40,80} (4 h), normal-a100-{40,80} (48 h)
# Billing: per-GPU; 1 GPU = 32 CPUs
# Internet: NOT available on GPU partitions — models must be pre-cached
#
# All variables can be overridden at launch time:
#   PARTITION=normal-a100-80 GPUS=4 SERVER=deucalion bash slurm/launch.sh
# ============================================================

# ── SETUP: Fill in your project details once ─────────────────
PROJECT_DIR="/projects/F202500007HPCVLABUPORTO/amachado.up"
GPU_ACCOUNT="f202500007hpcvlabuportog"
# ──────────────────────────────────────────────────────────────

# ── SLURM resource directives ───────────────────────────────
# Partition options: dev-a100-40, normal-a100-40, dev-a100-80, normal-a100-80
SLURM_PARTITION="${PARTITION:-normal-a100-40}"
SLURM_QOS=""                              # Deucalion has no QoS system
SLURM_ACCOUNT="${ACCOUNT:-$GPU_ACCOUNT}"
SLURM_TIME="${TIME:-48:00:00}"
SLURM_CPUS_PER_TASK="${CPUS:-32}"         # 32 CPUs per GPU (billing rule)
SLURM_MEM="${MEM:-}"                      # not needed on Deucalion GPU nodes
SLURM_GPU_DIRECTIVE="--gpus=${GPUS:-2}"   # Deucalion uses --gpus syntax

# ── Environment setup ────────────────────────────────────────
CONDA_INIT_SCRIPT="source ${PROJECT_DIR}/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV_PATH="${PROJECT_DIR}/envs/negotiation"
HF_HOME_TEMPLATE="${PROJECT_DIR}/huggingface"

# Module loads (space-separated)
MODULE_LOADS="Python/3.12.3-GCCcore-13.3.0"

# Extra environment variables (space-separated KEY=VALUE pairs)
# Offline mode required: GPU partitions have no internet access
EXTRA_ENV_VARS="TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1"
