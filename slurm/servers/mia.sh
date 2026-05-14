# ============================================================
# MIA HPC Server Profile — University of Porto MSc AI cluster
#
# Hardware: 2-3× NVIDIA L40S (48 GB VRAM) per node
# Partitions: teach, normal, fast
# QoS: gpu_batch (batch, ≤1 GPU per job & per user, ≤12 h), gpu (interactive, ≤2 GPUs, ≤3 h, faculty/teach only)
#
# All variables can be overridden at launch time:
#   GPUS=1 MEM=32G SERVER=mia bash slurm/launch.sh
# ============================================================

# ── SLURM resource directives ───────────────────────────────
SLURM_PARTITION="${PARTITION:-normal}"
SLURM_QOS="${QOS:-gpu_batch}"
SLURM_ACCOUNT="${ACCOUNT:-}"              # MIA does not require accounts
SLURM_TIME="${TIME:-12:00:00}"
SLURM_CPUS_PER_TASK="${CPUS:-2}"
SLURM_MEM="${MEM:-8G}"
SLURM_GPU_DIRECTIVE="--gres=gpu:${GPUS:-1}"   # MIA uses --gres syntax; gpu_batch QoS caps at 1 GPU
SLURM_NODELIST="${NODELIST:-}"

# ── Environment setup ────────────────────────────────────────
CONDA_INIT_SCRIPT='source ~/miniconda3/etc/profile.d/conda.sh'
CONDA_ENV_PATH="/data/01/up202105352/envs/negotiation"
HF_HOME_TEMPLATE='/data/01/$(whoami)/huggingface'

# Module loads (space-separated, empty if none)
MODULE_LOADS=""

# Extra environment variables (space-separated KEY=VALUE pairs)
EXTRA_ENV_VARS=""