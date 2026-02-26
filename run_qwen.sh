#!/bin/bash
#SBATCH --job-name=qwen-negotiation
#SBATCH --output=qwen_buysell_%j.log
#SBATCH --error=qwen_buysell_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH -p normal
#SBATCH --qos=gpu_batch
#SBATCH --gres=gpu:1

# Load env variables
set -a
source .env
set +a

export HF_HOME="/data/01/up202105352/huggingface"

python3 runner/buysell_qwen.py
