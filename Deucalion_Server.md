# Deucalion HPC Cluster — User Reference Guide

> **Purpose:** This document contains everything a user needs to understand Deucalion's architecture, access model, storage layout, module system, and — most importantly — how to correctly write and submit Slurm jobs.  
> **Source:** Deucalion official documentation (docs.macc.fccn.pt) + HPCvLAB@UPORTO tutorial.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Access & Connection](#2-access--connection)
3. [Storage & File System](#3-storage--file-system)
4. [Partitions & Accounts](#4-partitions--accounts)
5. [Module System](#5-module-system)
6. [Submitting Jobs with Slurm](#6-submitting-jobs-with-slurm)
7. [GPU Jobs](#7-gpu-jobs)
8. [Interactive Sessions](#8-interactive-sessions)
9. [Containers (Singularity / Enroot)](#9-containers-singularity--enroot)
10. [Python Environments](#10-python-environments)
11. [Billing & Quotas](#11-billing--quotas)
12. [Compilation](#12-compilation)
13. [Useful Commands Cheatsheet](#13-useful-commands-cheatsheet)

---

## 1. System Overview

Deucalion is a Portuguese EuroHPC supercomputer operated by MACC/FCCN, running **Rocky Linux 8**. It is the first European HPC cluster with both ARM and x86 CPUs in the same infrastructure.

- **Job scheduler:** Slurm 23.11.4
- **Container runtime:** Singularity 3.11 (available on all partitions), Enroot + Pyxis (GPU nodes only)
- **Module system:** Lmod 8.5
- **Web portal:** https://login.deucalion.macc.fccn.pt (Open OnDemand)
- **Documentation:** https://docs.macc.fccn.pt

### Compute Nodes

| Partition | Nodes | CPU | GPU | RAM | Interconnect |
|-----------|-------|-----|-----|-----|-------------|
| `arm` | 1632 | Fujitsu A64FX 48-core 2.0 GHz | — | 32 GB | ConnectX-6 100 Gb/s |
| `x86` | 500 | 2× AMD EPYC 7742 64-core 2.25 GHz | — | 256 GB | ConnectX-6 100 Gb/s |
| `a100-40` | 17 | 2× AMD EPYC 7742 64-core 2.25 GHz | 4× NVIDIA A100 40 GB | 512 GB | ConnectX-6 200 Gb/s |
| `a100-80` | 16 | 2× AMD EPYC 7742 64-core 2.25 GHz | 4× NVIDIA A100 80 GB | 512 GB | ConnectX-6 200 Gb/s |

### Key Properties

- **CPU nodes are exclusive:** requesting 1 node allocates *all* cores (48 for ARM, 128 for x86). You are billed for the full node.
- **GPU nodes are non-exclusive:** you can request 1, 2, 3, or 4 GPUs per node. Each GPU requires 32 CPUs.
- **Login nodes:** `ln01`–`ln04`, all reachable via `login.deucalion.macc.fccn.pt`. Login nodes have internet access.
- **Dev partitions** also have internet access; all other compute partitions do not.

---

## 2. Access & Connection

### SSH Connection

```bash
# Linux/macOS
ssh -i <path-to-private-key> <username>@login.deucalion.macc.fccn.pt

# Windows PowerShell
ssh -i <path-to-private-key> \
  -o Ciphers=aes256-gcm@openssh.com \
  -o MACs=hmac-sha2-512-etm@openssh.com \
  <username>@login.deucalion.macc.fccn.pt
```

**SSH Key Requirements:** Only `ed25519` keys are accepted. Generate with:

```bash
ssh-keygen -t ed25519
```

Register the public key in the Deucalion User Portal under **Settings → Security**.

> **Note:** Up to 30–60 minutes may pass between uploading the key and being able to log in.

### Recommended `~/.ssh/config` Entry

```
Host deucalion
    Hostname login.deucalion.macc.fccn.pt
    User <username>
    ForwardAgent no
    IdentityFile ~/.ssh/id_ed25519
    Ciphers aes256-gcm@openssh.com
    MACs hmac-sha2-512-etm@openssh.com
```

After this, connect with: `ssh deucalion`

### Host Key Fingerprint

```
ED25519  SHA256:AMzlfN8g69jQGN26hYJqtag20d2CYG2Xg16VXmJv5V0
```

Verify this fingerprint on first connection. If it doesn't match, abort.

---

## 3. Storage & File System

| Mount | Type | Quota | Purpose |
|-------|------|-------|---------|
| `/home/<username>` | NFS (SSD, 50TB total) | 25 GB, 25 000 files | Personal config & small scripts only |
| `/projects/<project>` | Lustre (10 PB HDD + 430 TB NVMe) | Per-project (check with `quotaprojects`) | **All job data, environments, code** |
| `/apps` | NFS | — | System-installed applications |
| `/share` | Lustre | — | Shared resources |

### Critical Storage Rules

- **Run all jobs from `/projects/`**, never from `/home/`.
- **Create all Python/conda environments under `/projects/`**, not `/home/`. The home directory has a strict file count limit (25 000 files) that conda environments will quickly exhaust.
- Compilation is faster in `/projects/` due to Lustre's higher throughput vs NFS home.
- Data is retained for **6 months after project end**, then deleted.

### Quota Commands

```bash
quotahome        # Home directory usage
quotaprojects    # Project directory usage
```

---

## 4. Partitions & Accounts

### Available Partitions

| Partition | Architecture | Max Nodes | Time Limit | Internet |
|-----------|-------------|-----------|------------|---------|
| `dev-arm` | aarch64 | 2 | 4 hours | Yes |
| `normal-arm` | aarch64 | 128 | 48 hours | No |
| `large-arm` | aarch64 | 512 | 72 hours | No |
| `dev-x86` | x86_64 | 2 | 4 hours | Yes |
| `normal-x86` | x86_64 | 64 | 48 hours | No |
| `large-x86` | x86_64 | 128 | 72 hours | No |
| `dev-a100-40` | x86_64 | 1 | 4 hours | No |
| `normal-a100-40` | x86_64 | 4 | 48 hours | No |
| `dev-a100-80` | x86_64 | 1 | 4 hours | No |
| `normal-a100-80` | x86_64 | 4 | 48 hours | No |

### Account Naming Convention

Accounts follow the pattern `<project_id><arch_suffix>`:

| Suffix | Architecture |
|--------|-------------|
| `a` | ARM |
| `x` | x86 |
| `g` | GPU (A100) |

**Example:** `F20240001a` → ARM account, `F20240001g` → GPU account.

Check your accounts:
```bash
billing
# or
sacctmgr show Association where User=<username> format=Cluster,Account%30,User
```

### Partition Selection Guidelines

- **Prefer ARM for CPU-only work:** more core-hours available, more cores total, more power-efficient.
- **Use x86 only if:** software has ARM incompatibilities, or you need >32 GB RAM per node.
- **Use GPU partitions only if:** your code is GPU-accelerated.
- **Use `dev-*` partitions for:** compilation, environment setup, debugging, interactive sessions.
- **`large-*` partitions** are for jobs exceeding `normal-*` node limits only.

---

## 5. Module System

Software is provided as loadable modules (Lmod). Modules must be loaded both on login nodes and inside job scripts.

```bash
module avail                          # List all available modules
module overview                       # Compact list
module spider <name>                  # Search for a module (e.g., "numpy", "OpenMPI")
module spider <name>/<version>        # Find which parent module provides it
module load <module>                  # Load a module
module load <module>/<version>        # Load specific version
module list                           # Show loaded modules
module purge                          # Unload all modules
module unload <module>                # Unload one module
module switch <old> <new>             # Switch version
module whatis <module>/<version>      # Show module description
module help <module>/<version>        # Show module help
```

### Finding a Python Package via Modules

```bash
module spider numpy          # Find which modules provide numpy
module spider numpy/1.26.4   # Get the parent module name
# Output will show: "SciPy-bundle/2024.05-gfbf-2024a"
module load SciPy-bundle/2024.05-gfbf-2024a
```

> **Important:** Module availability differs by architecture. Always check available modules while inside the target partition (ARM or x86), not just from the login node.

---

## 6. Submitting Jobs with Slurm

### Core Slurm Commands

| Command | Description |
|---------|-------------|
| `sbatch <script.sh>` | Submit a batch job |
| `squeue` | View all jobs in queue |
| `squeue --me` | View only your jobs |
| `squeue --me --start` | Show estimated start time of pending jobs |
| `scancel <jobid>` | Cancel a job |
| `sinfo` | Show partition and node status |
| `srun` | Run a command interactively on compute nodes |
| `salloc` | Allocate resources and get a shell |

### Batch Job Script Structure

A batch script has three sections: interpreter declaration, `#SBATCH` directives, and shell commands.

```bash
#!/bin/bash
#SBATCH --job-name=myJob
#SBATCH --account=<account>          # e.g., F20240001a (ARM), F20240001x (x86), F20240001g (GPU)
#SBATCH --partition=<partition>      # e.g., normal-arm, normal-x86, normal-a100-40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48           # for ARM full node; 128 for x86 full node
#SBATCH --time=02:00:00              # format: hh:mm:ss or d-hh:mm:ss
#SBATCH --output=results/%j.out      # %j = job ID
#SBATCH --error=results/%j.err

# Load modules
module load <SomeModule>/<version>

# Run your application
./myapplication --input data.txt --output results.txt
```

Submit with:
```bash
sbatch myjob.sh
# Output: Submitted batch job 123456
```

### Common `#SBATCH` Directives Reference

| Directive | Description |
|-----------|-------------|
| `--job-name=<name>` | Human-readable job name |
| `--account=<account>` | Slurm account to bill |
| `--partition=<partition>` | Target partition |
| `--nodes=<N>` | Number of nodes |
| `--ntasks=<N>` | Number of tasks/MPI ranks |
| `--ntasks-per-node=<N>` | Tasks per node |
| `--cpus-per-task=<N>` | CPU threads per task (for OpenMP) |
| `--mem=<size>` | Total memory per node (e.g., `2G`, `64G`) |
| `--mem-per-cpu=<size>` | Memory per CPU core |
| `--time=<hh:mm:ss>` | Wall-time limit |
| `--output=<path>` | stdout file (`%j` = job ID) |
| `--error=<path>` | stderr file |
| `--mail-type=<type>` | Email triggers (e.g., `END,FAIL`) |
| `--mail-user=<email>` | Email address for notifications |
| `--exclusive` | Request exclusive node access (relevant for GPU nodes) |
| `--dependency=<type:jobid>` | Job dependency |

### Job Dependencies

```bash
sbatch job1.sh
# Submitted batch job 123456

sbatch --dependency=afterany:123456 job2.sh
# Starts job2.sh only after job1.sh completes
```

| Dependency type | Description |
|----------------|-------------|
| `after:jobid` | Start after the specified job has started |
| `afterany:jobid` | Start after the specified job finishes (any status) |
| `afterok:jobid` | Start after the specified job completes successfully |
| `afternotok:jobid` | Start after the specified job has failed |

### Chained Job Submission Script

```bash
#!/bin/bash
submit_job() {
    sub="$(sbatch "$@")"
    if [[ "$sub" =~ Submitted\ batch\ job\ ([0-9]+) ]]; then
        echo "${BASH_REMATCH[1]}"
    else
        exit 1
    fi
}

id1=$(submit_job job1.sh)
id2=$(submit_job --dependency=afterany:$id1 job2.sh)
id3=$(submit_job --dependency=afterany:$id1 job3.sh)
id4=$(submit_job --dependency=afterany:$id2:$id3 job4.sh)
```

### Checking Job Output

By default, stdout and stderr are written to `slurm-<jobid>.out` in the working directory. You can customize this with `--output` and `--error`.

---

## 7. GPU Jobs

GPU nodes (`gnx[501-533]`) are **non-exclusive**. Each node has 4 GPUs. You can request 1–4 GPUs per node.

### GPU Billing Rule

**1 GPU = 32 CPUs.** Requesting more CPUs than `32 × num_gpus` causes billing for additional GPU-hours.

### Single GPU Job

```bash
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=4:00:00
#SBATCH --partition=normal-a100-40
#SBATCH --account=<account_ending_in_g>

module load OpenMPI/5.0.3-GCC-13.3.0 CUDA/11.8.0 NCCL/2.20.5-GCCcore-13.3.0-CUDA-12.4.0

srun -n1 ./my_gpu_code
```

### Multi-GPU Single Node (4 GPUs)

```bash
#!/bin/bash
#SBATCH -A <account_g>
#SBATCH -t 00:30:00
#SBATCH -p normal-a100-40
#SBATCH -N 1
#SBATCH --gpus=4
#SBATCH --tasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --output=results/%j.out

module load OpenMPI/5.0.3-GCC-13.3.0 CUDA/11.8.0 NCCL/2.20.5-GCCcore-13.3.0-CUDA-12.4.0

srun my_gpu_app
```

### Multi-Node Multi-GPU (16 GPUs across 4 nodes)

```bash
#!/bin/bash
#SBATCH -A <account_g>
#SBATCH -t 00:30:00
#SBATCH -p normal-a100-40
#SBATCH -N 4
#SBATCH --gpus=16
#SBATCH --tasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --output=results/%j.out

module load OpenMPI/5.0.3-GCC-13.3.0 CUDA/11.8.0 NCCL/2.20.5-GCCcore-13.3.0-CUDA-12.4.0

srun my_multi_node_gpu_app
```

> **Tip:** Use `nvidia-smi` inside a job to verify GPU allocation.

---

## 8. Interactive Sessions

### Using `srun` (simple one-liner)

```bash
srun --time=00:30:00 --partition=dev-arm --account=<account_a> --nodes=1 --pty bash
```

### Using `salloc` (allocate, then use `srun` inside)

```bash
salloc --nodes=2 --partition=normal-arm --account=<account_a> --time=00:30:00
# Once inside, run parallel tasks:
srun --ntasks=32 --cpus-per-task=8 ./mpi_application
# Exit the allocation:
exit
```

### Connecting to a Running Job

```bash
# Open a shell inside a running job
srun --interactive --pty --jobid=<jobid> $SHELL

# Check resource usage (top) inside a running job
srun --interactive --pty --jobid=<jobid> top

# Target a specific compute node
srun --interactive --pty --jobid=<jobid> -w cnXXXXX top
```

### Node Hostname Patterns

| Partition | Hostname prefix | Example |
|-----------|----------------|---------|
| ARM | `cna` | `cna0017` |
| x86 | `cnx` | `cnx0042` |
| GPU | `gnx` | `gnx0501` |
| Login | `ln` | `ln03` |

---

## 9. Containers (Singularity / Enroot)

### Singularity (available on all partitions)

```bash
# Pull from Docker Hub (do this on a dev partition with internet)
singularity pull tensorflow.sif docker://tensorflow/tensorflow:2.10.1-gpu

# Run a container
singularity run mycontainer.sif

# Execute a command inside
singularity exec mycontainer.sif python myscript.py

# Start as root inside container
singularity start --root mycontainer.sif

# Start instance with GPU support
singularity instance start --nv mycontainer.sif myinstance
singularity exec instance://myinstance python train.py
singularity instance stop -t 30 myinstance
```

### Singularity in a Batch Job

```bash
#!/bin/bash
#SBATCH -p normal-a100-40 -t 00:30:00

CONTAINER_URI='tensorflow.sif'

singularity instance start --nv $CONTAINER_URI tensorflow
singularity exec instance://tensorflow python ./train.py
singularity instance stop -t 30 tensorflow
```

### Enroot + Pyxis (GPU nodes only — `gnx[501-533]`)

```bash
# Import Docker image
enroot import docker://ubuntu

# Create container
enroot create ubuntu.sqsh

# Start container
enroot start ubuntu

# List containers
enroot list -f

# Remove container
enroot remove ubuntu
```

**Pyxis with `srun`:**

```bash
# Run inside a container on GPU node
srun --container-image=/share/apps-x86/containers/enroot/ubuntu-rolling.sqsh \
     --container-name=ubuntu \
     -p normal-a100-40 -A <account_g> -t 00:30:00 \
     cat /etc/os-release

# Mount host path inside container
srun --container-image=ubuntu-rolling.sqsh \
     --container-mounts=/path/on/host:/path/in/container \
     --container-name=ubuntu \
     -p normal-a100-40 -A <account_g> -t 00:30:00 \
     my_command
```

**Pyxis with `sbatch`:**

```bash
#!/bin/bash
#SBATCH -p normal-a100-40 -t 30:00
#SBATCH --container-mounts=/path/to/code
#SBATCH --container-workdir=/path/to/code
#SBATCH --container-image=/path/to/container.sqsh
#SBATCH --container-name=mycontainer

srun code input
```

---

## 10. Python Environments

### Critical Rules

- All environments **must** be created inside `/projects/`, not `/home/`.
- Create ARM environments on an ARM node (`dev-arm`); create x86/GPU environments on `dev-x86`.
- An environment built for ARM will **not** run on x86/GPU, and vice versa.

### virtualenv

```bash
# Start interactive session on target partition first
srun --account=<account> --partition=dev-arm --nodes=1 --time=00:30:00 --pty bash

# Load Python module
module load Python/3.12.3-GCCcore-13.3.0

# Create environment inside project folder
python3 -m venv /projects/<project>/venv_arm

# Activate and install
source /projects/<project>/venv_arm/bin/activate
pip install -r requirements.txt
pip install ipykernel   # if using Jupyter
```

### Conda

Modify `~/.condarc` to redirect conda to `/projects/`:

```yaml
envs_dirs:
  - /projects/<project>/<yourfolder>/.conda/conda_envs
pkgs_dirs:
  - /projects/<project>/<yourfolder>/.conda/conda_pkgs
```

### Using the Environment in a Job Script

```bash
#!/bin/bash
#SBATCH --partition=normal-arm
#SBATCH --account=<account_a>
#SBATCH --nodes=1
#SBATCH --time=02:00:00

# Option A: Load module
module load SciPy-bundle/2024.05-gfbf-2024a

# Option B: Activate virtualenv
source /projects/<project>/venv_arm/bin/activate

python my_script.py
```

### Jupyter Custom Kernel

```bash
# Inside the activated environment
pip install ipykernel
python -m ipykernel install --user --name=myenv_arm --display-name "myenv_arm"
```

---

## 11. Billing & Quotas

### How Billing Works

- **CPU billing:** `core-hours = num_nodes × cores_per_node × walltime_hours`
  - ARM node: 48 cores × hours used
  - x86 node: 128 cores × hours used
  - You are billed for the **full node** regardless of how many cores your job uses.
- **GPU billing:** `GPU-hours = num_gpus × walltime_hours`
  - GPU nodes are non-exclusive; you are only billed for the GPUs you request.
  - Requesting >32 CPUs per GPU implicitly bills for additional GPUs.
- Billing is based on **actual runtime**, not requested time. A 1-hour request that finishes in 5 minutes is billed for 5 minutes.

### Billing Commands

```bash
# Check all your accounts and usage
billing

# Check usage per user for a specific account
billing -a <account>

# Check energy consumption
get_energy -u $USER --all
get_energy -j <job_number>           # Energy for a specific job
get_energy -a <account>              # Energy for a project
```

### Sample `billing` Output

```
┌────────────┬──────────┬───────────┬──────────┐
│ Account    │ Used (h) │ Limit (h) │ Used (%) │
├────────────┼──────────┼───────────┼──────────┤
│ F20240001a │   113656 │    200000 │    56.83 │
│ F20240001g │        0 │      2000 │     0.04 │
│ F20240001x │      231 │     10000 │     2.31 │
└────────────┴──────────┴───────────┴──────────┘
```

---

## 12. Compilation

### ARM Compilation

ARM nodes must be used for native ARM compilation (login nodes are x86). Get an ARM node first:

```bash
salloc -N1 -p dev-arm -A <account_a> -t 4:00:00
ssh cna<XXXX>   # or check with: squeue --me
```

**Load compilers on ARM:**

```bash
ml FJSVstclanga   # Fujitsu compilers
ml OpenMPI        # GNU + MPI
```

| Type | Compiler | Native (on ARM node) | Cross (on login node) |
|------|----------|---------------------|----------------------|
| Fujitsu Fortran | | `frt` | `frtpx` |
| Fujitsu C | | `fcc` | `fccpx` |
| Fujitsu C++ | | `FCC` | `FCCpx` |
| GNU Fortran | | `gfortran` | — |
| GNU C | | `gcc` | — |
| MPI Fortran | GNU | `mpifort` | `mpifrtpx` (Fujitsu) |

Recommended GNU flags (ARM): `-O2 -ftree-vectorize -march=native -fno-math-errno`

### x86 Compilation (on login nodes)

```bash
ml GCCcore/11.3.0   # GNU compilers
ml intel            # Intel oneAPI compilers
```

Recommended GCC flags: `-O2 -ftree-vectorize -march=native -fno-math-errno`  
Recommended Intel flags: `-O2 -march=core-avx2 -ftz -fp-speculation=safe -fp-model precise`

### GPU Compilation

```bash
# CUDA (C/C++)
ml CUDA/11.8.0 GCC/11.3.0
nvcc --generate-code arch=compute_80,code=sm_80 -o sample sample.cu

# OpenACC / CUDA Fortran
ml NVHPC/22.9-CUDA-11.8.0
nvfortran -acc -gpu=cc80 -Minfo=accel -Mpreprocess -o sample sample_acc.f90
```

---

## 13. Useful Commands Cheatsheet

### Storage & Quotas

```bash
quotahome                     # Home directory usage/limits
quotaprojects                 # Project directories usage/limits
billing                       # Compute hour usage per account
billing -a <account>          # Per-user breakdown for an account
```

### Job Management

```bash
sbatch myjob.sh               # Submit job
squeue --me                   # View your jobs
squeue --me --start           # Estimated start time for pending jobs
scancel <jobid>               # Cancel a job
sinfo                         # Partition and node status
scontrol show job <jobid>     # Detailed job info
```

### Interactive Work

```bash
# Get a shell on ARM dev node
srun --account=<account_a> --partition=dev-arm --nodes=1 --time=00:30:00 --pty bash

# Get a shell on x86 dev node
srun --account=<account_x> --partition=dev-x86 --nodes=1 --time=00:30:00 --pty bash

# Get a shell on GPU dev node
srun --account=<account_g> --partition=dev-a100-40 --nodes=1 --gpus=1 --time=00:30:00 --pty bash
```

### Modules

```bash
module avail                        # List all modules
module spider <name>                # Search for module by name
module spider <name>/<version>      # Find parent module for specific version
module load <module>/<version>      # Load a module
module list                         # Show loaded modules
module purge                        # Unload all
```

### File Transfer (from your local machine)

```bash
# Upload to Deucalion
scp /local/path deucalion:/remote/path

# Download from Deucalion
scp deucalion:/remote/path /local/path

# Copy entire directory
scp -r /local/dir deucalion:/remote/dir
```

### Energy Monitoring

```bash
get_energy -u $USER --all           # Total energy for all your jobs
get_energy -j <jobid>               # Energy for a specific job
get_energy -a <account>             # Energy for a project account
get_energy -h                       # All options
```

---

## Quick Reference: Minimal Job Templates

### CPU ARM Job

```bash
#!/bin/bash
#SBATCH --job-name=myjob
#SBATCH --account=<project_id>a
#SBATCH --partition=normal-arm
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --output=%j.out

module load <SomeModule>
python myscript.py
```

### CPU x86 Job

```bash
#!/bin/bash
#SBATCH --job-name=myjob
#SBATCH --account=<project_id>x
#SBATCH --partition=normal-x86
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --output=%j.out

module load <SomeModule>
./myapp
```

### Single GPU Job

```bash
#!/bin/bash
#SBATCH --job-name=mygpujob
#SBATCH --account=<project_id>g
#SBATCH --partition=normal-a100-40
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --output=%j.out

module load OpenMPI/5.0.3-GCC-13.3.0 CUDA/11.8.0
srun -n1 python train.py
```

### Multi-GPU Job (4 GPUs, 1 node)

```bash
#!/bin/bash
#SBATCH --job-name=multigpu
#SBATCH --account=<project_id>g
#SBATCH --partition=normal-a100-40
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --tasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --time=08:00:00
#SBATCH --output=%j.out

module load OpenMPI/5.0.3-GCC-13.3.0 CUDA/11.8.0 NCCL/2.20.5-GCCcore-13.3.0-CUDA-12.4.0
srun python distributed_train.py
```

---

---

## 14. NegotiationArena Project Setup on Deucalion

This section documents the concrete setup of this project on Deucalion to have full context.

### Account & Project Details

| Field | Value |
|-------|-------|
| Username | `amachado.up` |
| SSH | `ssh deucalion` (via `~/.ssh/config`) |
| Project directory | `/projects/F202500007HPCVLABUPORTO` |
| User workspace | `/projects/F202500007HPCVLABUPORTO/amachado.up` |
| GPU account | `f202500007hpcvlabuportog` |
| x86 account | `f202500007hpcvlabuportox` |
| ARM account | `f202500007hpcvlabuportoa` |

### Directory Layout

```
/projects/F202500007HPCVLABUPORTO/amachado.up/
  MultiAgent-Negotiation/     # Git repo (cloned here)
  miniconda3/                  # Miniconda installation
  envs/
    negotiation/               # Conda environment (Python 3.12)
  huggingface/                 # HF_HOME — all model weights cached here
  .conda/
    conda_envs/                # Conda env cache (redirected from ~/.conda)
    conda_pkgs/                # Conda package cache (redirected from ~/.conda)
```

**Symlinks in `$HOME` (to avoid /home/ quota):**
```
~/.vscode-server  → /projects/F202500007HPCVLABUPORTO/amachado.up/.vscode-server
~/.cache          → /projects/F202500007HPCVLABUPORTO/amachado.up/.cache  (if created)
```

### Conda Configuration

`~/.condarc` redirects all conda data to `/projects/`:
```yaml
envs_dirs:
  - /projects/F202500007HPCVLABUPORTO/amachado.up/.conda/conda_envs
pkgs_dirs:
  - /projects/F202500007HPCVLABUPORTO/amachado.up/.conda/conda_pkgs
```

### Environment Activation (inside a job or interactive session)

```bash
source /projects/F202500007HPCVLABUPORTO/amachado.up/miniconda3/etc/profile.d/conda.sh
conda activate /projects/F202500007HPCVLABUPORTO/amachado.up/envs/negotiation
```

### Installed Python Packages (negotiation env)

Core: `torch`, `transformers`, `accelerate`, `bitsandbytes`, `sentencepiece`, `protobuf`
Plus: `openai`, `anthropic`, `python-dotenv`, `matplotlib`, `streamlit`, `huggingface_hub`

### Pre-downloaded Models

All model weights are cached in `$HF_HOME` (`/projects/.../amachado.up/huggingface/`).
Models were downloaded from a `dev-x86` interactive session (which has internet access).
GPU partitions have **no internet**, so `TRANSFORMERS_OFFLINE=1` and `HF_DATASETS_OFFLINE=1` are set in the server profile.

| Size Group | Models |
|------------|--------|
| very_small | `meta-llama/Llama-3.1-8B-Instruct`, `google/gemma-3-4b-it`, `mistralai/Ministral-3-8B-Instruct-2512` (vlm), `Qwen/Qwen3.5-9B` |
| small | `meta-llama/Llama-2-13b-chat-hf`, `google/gemma-3-12b-it`, `mistralai/Ministral-3-14B-Instruct-2512` (vlm), `qwen/Qwen3-14B` |
| medium | `google/gemma-3-27b-it`, `mistralai/Mistral-Small-3.2-24B-Instruct-2506` (vlm, 8bit), `Qwen/Qwen3.5-27B` |
| big | `meta-llama/Llama-3.3-70B-Instruct` (4bit), `Qwen/Qwen2.5-72B-Instruct` |

To download additional models, get a dev-x86 session with internet:
```bash
srun --account=f202500007hpcvlabuportox --partition=dev-x86 --nodes=1 --time=4:00:00 --pty bash
export HF_HOME=/projects/F202500007HPCVLABUPORTO/amachado.up/huggingface
source /projects/F202500007HPCVLABUPORTO/amachado.up/miniconda3/etc/profile.d/conda.sh
conda activate /projects/F202500007HPCVLABUPORTO/amachado.up/envs/negotiation
python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; AutoTokenizer.from_pretrained('model/name'); AutoModelForCausalLM.from_pretrained('model/name', torch_dtype='auto', device_map='cpu')"
```

### Running Experiments

The server profile is at `slurm/servers/deucalion.sh`. All SLURM parameters can be overridden at launch time.

```bash
# From the repo root on Deucalion
cd /projects/F202500007HPCVLABUPORTO/amachado.up/MultiAgent-Negotiation

# Dry run (inspect commands without submitting)
SERVER=deucalion DRY_RUN=1 bash slurm/launch.sh

# Run all experiments with defaults (normal-a100-40, 2 GPUs, 48h)
SERVER=deucalion bash slurm/launch.sh

# Choose partition and resources
SERVER=deucalion PARTITION=dev-a100-40 TIME=00:15:00 GPUS=1 EXPERIMENTS="buysell_section_one" SIZES="very_small" bash slurm/launch.sh

# Experiments with their own model lists  — use SIZES=none
SERVER=deucalion PARTITION=dev-a100-40 TIME=00:15:00 GPUS=1 EXPERIMENTS="buysell_section_one" SIZES=none bash slurm/launch.sh

# Monitor
squeue --me
tail -f logs/slurm/<job_name>_*.log
```

### Git Pull (SSH key setup per session)
The SSH agent does not persist across login sessions. Before `git pull`, run:

```bash
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
git pull
```

### Some Warnings 
- **Home quota:** Never install anything in `/home/`. The 25 GB / 25,000 file limit fills up fast. Use symlinks for `.vscode-server`, `.cache`, etc.
- **No internet on GPU partitions:** All models must be pre-downloaded. The profile sets `TRANSFORMERS_OFFLINE=1` automatically.
- **GPU billing:** 1 GPU = 32 CPUs. Requesting >32 CPUs per GPU bills for extra GPU-hours.
- **tmux for long sessions:** Start `tmux` on the login node before `srun` so model downloads survive SSH disconnects.
- **VS Code Remote:** Works via Remote-SSH extension. Requires `~/.vscode-server` symlinked to `/projects/` to avoid home quota.
- **HuggingFace auth:** Token is stored in `$HF_HOME/token`. If you get 401 errors, re-login: `python -c "from huggingface_hub import login; login()"` (from a dev-x86 session with internet).

---