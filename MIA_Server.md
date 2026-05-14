# HPC (MSc. AI)

This is a (very) short documentation on the high-performance computing (HPC) cluster located in the School of Engineering, University of Porto, and maintained by the Computer Science and Engineering (CSE) department.
This HPC cluster aims to support curricular and research activities for students in the Artificial Intelligence Master's program.

This **is not** intended to be a thorough tutorial on Linux and/or SLURM. Users are expected to have proficiency on both in order to understand this quick start guide.

## Overview

Using your favorite SSH client enter the login node of the cluster:

```bash
ssh <your username>@10.227.246.75
```

Your `<your username>` is usually something like `up123456789`. If you have been given clearance to use the cluster you should have received your (initial) password. Talk to your Advisor if you have any questions.

## Compute Resources

Here is the breakdown of the available computing resources:

| Node | GPU Model | GPU Count | VRAM (per GPU) | Driver Version | CUDA Version |
| --- | --- | --- | --- | --- | --- |
| `srv01` | NVIDIA L40S | 2 | 46,068 MiB (~48GB) | 575.57.08 | 12.9 |
| `srv02` | NVIDIA L40S | 3 | 46,068 MiB (~48GB) | 575.57.08 | 12.9 |

## Partitions

There are currently three partitions and three levels of QoS.

| partition | description |
| --- | --- |
| `teach` | to be used for quick demonstrations only (e.g. classes) |
| `normal` | the normal partition that users are expected to use more often |
| `fast` | a partition for quick experiments of code (i.e. debug, before a large job submission) |

Not all users have access to all partitions, and each partition can only accept jobs with certain characteristics.

## Quality-of-Service

In the table below you'll find a short description with the main characteritics of each QoS policy. There are further SLURM parameterizations that are ommitted here for clarity of the description.

| QoS policy | description |
| --- | --- |
| `gpu` | allows jobs that request up to 2 gpus, both interactive (`srun`) and batch (`sbatch`); max time per job is 3 hours; only 1 job may be submitted/running at a time. |
| `gpu_batch` | allows **batch jobs only** (`sbatch`) with requests of up to 1 gpu; max time per job is 12 hours; each user is capped at 1 gpu total across all running jobs, and up to 10 submitted jobs. |
| `cpu` | allows both interactive (`srun`) and batch (`sbatch`) without gpus; max time per job is 30 minutes; up to 3 submitted jobs per user. |

## Specific Limits

Based on the QoS policies, user jobs are strictly capped at the following resource limits. Requests exceeding these bounds will fail or be rejected by SLURM:

| Resource | Maximum Limit | Notes |
| --- | --- | --- |
| **CPUs** | 2 CPUs per task | Job requests for > 2 CPUs (e.g., `--cpus-per-task=3`) will be rejected. |
| **Memory** | 8 GB | Job requests exceeding 8192 MB (e.g., `--mem=8193`) will fail. |
| **GPUs** | 2 GPUs (`gpu` QoS) / 1 GPU (`gpu_batch` QoS) | Only the `gpu` and `gpu_batch` QoS policies permit GPU allocation. Under `gpu_batch`, a user may hold at most 1 GPU total across all running jobs. |
| **Time (`cpu`)** | 30 minutes | Maximum execution time for the `cpu` QoS. |
| **Time (`gpu`)** | 3 hours | Maximum execution time for the `gpu` QoS. |
| **Time (`gpu_batch`)** | 12 hours | Maximum execution time for the `gpu_batch` QoS. |

## Users and SLURM accounts

In the table below you'll find which QoS policies are allowed under each partition and which group of users are allowed to use each partition.

| Partition | Allowed QoS policies | Allowed SLURM accounts |
| --- | --- | --- |
| `teach` | `gpu` | `faculty` |
| `normal` | `gpu_batch`, `cpu` | all accounts |
| `fast` | `cpu` | all accounts |

> **Note:** interactive jobs (`srun`) are **disabled on the `normal` partition** — attempting `srun -p normal ...` will be rejected with *"Interactive jobs (srun) are not allowed on partition 'normal'. Please use sbatch."* Since `teach` is faculty-only and `fast` allows only the `cpu` QoS (no GPUs), students currently have **no way to run interactive GPU jobs** and must use `sbatch` for any GPU work.

## Job submission examples

Jobs can be submitted by using the commands `srun` (interactive) or `sbatch` (non-interactive). The latter is usually prefered for longer running jobs, where you simply submit the job to the queue and it stays there until the system has the necessary resources to run it. When all tasks in your job are done (or if there is some error on your script) the job terminates and you can inspect the output.

Here are some working examples for the `fast` partition:

```bash
# default behaviour: cpu=1 mem=1G
srun -p fast --qos=cpu  --job-name "my_interactive" --pty bash -i

# raise mem=4G
srun -p fast --qos=cpu  --mem=4G  --job-name "my_interactive" --pty bash -i

# limit of qos=cpu
srun -p fast --qos=cpu  --cpus-per-task=2  --mem=8G  --job-name "my_interactive" --pty bash -i
```

Job submission examples that **will fail** (and why they will fail):

```bash
# must specify QoS
srun -p fast --job-name "my_bad" --pty bash -i

# qos=gpu not allowed on 'fast'
srun -p fast --qos=gpu  --job-name "my_bad" --pty bash -i

# gres=gpu not allowed on qos=cpu
srun -p fast --qos=cpu  --gres=gpu  --job-name "my_bad" --pty bash -i

# request exceeds cpu limits
srun -p fast --qos=cpu  --cpus-per-task=3  --mem=1G  --job-name "my_bad" --pty bash -i

# request exceeds mem limits
srun -p fast --qos=cpu  --cpus-per-task=2  --mem=8193  --job-name "my_bad" --pty bash -i
```

For a batch job, you first need to create a script (e.g. `minimal_gpu.sh`) that specifies the parameters of the job and what to execute. Create a file with this content:

```bash
#!/bin/bash

#SBATCH -p normal
#SBATCH --qos gpu_batch
#SBATCH --gres=gpu:1

echo "Hello world! I'm ready to do some GPU business..."
echo "-----------------------------------------------"
echo `hostname`
echo "-----------------------"
echo `uname -a`
echo "-----------------------"
nvidia-smi
```

Now run the following command:

```bash
sbatch minimal_gpu.sh
```

After job is completed you should see a file named `slurm-<XYZ>.out`, where `<XYZ>` is your job number, with the output of your script.

## Project status

This project is an ongoing effort from the CSE department. Sudden changes may occur. We appreciate your patience and contribution.

## System Environment
* **CUDA Version:** 12.9

