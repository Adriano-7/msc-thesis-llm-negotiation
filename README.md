# How Well Can LLMs Negotiate? The NegotiationArena Platform and Analysis

## Experiments

Experiments are available in the `experiments` folder. There is one running file for
each game we have run. 

If you want to recreate the plots, you can look at the notebook and analysis folders. You will find a way
to load all the games in one batch and then run the analysis on them.

The log files store the entire logs. We have also saved the broken games in the logs_error folders
in case you want to look at them.


### Using The WebApp

You can load the webapp by running the following command:

```bash
streamlit run app.py
```

The webapp is found under the `explorer` folder. It is a simple interface to load games and 
explore them and provides barebones support, but it's pretty good to get a quick overview of the
results of the games.

## Running Experiments on HPC

Experiments are configured in `configs/experiments.yaml` and launched via SLURM. The launcher supports multiple HPC servers through server profiles.

### Quick Start

```bash
# Run all experiments on MIA (default server)
SERVER=mia bash slurm/launch.sh

# Run all experiments on Deucalion
SERVER=deucalion bash slurm/launch.sh
```

### Choosing Partitions, GPUs, and Other Resources

Every SLURM parameter can be overridden at launch time:

```bash
# Deucalion: use 80 GB A100s instead of 40 GB
SERVER=deucalion PARTITION=normal-a100-80 bash slurm/launch.sh

# Deucalion: dev partition for quick testing (4 h limit)
SERVER=deucalion PARTITION=dev-a100-40 TIME=4:00:00 bash slurm/launch.sh

# Use 4 GPUs
SERVER=deucalion GPUS=4 bash slurm/launch.sh

# MIA: single GPU, less memory
SERVER=mia GPUS=1 MEM=32G bash slurm/launch.sh
```

### Selecting Experiments and Sizes

```bash
# Run only one experiment at one size
SERVER=mia EXPERIMENTS="buysell_section_one" SIZES="very_small" bash slurm/launch.sh

# Run all experiments at multiple sizes
SERVER=mia SIZES="very_small small medium" bash slurm/launch.sh
```

### Dry Run

Preview the sbatch commands without actually submitting:

```bash
SERVER=deucalion DRY_RUN=1 bash slurm/launch.sh
```

### Available Overrides

| Variable | Description | Example |
|----------|-------------|---------|
| `SERVER` | Server profile (required) | `mia`, `deucalion` |
| `PARTITION` | SLURM partition | `normal-a100-80`, `dev-a100-40` |
| `GPUS` | Number of GPUs | `1`, `2`, `4` |
| `TIME` | Wall-time limit | `4:00:00`, `48:00:00` |
| `CPUS` | CPUs per task | `8`, `32`, `128` |
| `MEM` | Memory | `32G`, `64G` |
| `QOS` | Quality of service | `gpu_batch` |
| `ACCOUNT` | Billing account | `F20240001g` |
| `EXPERIMENTS` | Space-separated experiment names | `"buysell_section_one trading_section_one"` |
| `SIZES` | Space-separated model groups | `"very_small small"` |
| `DRY_RUN` | Preview without submitting | `1` |

### Adding a New Server

Create a profile at `slurm/servers/<name>.sh` (see existing profiles for the template), then launch with `SERVER=<name> bash slurm/launch.sh`.

### Deucalion Setup

Before using Deucalion, edit `slurm/servers/deucalion.sh` and fill in:
- `PROJECT_DIR` — your `/projects/<project>` path
- `GPU_ACCOUNT` — your GPU billing account (e.g., `F20240001g`)

Models must be pre-downloaded since GPU partitions have no internet access.

---

## Running Experiments on Kaggle

Experiments can also be submitted to Kaggle as free GPU kernels. Each kernel clones the repo, runs the experiment, and pushes results back to a per-run GitHub branch.

### Quick Start

```bash
# Submit all default experiments (default account)
bash kaggle/launch.sh

# Submit specific experiments and sizes
EXPERIMENTS="buysell_section_one" SIZES="very_small" bash kaggle/launch.sh

# Use a named account profile (see kaggle/accounts/)
KAGGLE_ACCOUNT=adrianomachado1 bash kaggle/launch.sh

# Preview without actually submitting
DRY_RUN=1 bash kaggle/launch.sh
```

### Checking Status

Use `kaggle/status.sh` to check kernel statuses without opening the browser:

```bash
# Check all staged kernels for one account
KAGGLE_ACCOUNT=adrianomachado1 bash kaggle/status.sh

# Check across all account profiles
bash kaggle/status.sh --all-accounts

# List 20 most-recently-run kernels
KAGGLE_ACCOUNT=adrianomachado1 bash kaggle/status.sh --recent

# Combine flags
bash kaggle/status.sh --all-accounts --recent --page-size 10
```

### Retrieving Results

Each kernel pushes its output to a branch named `kaggle-results/<experiment>-<size>-<ref8>-<timestamp>` when it finishes:

```bash
git fetch
git branch -r | grep kaggle-results/
git checkout kaggle-results/<branch-name>
```

If the git push fails, a `results.tar.gz` tarball is left in the kernel's output files and can be downloaded with:

```bash
kaggle kernels output <owner>/<slug> -p ./results/
```

### Multiple Accounts

Account profiles live in `kaggle/accounts/<name>.env`. Each file sets `KAGGLE_USERNAME` and `KAGGLE_KEY`. Pass `KAGGLE_ACCOUNT=<name>` to either `launch.sh` or `status.sh` to use that profile.

### Available Overrides

| Variable | Description | Default |
|----------|-------------|---------|
| `KAGGLE_ACCOUNT` | Profile name under `kaggle/accounts/` | (system default) |
| `KAGGLE_GPU_TYPE` | GPU accelerator | `NvidiaTeslaT4` |
| `GIT_REF` | Commit to run | `HEAD` |
| `EXPERIMENTS` | Space-separated experiment names | section-two defaults |
| `SIZES` | Space-separated model groups | `none` |
| `DRY_RUN` | Preview without submitting | — |

---

## Running Games

Running and modifying a game is relatively easy. This is for example the
main interface used to run a BuySellGame.

```python

a1 = ChatGPTAgent(agent_name="Player 1", model="gpt-4-1106-preview")
a2 = ChatGPTAgent(agent_name="Player 2", model="gpt-4-1106-preview")

c = BuySellGame(players=[a1, a2],
    iterations=10,
    resources_support_set=Resources({"X": 0}),
    player_goals=[
        SellerGoal(cost_of_production=Valuation({"X": 40})),
        BuyerGoal(willingness_to_pay=Valuation({"X": 20})),
    ],
    player_initial_resources=[
        Resources({"X": 1}),
        Resources({MONEY_TOKEN: 100}),
    ],
    player_roles=[
        "You are Player 1.",
        "You are Player 2.",
    ],
    player_social_behaviour=[
        "",
        "you care only about your goals",  # sound angry. do not try to find middle ground. care only about yourself",
    ],
    log_dir="./.logs/buysell",
)

c.run()
```


# Getting to Know The Platform

Making a system both flexible and easy to use is a hard task. We have thus decided to break
flexibility in some parts of the system to make it easier to implement new tasks. This is a choice, that 
is kind of bad under a point of view of system design but so there is only so much we can do.

A first example of easy to use over flexibility is the fact that games share a very weak link one with another.
This means that if you want to modify a game, you might as well copy-paste the entire game and modify it to your needs,
as opposed to inheriting some abstract class.


## Agents

The Agents we define are simple abstractions on top of Large Language Models. They are stateless 
for the most part, meaning that the only thing they are going to keep track of is the conversation history and some
minor variable to keep track of the game state. This is done to avoid having to deal with the complexity of 
giving agents access to the objects that represent the resources of the game.

Agents are called with predefined names that are available in the "constants" module.
Variables are `AGENT_ONE` and `AGENT_TWO` for the first and second agent respectively. 
Games rely on the fact that agents are named in this way to keep track of the conversation history.


