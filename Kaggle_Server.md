# Kaggle (free GPU notebooks)

The dispatcher lives in `kaggle/` and is analogous to `slurm/`. It uses the same `configs/experiments.yaml`, the same `runner/run_experiment.py`, and the same `EXPERIMENTS` / `SIZES` env-var contract as `slurm/launch.sh`.

## Kaggle resources and limits

| Resource | Limit |
| --- | --- |
| **Free GPU quota** | ~30 GPU-hours / week per account |
| **Wall time per kernel** | 12 hours hard limit |
| **Concurrent kernels** | ~4–5 running at once (rest queue) |
| **Output size** | 20 GB max in `/kaggle/working` |
| **Internet** | available (we enable it; required for HF Hub + git clone) |
| **Persistent state** | none — every kernel run starts from a clean image |

Available accelerators (set via `KAGGLE_GPU_TYPE` directly with Kaggle CLI IDs):

| `KAGGLE_GPU_TYPE` | Hardware | Total VRAM |
| --- | --- | --- |
| `NvidiaTeslaT4` (default) | T4-class GPU |
| `NvidiaTeslaP100` | P100 |


## One-time setup

### 1. Install the Kaggle CLI

```bash
pip install kaggle           # CLI >= 1.8.0 supports the new access tokens
kaggle --version
```

### 2. Get an API token and put credentials in `.env`

On https://www.kaggle.com/settings → **API** → **Generate New Token**, download the JSON. Open it and copy the fields into the repo's `.env` with a `KAGGLE_` prefix (the CLI maps `KAGGLE_<KEY>` → `<key>` automatically):

```bash
# .env
KAGGLE_API_TOKEN=your-api-token
```

```bash
set -a; source .env; set +a
kaggle kernels list --mine
```

### 3. Add Kaggle Secrets for the kernel

The kernel runs as your user and inherits secrets you've registered in your Kaggle account. Open any of your kernels (or create a throwaway one), go to **Add-ons → Secrets**, and add:

| Secret label | When you need it |
| --- | --- |
| `HF_TOKEN` | Always — Llama and Gemma weights are gated |
| `GITHUB_TOKEN` | Always for this repo (it's private). PAT with `repo` read scope. |

These are visible to all kernels you submit when `enable_internet: true` (which `kernel-metadata.template.json` sets).

## Running experiments

### Default run (uses the experiment list baked into `kaggle/launch.sh`)

```bash
bash kaggle/launch.sh
```

### Pick experiments / sizes (same env-var contract as `slurm/launch.sh`)

```bash
EXPERIMENTS="buysell_section_two_personas" SIZES="very_small" bash kaggle/launch.sh
```

```bash
EXPERIMENTS="buysell_section_one trading_section_one" SIZES="very_small small" bash kaggle/launch.sh
```

Use `SIZES="none"` for experiments that define their own model list inline (no `_shared` lookup).

### Dry run (no submission)

```bash
DRY_RUN=1 bash kaggle/launch.sh
```

Prints the `kaggle kernels push` invocation and resolved kernel id for each combo, without contacting Kaggle.

### Override the GPU type

```bash
KAGGLE_GPU_TYPE="NvidiaL4" EXPERIMENTS="trading_section_one" SIZES="medium" bash kaggle/launch.sh
```

### Run a specific commit instead of `HEAD`

```bash
GIT_REF=abc1234 bash kaggle/launch.sh
```

### Push from a fork or alternate URL

```bash
GIT_REPO=https://github.com/your-fork/MultiAgent-Negotiation.git bash kaggle/launch.sh
```

## Pulling results back

There are two independent paths to get a kernel's results into your local repo:

### 1. Direct push (per-run branch, default)

After a successful run, the kernel pushes the new `.logs/...` files onto its own branch named `kaggle-results/<experiment>-<size>-<gitref8>` on this GitHub repo, using the same `GITHUB_TOKEN` Kaggle Secret it used to clone. Each run gets a unique branch, so concurrent kernels never collide.

To pull the results locally (replace the branch name with the one printed in the kernel log):

```bash
# one-shot, no merge commit on main:
git fetch origin 'kaggle-results/<experiment>-<size>-<ref8>'
git checkout origin/kaggle-results/<experiment>-<size>-<ref8> -- .logs/

# list all per-run branches:
git branch -r | grep kaggle-results/
```

Files land at the exact same `.logs/<section>/<experiment>/<size>/...` paths the SLURM/MIA runs produce, so the Streamlit dashboard sees them transparently.

The kernel prints a `[push] pushed to refs/heads/kaggle-results/<experiment>-<size>-<ref8>: <sha>` line on success. If the push fails for any reason the kernel prints `[push] FAILED: …` and the tarball/`fetch_results.py` flow below remains the fallback. No retry.

### 2. Tarball + `fetch_results.py` (always available, used as fallback)

`kaggle/launch.sh` is fire-and-forget. After submission, the manifest at `kaggle/manifest.jsonl` tracks every kernel you've pushed. To poll once and download anything that has finished:

```bash
python kaggle/fetch_results.py
```

To block and watch until all submitted kernels are terminal (fetched or failed):

```bash
python kaggle/fetch_results.py --watch                # 60 s polling
python kaggle/fetch_results.py --watch --interval 30  # custom interval
```

For each kernel that reports `complete`, `fetch_results.py`:
1. Runs `kaggle kernels output <id> -p kaggle/.outputs/<slug>/`.
2. Extracts `kaggle/.outputs/<slug>/results.tar.gz` over the local `.logs/` tree.
3. Marks the manifest row `fetched`.

## Inspecting and debugging

```bash
kaggle kernels list --mine                             # all kernels you've ever pushed
kaggle kernels status <user>/<slug>                    # one kernel's state
kaggle kernels output <user>/<slug> -p /tmp/out        # raw output download
```

To see the live log of a specific run, open the kernel page on https://www.kaggle.com/kernels — the right-hand "Logs" tab streams stdout as the kernel runs.

A run typically prints these phase markers (from `kaggle/kernel.py`):

```
[bootstrap] secret loaded: HF_TOKEN
[bootstrap] running: python runner/run_experiment.py --config ... --experiment ... --model_group ...
[bootstrap] wrote /kaggle/working/results.tar.gz (12.3 MB)
[push] pushed to refs/heads/kaggle-results/<experiment>-<size>-<ref8>: <sha>
```

`[push] FAILED: …` (or `[push] skipped: …`) replaces the last line when the kernel can't reach the remote; the tarball is still produced either way.

If a kernel fails, the manifest will show `status: "failed"` after the next `fetch_results.py`. The Kaggle web UI shows the full traceback.

## Caveats

- **Quota.** A `T4 x2` kernel that runs for 8 h burns 8 h of your 30 h/week. Plan submissions accordingly; `manifest.jsonl` is the record of what you've spent.
- **First-run model download.** `transformers.from_pretrained()` pulls the weights from HF Hub on every kernel run (no persistent cache between runs). Expect ~5–10 min for an 8B model. This is noise next to a multi-hour experiment but adds up if you submit many short kernels.
- **VRAM ceiling.** `T4 x2` (32 GB) fits very_small (4–9B) comfortably. `small` (12–14B) needs careful quantization. `medium` (24–27B) requires `L4 x4`. `big` (70B) is impractical on free Kaggle even with 4-bit.
- **Accelerator IDs.** Kaggle CLI expects accelerator IDs such as `NvidiaTeslaT4`. `kaggle/launch.sh` now uses those IDs directly and passes them via `kaggle kernels push --accelerator ...`.
- **Repo visibility.** The repo is private; the kernel reads `GITHUB_TOKEN` from Kaggle Secrets and passes it via `git -c http.extraHeader=...` so the token is never stored in `.git/config`. Use a PAT with read access to this repo.
- **Concurrency.** Kaggle queues kernels past ~4–5 concurrent. Submitting 12 jobs is fine — they'll just complete serially.

## Quick reference

```bash
# 1. one-time
pip install kaggle
echo "KAGGLE_USERNAME=...\nKAGGLE_KEY=..." >> .env

# 2. dry run
DRY_RUN=1 bash kaggle/launch.sh

# 3. real submission (cheapest combo first)
EXPERIMENTS="buysell_section_two_personas" SIZES="very_small" bash kaggle/launch.sh

# 4. wait + collect
python kaggle/fetch_results.py --watch
```
