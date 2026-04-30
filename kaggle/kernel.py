"""Kaggle kernel body for one (experiment, size) run of NegotiationArena.

Substituted by kaggle/render_kernel.py:
    {{EXPERIMENT}}     experiment name from configs/experiments.yaml
    {{SIZE}}           model_group ("none" if the experiment defines its own list)
    {{GIT_REF}}        commit hash to check out
    {{GIT_REPO}}       HTTPS git URL
    {{KAGGLE_GPU_TYPE}} requested Kaggle accelerator
    {{SLUG}}           kernel slug (used in the push commit message)
    {{KERNEL_ID}}      "<user>/<slug>" (used in the push commit message)
    {{SUBMITTED_AT}}   render-time UTC timestamp (used in the push commit message)

Bootstraps the env, runs runner/run_experiment.py exactly as the SLURM path does
(transformers downloads weights from HF Hub on first use), tars the .logs/ tree
into /kaggle/working/results.tar.gz, and then tries to push the new .logs/
files onto its own `kaggle-results/<experiment>-<size>-<ref8>-<YYYYMMDD-HH>` branch on GitHub
(unique per run, so concurrent kernels never collide). The tarball is the
fallback if the push fails.
"""
import datetime
import os
import shutil
import subprocess
import sys
import base64

REPO_DIR = "/kaggle/working/repo"
HF_HOME = "/kaggle/working/hf_cache"
RESULT_TAR = "/kaggle/working/results.tar.gz"
_GIT_REF = "{{GIT_REF}}"  # substituted by render_kernel.py; sliced below for branch name
_RUN_TS = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H")
RESULTS_BRANCH = f"kaggle-results/{{EXPERIMENT}}-{{SIZE}}-{_GIT_REF[:8]}-{_RUN_TS}"
RESULTS_WT = "/kaggle/working/results-wt"


def validate_runtime() -> None:
    """Fail fast when Kaggle assigns an incompatible GPU/runtime."""
    try:
        import torch
    except Exception as exc:
        print(f"[bootstrap] warning: unable to import torch for runtime check: {exc}")
        return

    if not torch.cuda.is_available():
        print("[bootstrap] CUDA unavailable; proceeding on CPU")
        return

    gpu_name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    print(f"[bootstrap] GPU: {gpu_name} | compute capability sm_{capability[0]}{capability[1]}")

    # The Kaggle image currently installs a PyTorch build that requires sm_70+.
    if capability[0] < 7:
        requested = "{{KAGGLE_GPU_TYPE}}"
        raise RuntimeError(
            f"Incompatible Kaggle GPU assigned: {gpu_name} (sm_{capability[0]}{capability[1]}). "
            f"This run requires a T4/L4-class GPU because the installed PyTorch build "
            f"supports sm_70+. Re-submit with KAGGLE_GPU_TYPE='{requested}' and ensure "
            f"`kaggle kernels push` passes `--accelerator`."
        )

def export_secrets() -> None:
    """Inject API keys via local template substitution."""
    os.environ["GITHUB_TOKEN"] = "{{GITHUB_TOKEN}}"
    os.environ["HF_TOKEN"] = "{{HF_TOKEN}}"
    print("[bootstrap] secrets injected via template")

def clone_repo() -> None:
    """Clone the repo, using GITHUB_TOKEN for private repos when present.

    GitHub git-over-HTTPS expects Basic auth semantics for PATs. Passing the
    credential via `git -c http.extraHeader=...` avoids persisting secrets to
    `.git/config` after the clone (unlike embedding them in the URL).
    """
    if os.path.isdir(REPO_DIR):
        return
    cmd = ["git"]
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    repo_url = "{{GIT_REPO}}"
    if token:
        basic_auth = base64.b64encode(f"x-access-token:{token}".encode("utf-8")).decode("ascii")
        cmd += [
            "-c",
            f"http.extraHeader=Authorization: Basic {basic_auth}",
            "-c",
            "credential.helper=",
        ]
    elif "github.com" in repo_url:
        # Surface this before git emits "No such device or address" for stdin auth.
        raise RuntimeError(
            "GITHUB_TOKEN not in env. The repo is private; attach the GITHUB_TOKEN "
            "secret to this kernel via Add-ons → Secrets, or enable 'auto-attach to "
            "new notebooks' on the secret in your Kaggle account settings."
        )
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    cmd += ["clone", repo_url, REPO_DIR]
    subprocess.run(cmd, check=True, env=env)
    subprocess.run(["git", "-C", REPO_DIR, "checkout", "{{GIT_REF}}"], check=True, env=env)


def install_deps() -> None:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "--upgrade", "transformers"],
        check=True,
    )
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q",
         "-r", f"{REPO_DIR}/requirements.txt",
         "accelerate", "bitsandbytes",
         "sentencepiece", "protobuf", "pyyaml"],
        check=True,
    )


def run_experiment() -> None:
    os.environ["HF_HOME"] = HF_HOME
    os.makedirs(HF_HOME, exist_ok=True)
    cmd = [
        sys.executable, "runner/run_experiment.py",
        "--config", "configs/experiments.yaml",
        "--experiment", "{{EXPERIMENT}}",
    ]
    if "{{SIZE}}" not in ("", "none"):
        cmd += ["--model_group", "{{SIZE}}"]
    print(f"[bootstrap] running: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=REPO_DIR, check=True)


def archive_results() -> None:
    subprocess.run(
        ["tar", "-czf", RESULT_TAR, "-C", REPO_DIR, ".logs"],
        check=True,
    )
    size_mb = os.path.getsize(RESULT_TAR) / (1024 * 1024)
    print(f"[bootstrap] wrote {RESULT_TAR} ({size_mb:.1f} MB)")


def _auth_git_args(token: str) -> list[str]:
    """git -c flags that send the PAT as a Basic auth header (no on-disk creds)."""
    basic_auth = base64.b64encode(f"x-access-token:{token}".encode("utf-8")).decode("ascii")
    return [
        "-c", f"http.extraHeader=Authorization: Basic {basic_auth}",
        "-c", "credential.helper=",
    ]


def push_results_to_github() -> None:
    """Push the run's new .logs/ files onto a per-run branch on GitHub.

    Branch name: kaggle-results/<experiment>-<size>-<ref8> — unique per run,
    so concurrent kernels never collide. No retry: any push failure is logged
    and swallowed; results.tar.gz is left in the kernel's output for manual
    download from kaggle.com as the fallback.
    """
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if not token:
        print("[push] skipped: GITHUB_TOKEN not in env")
        return

    logs_src = os.path.join(REPO_DIR, ".logs")
    if not os.path.isdir(logs_src):
        print(f"[push] skipped: no {logs_src} produced by this run")
        return

    auth = _auth_git_args(token)
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"

    try:
        base_ref = "HEAD"  # per-run branch; always a fresh branch off the cloned ref
        if os.path.exists(RESULTS_WT):
            subprocess.run(
                ["git", "-C", REPO_DIR, "worktree", "remove", "--force", RESULTS_WT],
                env=env, check=False,
            )
        subprocess.run(
            ["git", "-C", REPO_DIR, "worktree", "add", "-B", RESULTS_BRANCH,
             RESULTS_WT, base_ref],
            env=env, check=True,
        )

        shutil.copytree(logs_src, os.path.join(RESULTS_WT, ".logs"), dirs_exist_ok=True)

        subprocess.run(["git", "-C", RESULTS_WT, "add", "-f", ".logs"], env=env, check=True)

        # Bail out if there is nothing new to commit (e.g. an identical re-run).
        diff = subprocess.run(
            ["git", "-C", RESULTS_WT, "diff", "--cached", "--quiet"],
            env=env, check=False,
        )
        if diff.returncode == 0:
            print("[push] skipped: no new files to commit")
            return

        message = (
            "[kaggle] {{EXPERIMENT}} / {{SIZE}} @ {{GIT_REF}}\n"
            "\n"
            "slug:        {{SLUG}}\n"
            "kernel_id:   {{KERNEL_ID}}\n"
            "gpu_type:    {{KAGGLE_GPU_TYPE}}\n"
            "submitted:   {{SUBMITTED_AT}}\n"
        )
        subprocess.run(
            ["git", "-C", RESULTS_WT,
             "-c", "user.email=kaggle-bot@noreply.local",
             "-c", "user.name=Kaggle Kernel Bot",
             "commit", "-m", message],
            env=env, check=True,
        )

        sha = subprocess.run(
            ["git", "-C", RESULTS_WT, "rev-parse", "HEAD"],
            env=env, check=True, capture_output=True, text=True,
        ).stdout.strip()

        subprocess.run(
            ["git", "-C", RESULTS_WT, *auth, "push", "origin", RESULTS_BRANCH],
            env=env, check=True,
        )
        print(f"[push] pushed to refs/heads/{RESULTS_BRANCH}: {sha}")

    except subprocess.CalledProcessError as exc:
        print(f"[push] FAILED: {exc} (stdout={exc.stdout!r} stderr={exc.stderr!r})")
        print(f"[push] tarball at {RESULT_TAR} is intact; "
              "download it from this kernel's Output tab on kaggle.com and "
              "extract over the local repo to recover this run.")
    except Exception as exc:
        print(f"[push] FAILED: {exc}")
        print(f"[push] tarball at {RESULT_TAR} is intact; "
              "download it from this kernel's Output tab on kaggle.com and "
              "extract over the local repo to recover this run.")
    finally:
        if os.path.exists(RESULTS_WT):
            subprocess.run(
                ["git", "-C", REPO_DIR, "worktree", "remove", "--force", RESULTS_WT],
                env=env, check=False,
            )


def main() -> None:
    export_secrets()
    clone_repo()
    install_deps()
    validate_runtime()
    run_experiment()
    archive_results()
    push_results_to_github()


if __name__ == "__main__":
    main()
