"""Pre-download HF model snapshots for a size group defined in configs/experiments.yaml.

Usage (run from a node with internet, e.g. Deucalion dev-x86):
    export HF_HOME=/projects/.../huggingface
    python download_models.py very_small
    python download_models.py small --config configs/experiments.yaml
    python download_models.py medium --dry-run

Size groups come from the `_shared.models_<size>` lists in the YAML config.
"""
import argparse
import os
import sys
from pathlib import Path

import yaml
from huggingface_hub import snapshot_download
from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
from ratbench.utils import normalize_model  # noqa: E402

IGNORE_PATTERNS = [
    "*.gguf",
    "*.onnx",
    "*.onnx_data",
    "original/*",
    "consolidated.*",
    "*.pt",
    "*.bin",
]


def resolve_models(config_path: Path, size: str) -> list[dict]:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    key = f"models_{size}"
    shared = cfg.get("_shared", {})
    if key not in shared:
        available = sorted(k.removeprefix("models_") for k in shared if k.startswith("models_"))
        sys.exit(f"Group '{size}' not found in {config_path}. Available: {available}")
    return [normalize_model(m) for m in shared[key]]


def download(model_id: str, hf_home: str) -> None:
    print(f"\n→ {model_id}")
    try:
        path = snapshot_download(
            repo_id=model_id,
            ignore_patterns=IGNORE_PATTERNS,
            cache_dir=os.path.join(hf_home, "hub"),
        )
        print(f"  cached at: {path}")
    except GatedRepoError:
        sys.exit(
            f"  Gated repo. Accept the license on the model page and ensure "
            f"$HF_HOME/token is set (huggingface-cli login)."
        )
    except RepositoryNotFoundError:
        sys.exit(f"  Repo not found: {model_id}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("size", choices=["very_small", "small", "medium", "big"])
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "experiments.yaml"))
    ap.add_argument("--dry-run", action="store_true", help="List models without downloading.")
    args = ap.parse_args()

    hf_home = os.environ.get("HF_HOME")
    if not hf_home:
        sys.exit("HF_HOME is not set. Export it to the shared project cache before running.")

    models = resolve_models(Path(args.config), args.size)
    print(f"Group '{args.size}' → {len(models)} model(s)")
    for m in models:
        flags = []
        if m["quantization"]:
            flags.append(f"quant={m['quantization']}")
        if m["model_type"] != "llm":
            flags.append(f"type={m['model_type']}")
        suffix = f"  [{', '.join(flags)}]" if flags else ""
        print(f"  - {m['id']}{suffix}")

    if args.dry_run:
        return

    print(f"\nHF_HOME={hf_home}")
    for m in models:
        download(m["id"], hf_home)
    print("\nDone.")


if __name__ == "__main__":
    main()
