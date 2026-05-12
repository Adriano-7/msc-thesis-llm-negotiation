#!/usr/bin/env python3
"""
Delete run directories from .logs/ that are:
  - "unknown" (pre-fix): incomplete game with no captured error_type
  - OutOfMemoryError: game that hit CUDA OOM

Usage:
  python cleanup_error_runs.py           # dry-run (safe, no deletions)
  python cleanup_error_runs.py --delete  # actually delete
"""

import argparse
import json
import shutil
from pathlib import Path

LOGS_ROOT = Path(__file__).parent / ".logs"


def classify(gs_path: Path) -> str | None:
    """Return 'unknown', 'oom', or None (keep)."""
    try:
        with open(gs_path) as f:
            data = json.load(f)
        game_states = data.get("game_state", [])
        if not isinstance(game_states, list) or not game_states:
            return None
        last = game_states[-1]
        ci = last.get("current_iteration")
        et = last.get("error_type")
        em = last.get("error_message") or ""
        if et == "OutOfMemoryError":
            return "oom"
        if "Some modules are dispatched on the CPU or the disk" in em:
            return "cpu_dispatch"
        if et == "KeyError" and "player_public_info_dict" in em:
            return "key_error_infra"
        if et == "AttributeError" and "execute_trade" in em:
            return "attr_error_infra"
        if et == "OSError" and "Unable to load vocabulary" in em:
            return "oserror_vocab"
        if et == "RuntimeError":
            return "runtime_error"
        if et == "TokenizerException":
            return "tokenizer_exception"
        if et == "ValueError" and "Couldn't instantiate the backend tokenizer" in em:
            return "backend_tokenizer"
        if et == "ValueError" and "Some kwargs in ['fix_mistral_regex'" in em:
            return "mistral_kwargs"
        if ci != "END" and (et is None or et == ""):
            return "unknown"
    except Exception:
        pass
    return None


def remove_empty_parents(path: Path, stop_at: Path) -> None:
    """Walk up and remove empty directories until stop_at."""
    current = path.parent
    while current != stop_at and current.is_dir():
        try:
            current.rmdir()  # only succeeds if empty
            current = current.parent
        except OSError:
            break


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--delete", action="store_true", help="Actually delete (default is dry-run)")
    args = parser.parse_args()
    dry_run = not args.delete

    if dry_run:
        print("DRY-RUN mode — pass --delete to actually remove files\n")

    unknown: list[Path] = []
    oom: list[Path] = []
    cpu_dispatch: list[Path] = []
    key_error_infra: list[Path] = []
    attr_error_infra: list[Path] = []
    oserror_vocab: list[Path] = []
    runtime_error: list[Path] = []
    tokenizer_exception: list[Path] = []
    backend_tokenizer: list[Path] = []
    mistral_kwargs: list[Path] = []

    for gs_path in sorted(LOGS_ROOT.rglob("game_state.json")):
        kind = classify(gs_path)
        run_dir = gs_path.parent
        if kind == "unknown":
            unknown.append(run_dir)
        elif kind == "oom":
            oom.append(run_dir)
        elif kind == "cpu_dispatch":
            cpu_dispatch.append(run_dir)
        elif kind == "key_error_infra":
            key_error_infra.append(run_dir)
        elif kind == "attr_error_infra":
            attr_error_infra.append(run_dir)
        elif kind == "oserror_vocab":
            oserror_vocab.append(run_dir)
        elif kind == "runtime_error":
            runtime_error.append(run_dir)
        elif kind == "tokenizer_exception":
            tokenizer_exception.append(run_dir)
        elif kind == "backend_tokenizer":
            backend_tokenizer.append(run_dir)
        elif kind == "mistral_kwargs":
            mistral_kwargs.append(run_dir)

    print(f"Unknown (pre-fix) runs : {len(unknown)}")
    print(f"OutOfMemoryError runs  : {len(oom)}")
    print(f"CPU dispatch errors    : {len(cpu_dispatch)}")
    print(f"KeyError infra bugs    : {len(key_error_infra)}")
    print(f"AttributeError infra   : {len(attr_error_infra)}")
    print(f"OSError vocab load     : {len(oserror_vocab)}")
    print(f"RuntimeError           : {len(runtime_error)}")
    print(f"TokenizerException     : {len(tokenizer_exception)}")
    print(f"Backend tokenizer      : {len(backend_tokenizer)}")
    print(f"Mistral kwargs         : {len(mistral_kwargs)}")
    total = len(unknown) + len(oom) + len(cpu_dispatch) + len(key_error_infra) + len(attr_error_infra) + len(oserror_vocab) + len(runtime_error) + len(tokenizer_exception) + len(backend_tokenizer) + len(mistral_kwargs)
    print(f"Total to delete        : {total}\n")

    all_runs = unknown + oom + cpu_dispatch + key_error_infra + attr_error_infra + oserror_vocab + runtime_error + tokenizer_exception + backend_tokenizer + mistral_kwargs

    if dry_run:
        for label, group in [
            ("unknown", unknown), ("OOM", oom), ("CPU dispatch", cpu_dispatch),
            ("KeyError infra", key_error_infra), ("AttributeError infra", attr_error_infra),
            ("OSError vocab", oserror_vocab), ("RuntimeError", runtime_error),
            ("TokenizerException", tokenizer_exception), ("Backend tokenizer", backend_tokenizer),
            ("Mistral kwargs", mistral_kwargs),
        ]:
            if group:
                print(f"Sample {label} runs (first 5):")
                for p in group[:5]:
                    print(f"  {p.relative_to(LOGS_ROOT.parent)}")
                print()
        return

    deleted = 0
    for run_dir in all_runs:
        if run_dir.exists():
            shutil.rmtree(run_dir)
            remove_empty_parents(run_dir, LOGS_ROOT)
            deleted += 1

    print(f"Deleted {deleted} run directories.")


if __name__ == "__main__":
    main()
