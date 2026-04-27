#!/usr/bin/env python3
"""Poll submitted Kaggle kernels and pull their results into the local repo.

Reads kaggle/manifest.jsonl (one row per push from launch.sh), runs
`kaggle kernels status` on each row that hasn't been fetched yet, and for
completed ones runs `kaggle kernels output` to download results.tar.gz,
then extracts that archive over the local experiments/ tree so the existing
analysis tooling sees the runs as if they had landed from a SLURM job.

Idempotent: re-running picks up where the previous run left off. The
manifest is rewritten in place with status updates.

Usage:
    python kaggle/fetch_results.py           # poll once and exit
    python kaggle/fetch_results.py --watch   # poll every 60 s until all fetched/failed
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tarfile
import time
from pathlib import Path

KAGGLE_DIR = Path(__file__).resolve().parent
REPO_ROOT = KAGGLE_DIR.parent
MANIFEST = KAGGLE_DIR / "manifest.jsonl"
OUTPUTS_DIR = KAGGLE_DIR / ".outputs"

TERMINAL = {"fetched", "failed"}


def read_manifest() -> list[dict]:
    if not MANIFEST.exists():
        return []
    rows = []
    for line in MANIFEST.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def write_manifest(rows: list[dict]) -> None:
    tmp = MANIFEST.with_suffix(".jsonl.tmp")
    tmp.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    tmp.replace(MANIFEST)


def kernel_status(kernel_id: str) -> str:
    """Return one of: queued, running, complete, error, cancelRequested, cancelAcknowledged."""
    out = subprocess.run(
        ["kaggle", "kernels", "status", kernel_id],
        capture_output=True, text=True, check=False,
    )
    text = (out.stdout + out.stderr).strip()
    m = re.search(r'has status\s+"([^"]+)"', text, flags=re.IGNORECASE)
    if m:
        return m.group(1).lower()
    # Fallback: treat first non-empty word as the status
    return (text.split() or ["unknown"])[0].lower()


def download_output(kernel_id: str, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["kaggle", "kernels", "output", kernel_id, "-p", str(dest)],
        check=True,
    )


def extract_results(archive: Path) -> None:
    if not archive.exists():
        raise FileNotFoundError(f"results archive missing: {archive}")
    with tarfile.open(archive, "r:gz") as tf:
        tf.extractall(REPO_ROOT)
    print(f"  extracted {archive.name} → {REPO_ROOT}")


def poll_once(rows: list[dict]) -> tuple[int, int, int, int]:
    """Returns (running, complete_now, failed_now, fetched_now). Mutates rows."""
    running = complete_now = failed_now = fetched_now = 0
    for row in rows:
        if row["status"] in TERMINAL:
            continue
        status = kernel_status(row["kernel_id"])
        row["last_status"] = status
        if status in {"queued", "running", "scheduled"}:
            running += 1
            row["status"] = "pushed"
        elif status in {"error", "cancelacknowledged", "cancelled"}:
            row["status"] = "failed"
            failed_now += 1
            print(f"  ✗ {row['kernel_id']} failed (status={status})")
        elif status == "complete":
            complete_now += 1
            print(f"  ↓ {row['kernel_id']} complete; downloading…")
            dest = OUTPUTS_DIR / row["slug"]
            try:
                download_output(row["kernel_id"], dest)
                archive = dest / "results.tar.gz"
                extract_results(archive)
                row["status"] = "fetched"
                row["fetched_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                fetched_now += 1
            except Exception as e:
                print(f"  ✗ fetch error for {row['kernel_id']}: {e}")
                row["status"] = "failed"
                row["error"] = str(e)
                failed_now += 1
        else:
            running += 1
    return running, complete_now, failed_now, fetched_now


def summarise(rows: list[dict]) -> None:
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print("Manifest summary:")
    for status, n in sorted(counts.items()):
        print(f"  {status:>10} : {n}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--watch", action="store_true", help="poll until all rows are terminal")
    p.add_argument("--interval", type=int, default=60, help="seconds between polls in --watch")
    args = p.parse_args()

    if not MANIFEST.exists():
        print(f"No manifest at {MANIFEST}; submit something with kaggle/launch.sh first.")
        return 0

    while True:
        rows = read_manifest()
        if not rows:
            print("Manifest is empty.")
            return 0

        running, complete, failed, fetched = poll_once(rows)
        write_manifest(rows)
        print(
            f"poll: running={running} complete={complete} "
            f"failed={failed} fetched={fetched}"
        )
        summarise(rows)

        outstanding = sum(1 for r in rows if r["status"] not in TERMINAL)
        if not args.watch or outstanding == 0:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
