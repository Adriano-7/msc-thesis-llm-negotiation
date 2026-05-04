#!/usr/bin/env python3
"""Render a per-(experiment, size) Kaggle kernel staging directory.

Substitutes placeholders into kaggle/kernel.py and kaggle/kernel-metadata.template.json
and writes the pair to a fresh staging dir. The kernel itself downloads model weights
from HuggingFace Hub at runtime (same as the SLURM path), so no model map needed.

Outputs JSON to stdout describing the rendered staging dir; launch.sh consumes it.
"""
from __future__ import annotations

import argparse
import datetime
import json
import re
import sys
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
KAGGLE_DIR = Path(__file__).resolve().parent


def slugify(text: str) -> str:
    s = re.sub(r"[^a-z0-9-]+", "-", text.lower()).strip("-")
    return re.sub(r"-+", "-", s)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", required=True)
    p.add_argument("--size", default="none", help="model_group name or 'none'")
    p.add_argument("--git-ref", required=True, help="commit hash to check out in-kernel")
    p.add_argument("--git-repo", default="https://github.com/Adriano-7/MultiAgent-Negotiation.git")
    p.add_argument("--user", required=True, help="Kaggle username")
    p.add_argument("--out", required=True, help="staging dir to write into")
    p.add_argument("--gpu-type", default="T4 x2")
    p.add_argument("--extra-args", default="", help="Extra CLI args forwarded to run_experiment.py")
    p.add_argument("--kernel-template", default=str(KAGGLE_DIR / "kernel.py"))
    p.add_argument(
        "--metadata-template",
        default=str(KAGGLE_DIR / "kernel-metadata.template.json"),
    )
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    size_for_slug = args.size if args.size and args.size != "none" else "default"
    slug_base = slugify(f"{args.experiment}-{size_for_slug}")
    # Kaggle enforces a 50-char slug limit; reserve 9 chars for "-{ref8}"
    max_base = 50 - 9
    if len(slug_base) > max_base:
        slug_base = slug_base[:max_base].rstrip("-")
    slug = f"{slug_base}-{args.git_ref[:8]}"
    kernel_id = f"{args.user}/{slug}"
    # Kaggle auto-slugs the title; pick a title whose auto-slug == our slug
    # so the API doesn't reject (or warn) about title/id mismatch.
    title = slug.replace("-", " ")

    submitted_at = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    kernel_template = Path(args.kernel_template).read_text()
    rendered = (
        kernel_template
        .replace("{{EXPERIMENT}}", args.experiment)
        .replace("{{SIZE}}", args.size or "none")
        .replace("{{GIT_REF}}", args.git_ref)
        .replace("{{GIT_REPO}}", args.git_repo)
        .replace("{{KAGGLE_GPU_TYPE}}", args.gpu_type)
        .replace("{{SLUG}}", slug)
        .replace("{{KERNEL_ID}}", kernel_id)
        .replace("{{SUBMITTED_AT}}", submitted_at)
        .replace("{{GITHUB_TOKEN}}", os.environ.get("GITHUB_TOKEN", ""))
        .replace("{{HF_TOKEN}}", os.environ.get("HF_TOKEN", ""))
        .replace("{{EXTRA_ARGS}}", args.extra_args)
    )
    (out_dir / "kernel.py").write_text(rendered)

    metadata = json.loads(Path(args.metadata_template).read_text())
    metadata["id"] = kernel_id
    metadata["title"] = title
    (out_dir / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    print(json.dumps({
        "slug": slug,
        "kernel_id": kernel_id,
        "experiment": args.experiment,
        "size": args.size,
        "gpu_type": args.gpu_type,
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
