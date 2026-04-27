#!/bin/bash
# ============================================================
# Kaggle launcher for NegotiationArena experiments.
#
# Sibling of slurm/launch.sh. For each (experiment × size) combo:
#   1. python render_kernel.py emits a staging dir (kernel.py +
#      kernel-metadata.json with the right Kaggle Models attached).
#   2. kaggle kernels push submits the kernel.
#   3. The slug is appended to kaggle/manifest.jsonl for fetch_results.py.
#
# Async only: this script returns immediately after pushing all kernels.
# Pull results later with:  python kaggle/fetch_results.py
#
# Usage:
#   bash kaggle/launch.sh
#   EXPERIMENTS="buysell_section_one" SIZES="very_small" bash kaggle/launch.sh
#   DRY_RUN=1 bash kaggle/launch.sh
#
# Required env / config:
#   KAGGLE_USER (or kaggle config view → username)   — your Kaggle handle
#   ~/.kaggle/kaggle.json (chmod 600)                — API token
#
# Optional env:
#   KAGGLE_GPU_TYPE  — default "NvidiaTeslaT4"  (also: "NvidiaTeslaP100", "NvidiaL4")
#   GIT_REF          — default $(git rev-parse HEAD)
#   GIT_REPO         — default https://github.com/Adriano-7/MultiAgent-Negotiation.git
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Load .env (KAGGLE_USERNAME, KAGGLE_KEY, etc.) ───────────
if [ -f "$REPO_DIR/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source "$REPO_DIR/.env"
    set +a
fi

# ── Resolve Kaggle username ─────────────────────────────────
# Prefer KAGGLE_USER, then KAGGLE_USERNAME from .env, then `kaggle config view`.
KAGGLE_USER="${KAGGLE_USER:-${KAGGLE_USERNAME:-}}"
if [ -z "${KAGGLE_USER:-}" ] && command -v kaggle >/dev/null 2>&1; then
    KAGGLE_USER="$(kaggle config view 2>/dev/null | awk -F': ' '/^- username:/ {print $2; exit}')"
fi
if [ -z "${KAGGLE_USER:-}" ]; then
    echo "ERROR: set KAGGLE_USERNAME (and KAGGLE_KEY) in .env, or KAGGLE_USER inline." >&2
    exit 1
fi

# ── Resolve git ref / repo ──────────────────────────────────
GIT_REF="${GIT_REF:-$(git -C "$REPO_DIR" rev-parse HEAD)}"
GIT_REPO="${GIT_REPO:-https://github.com/Adriano-7/MultiAgent-Negotiation.git}"
KAGGLE_GPU_TYPE="${KAGGLE_GPU_TYPE:-NvidiaTeslaT4}"

# ── Experiments and sizes (mirror slurm/launch.sh) ──────────
DEFAULT_SIZES=("none")
DEFAULT_EXPERIMENTS=(
    "buysell_section_two_personas"
    "trading_section_two_personas"
    "ultimatum_section_two_personas"
)

if [ "${SIZES:-}" = "none" ]; then
    SIZE_ARRAY=("none")
elif [ -n "${SIZES:-}" ]; then
    IFS=' ' read -ra SIZE_ARRAY <<< "$SIZES"
else
    SIZE_ARRAY=("${DEFAULT_SIZES[@]}")
fi

if [ -n "${EXPERIMENTS:-}" ]; then
    IFS=' ' read -ra EXP_ARRAY <<< "$EXPERIMENTS"
else
    EXP_ARRAY=("${DEFAULT_EXPERIMENTS[@]}")
fi

# ── Summary ─────────────────────────────────────────────────
TOTAL=$((${#EXP_ARRAY[@]} * ${#SIZE_ARRAY[@]}))
echo "============================================"
echo "Target     : Kaggle"
echo "User       : $KAGGLE_USER"
echo "GPU type   : $KAGGLE_GPU_TYPE"
echo "Git ref    : $GIT_REF"
echo "Git repo   : $GIT_REPO"
echo "Sizes      : ${SIZE_ARRAY[*]}"
echo "Experiments: ${#EXP_ARRAY[@]}"
echo "Total jobs : $TOTAL"
echo "Dry run    : ${DRY_RUN:-no}"
echo "============================================"

mkdir -p "$SCRIPT_DIR/.staging"
MANIFEST="$SCRIPT_DIR/manifest.jsonl"

# ── Submit ──────────────────────────────────────────────────
for SIZE in "${SIZE_ARRAY[@]}"; do
    for EXP in "${EXP_ARRAY[@]}"; do
        STAGING_DIR="$SCRIPT_DIR/.staging/${EXP}__${SIZE}__${GIT_REF:0:8}"
        rm -rf "$STAGING_DIR"

        # Render. render_kernel.py prints a JSON summary to stdout.
        SUMMARY="$(python "$SCRIPT_DIR/render_kernel.py" \
            --experiment "$EXP" \
            --size "$SIZE" \
            --git-ref "$GIT_REF" \
            --git-repo "$GIT_REPO" \
            --user "$KAGGLE_USER" \
            --gpu-type "$KAGGLE_GPU_TYPE" \
            --out "$STAGING_DIR")"

        SLUG="$(echo "$SUMMARY" | python -c 'import json,sys; print(json.load(sys.stdin)["slug"])')"
        KERNEL_ID="$(echo "$SUMMARY" | python -c 'import json,sys; print(json.load(sys.stdin)["kernel_id"])')"

        PUSH_CMD=(kaggle kernels push -p "$STAGING_DIR" --accelerator "$KAGGLE_GPU_TYPE")

        if [ "${DRY_RUN:-0}" = "1" ]; then
            echo "[DRY RUN] ${PUSH_CMD[*]}"
            echo "[DRY RUN] kernel_id=$KERNEL_ID"
            continue
        fi

        echo "Submitting: $KERNEL_ID"
        "${PUSH_CMD[@]}"

        SUBMITTED_AT="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
        python -c "
import json, sys
print(json.dumps({
    'slug': '$SLUG',
    'kernel_id': '$KERNEL_ID',
    'experiment': '$EXP',
    'size': '$SIZE',
    'gpu_type': '$KAGGLE_GPU_TYPE',
    'git_ref': '$GIT_REF',
    'submitted_at': '$SUBMITTED_AT',
    'status': 'pushed',
}))
" >> "$MANIFEST"
        sleep 0.5
    done
done

echo "────────────────────────────────────────────"
echo "Done. Pull results with:  python kaggle/fetch_results.py"
echo "Manifest: $MANIFEST"
