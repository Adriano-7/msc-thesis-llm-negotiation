#!/bin/bash
# ============================================================
# Check the status of submitted Kaggle kernels.
#
# Modes:
#   Default          — query status of every kernel in .staging/
#   --recent         — list the 20 most-recently-run kernels
#
# Flags:
#   --all-accounts   — repeat for every profile in kaggle/accounts/
#   --recent         — use `kaggle kernels list` instead of staging dirs
#   --page-size N    — number of kernels to show in --recent mode (default 20)
#
# Usage:
#   bash kaggle/status.sh
#   bash kaggle/status.sh --recent
#   KAGGLE_ACCOUNT=adrianomachado bash kaggle/status.sh
#   bash kaggle/status.sh --all-accounts
#   bash kaggle/status.sh --all-accounts --recent --page-size 10
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Parse flags ─────────────────────────────────────────────
MODE="staged"
ALL_ACCOUNTS=0
PAGE_SIZE=20

while [[ $# -gt 0 ]]; do
    case "$1" in
        --recent)       MODE="recent"; shift ;;
        --all-accounts) ALL_ACCOUNTS=1; shift ;;
        --page-size)    PAGE_SIZE="$2"; shift 2 ;;
        *) echo "Unknown flag: $1" >&2; exit 1 ;;
    esac
done

# ── Account loading ─────────────────────────────────────────
load_account() {
    local account="${1:-}"

    # Load base .env first
    if [ -f "$REPO_DIR/.env" ]; then
        set -a
        # shellcheck disable=SC1091
        source "$REPO_DIR/.env"
        set +a
    fi

    # Override with per-account profile if requested
    if [ -n "$account" ]; then
        local profile="$SCRIPT_DIR/accounts/${account}.env"
        if [ ! -f "$profile" ]; then
            echo "ERROR: KAGGLE_ACCOUNT='$account' but $profile not found." >&2
            exit 1
        fi
        set -a
        # shellcheck disable=SC1090
        source "$profile"
        set +a
        KAGGLE_USER="$KAGGLE_USERNAME"
    fi

    # Resolve username
    KAGGLE_USER="${KAGGLE_USER:-${KAGGLE_USERNAME:-}}"
    if [ -z "${KAGGLE_USER:-}" ] && command -v kaggle >/dev/null 2>&1; then
        KAGGLE_USER="$(kaggle config view 2>/dev/null | awk -F': ' '/^- username:/ {print $2; exit}')"
    fi
    if [ -z "${KAGGLE_USER:-}" ]; then
        echo "ERROR: set KAGGLE_USERNAME (and KAGGLE_KEY) in .env, or KAGGLE_USER inline." >&2
        exit 1
    fi
}

# ── Status for one account ───────────────────────────────────
check_account() {
    local account="${1:-}"
    load_account "$account"

    local label="${account:-(default / $KAGGLE_USER)}"
    echo ""
    echo "Account: $label"
    printf '%.0s─' {1..64}; echo ""

    if [ "$MODE" = "recent" ]; then
        # Use CSV output for reliable column parsing (ref=col1, lastRunTime=col4)
        local entries
        entries="$(kaggle kernels list -m --sort-by dateRun --page-size "$PAGE_SIZE" -v \
            | python3 -c "
import csv, sys
for row in list(csv.reader(sys.stdin))[1:]:
    if row: print(row[0] + '|' + row[3][:16])
")"
        if [ -z "$entries" ]; then
            echo "  No kernels found."
            return
        fi
        local had_error=0
        while IFS='|' read -r kernel_id run_time; do
            local slug="${kernel_id##*/}"
            local status
            status="$(kaggle kernels status "$kernel_id" 2>&1 \
                | awk '{print $NF}' | tr -d '"' | sed 's/KernelWorkerStatus\.//')"
            printf "  %-52s  %-16s  → %s\n" "$slug" "$run_time" "$status"
            [ "$status" = "ERROR" ] && had_error=1
        done <<< "$entries"
        return $had_error
    fi

    # Staged-kernel mode: check only kernels owned by this user
    local found=0
    local had_error=0
    shopt -s nullglob
    for meta in "$SCRIPT_DIR/.staging/"*/kernel-metadata.json; do
        local kernel_id
        kernel_id="$(python3 -c "import json,sys; d=json.load(open('$meta')); print(d['id'])")"

        # Only check kernels that belong to this account
        local owner="${kernel_id%%/*}"
        if [ "$owner" != "$KAGGLE_USER" ]; then
            continue
        fi

        found=1
        local slug="${kernel_id##*/}"
        local raw_status
        raw_status="$(kaggle kernels status "$kernel_id" 2>&1)"
        # Output is like: "owner/slug has status "KernelWorkerStatus.COMPLETE""
        local status
        status="$(echo "$raw_status" | awk '{print $NF}' | tr -d '"' | sed 's/KernelWorkerStatus\.//')"

        printf "  %-55s → %s\n" "$slug" "$status"
        if [ "$status" = "error" ]; then
            had_error=1
        fi
    done
    shopt -u nullglob

    if [ "$found" = "0" ]; then
        echo "  No staged kernels found for user '$KAGGLE_USER'."
        echo "  Run 'bash kaggle/status.sh --recent' to list all your kernels."
    fi

    return $had_error
}

# ── Main ────────────────────────────────────────────────────
overall_error=0

if [ "$ALL_ACCOUNTS" = "1" ]; then
    for profile in "$SCRIPT_DIR/accounts/"*.env; do
        account="$(basename "$profile" .env)"
        check_account "$account" || overall_error=1
    done
else
    check_account "${KAGGLE_ACCOUNT:-}" || overall_error=1
fi

echo ""
exit $overall_error
