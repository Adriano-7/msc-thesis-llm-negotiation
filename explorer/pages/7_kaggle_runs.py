import sys
import os
import csv
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

sys.path.append("../")
sys.path.append(".")

import streamlit as st
import pandas as pd

REPO_DIR = Path(__file__).parent.parent.parent
ACCOUNTS_DIR = REPO_DIR / "kaggle" / "accounts"
DOWNLOADED_FILE = REPO_DIR / "kaggle" / ".downloaded.json"

# ── State helpers ────────────────────────────────────────────────────

def _write_state(state: dict) -> None:
    DOWNLOADED_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(DOWNLOADED_FILE, "w") as f:
        json.dump(state, f, indent=2, sort_keys=True)


def load_downloaded_state() -> dict:
    state: dict = {}
    if DOWNLOADED_FILE.exists():
        with open(DOWNLOADED_FILE) as f:
            state = json.load(f)

    # Auto-seed from merged kaggle-results PRs.
    # Branch: kaggle-results/buysell_section_two_personas-very_small-931d3544-20260504-23
    # Slug:   buysell-section-two-personas-very-small-931d3544
    try:
        result = subprocess.run(
            ["git", "-C", str(REPO_DIR), "log", "--merges", "--oneline"],
            capture_output=True, text=True, timeout=5,
        )
        new_entries = False
        for line in result.stdout.splitlines():
            m = re.search(r"kaggle-results/([^ ]+)", line)
            if m:
                slug = re.sub(r"-\d{8}-\d{2}$", "", m.group(1)).rstrip("'\"").replace("_", "-")
                if slug not in state:
                    state[slug] = "auto-detected"
                    new_entries = True
        if new_entries:
            _write_state(state)
    except Exception:
        pass

    return state


def load_account_profiles() -> list[dict]:
    profiles = []
    if not ACCOUNTS_DIR.exists():
        return profiles
    for env_file in sorted(ACCOUNTS_DIR.glob("*.env")):
        profile: dict = {"_account": env_file.stem}
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if "=" in line and not line.startswith("#"):
                    k, _, v = line.partition("=")
                    profile[k.strip()] = v.strip()
        profiles.append(profile)
    return profiles


# ── Kaggle CLI wrappers ──────────────────────────────────────────────

@st.cache_data(ttl=120)
def fetch_kernel_list(account_name: str, page_size: int, _username: str, _api_token: str) -> list[dict]:
    env = {**os.environ, "KAGGLE_USERNAME": _username, "KAGGLE_API_TOKEN": _api_token}
    try:
        result = subprocess.run(
            ["kaggle", "kernels", "list", "-m", "--sort-by", "dateRun",
             "--page-size", str(page_size), "-v"],
            capture_output=True, text=True, env=env, timeout=30,
        )
        rows = []
        skipped_header = False
        for row in csv.reader(result.stdout.splitlines()):
            if not skipped_header:
                skipped_header = True
                continue
            if row:
                rows.append({
                    "kernel_id": row[0],
                    "run_time": row[3][:16] if len(row) > 3 else "",
                })
        return rows
    except Exception:
        return []


@st.cache_data(ttl=120)
def fetch_kernel_status(kernel_id: str, _username: str, _api_token: str) -> str:
    env = {**os.environ, "KAGGLE_USERNAME": _username, "KAGGLE_API_TOKEN": _api_token}
    try:
        result = subprocess.run(
            ["kaggle", "kernels", "status", kernel_id],
            capture_output=True, text=True, env=env, timeout=15,
        )
        last_word = result.stdout.strip().split()[-1].strip('"')
        return last_word.replace("KernelWorkerStatus.", "")
    except Exception:
        return "UNKNOWN"


# ── Page ─────────────────────────────────────────────────────────────

st.set_page_config(page_title="Kaggle Runs", layout="wide")
st.title("Kaggle Runs")

col_size, col_btn, _ = st.columns([2, 1, 6], vertical_alignment="bottom")
with col_size:
    page_size = st.number_input("Kernels per account", min_value=1, max_value=50, value=5, step=1)
with col_btn:
    if st.button("⟳ Refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

st.markdown("---")

downloaded_state = load_downloaded_state()
profiles = load_account_profiles()

if not profiles:
    st.warning(f"No account profiles found in `{ACCOUNTS_DIR}`.")
    st.stop()

for profile in profiles:
    account_name = profile.get("_account", "unknown")
    username = profile.get("KAGGLE_USERNAME", "")
    api_token = profile.get("KAGGLE_API_TOKEN", "")
    email = profile.get("KAGGLE_EMAIL", "")
    gpu_limit = profile.get("KAGGLE_GPU_LIMIT_REACHED", "false").lower() == "true"

    gpu_badge = "🔴 GPU LIMIT" if gpu_limit else "🟢 GPU OK"

    with st.expander(f"{account_name}  ·  {email}  ·  {gpu_badge}", expanded=gpu_limit):
        kernels = fetch_kernel_list(account_name, int(page_size), username, api_token)

        if not kernels:
            st.info("No kernels found.")
            continue

        rows = []
        for k in kernels:
            kernel_id = k["kernel_id"]
            slug = kernel_id.split("/", 1)[-1]
            status = fetch_kernel_status(kernel_id, username, api_token)
            rows.append({
                "Downloaded": bool(downloaded_state.get(slug)),
                "Slug": slug,
                "Run Time": k["run_time"],
                "Status": status,
            })

        df = pd.DataFrame(rows)

        edited = st.data_editor(
            df,
            key=f"editor_{account_name}",
            column_config={
                "Downloaded": st.column_config.CheckboxColumn(width="small"),
                "Slug": st.column_config.TextColumn(width="large"),
                "Run Time": st.column_config.TextColumn(width="small"),
                "Status": st.column_config.TextColumn(width="small"),
            },
            disabled=["Slug", "Run Time", "Status"],
            use_container_width=True,
            hide_index=True,
        )

        # Persist any checkbox toggles immediately to the state file
        changed_mask = edited["Downloaded"] != df["Downloaded"]
        if changed_mask.any():
            for _, row in edited[changed_mask].iterrows():
                if row["Downloaded"]:
                    downloaded_state[row["Slug"]] = datetime.now(timezone.utc).isoformat()
                else:
                    downloaded_state.pop(row["Slug"], None)
            _write_state(downloaded_state)
