import sys
import os
import json

sys.path.append("../")
sys.path.append(".")

import streamlit as st
import pandas as pd
from pathlib import Path

from analysis.common import LOGS_ROOT

st.set_page_config(page_title="Experiment Status", layout="wide")
st.title("Experiment Status")


@st.cache_data
def load_status() -> pd.DataFrame:
    records = []
    logs_path = Path(LOGS_ROOT)

    if not logs_path.exists():
        return pd.DataFrame()

    for gs_path in logs_path.rglob("game_state.json"):
        parts = gs_path.relative_to(logs_path).parts
        section_raw = parts[0]

        try:
            if section_raw == "section_one" and len(parts) == 8:
                # section/game/condition/model_size/pair/setup/timestamp/file
                game_type = parts[1].replace("_section_one", "")
                condition = parts[2]
                model_size = parts[3]
                pair_tag = parts[4]
                setup_tag = parts[5]
            elif section_raw == "section_one" and len(parts) == 7:
                # section/game/condition/model_size/pair/timestamp/file (no setup_tag)
                game_type = parts[1].replace("_section_one", "")
                condition = parts[2]
                model_size = parts[3]
                pair_tag = parts[4]
                setup_tag = "-"
            elif section_raw == "section_two" and len(parts) >= 7:
                game_type = parts[1].replace("_section_two_personas", "")
                model_size = parts[2]
                pair_tag = parts[3]
                setup_tag = parts[4]
                condition = next(
                    (b for b in ["desperate", "cunning"] if setup_tag.endswith(f"_{b}")),
                    "default",
                )
            else:
                continue

            with open(gs_path) as f:
                data = json.load(f)
            completed = data["game_state"][-1].get("current_iteration") == "END"

            records.append(
                {
                    "Section": section_raw,
                    "Game": game_type,
                    "Condition": condition,
                    "Model Size": model_size,
                    "Pair": pair_tag,
                    "Setup": setup_tag,
                    "Completed": completed,
                }
            )
        except Exception:
            continue

    return pd.DataFrame(records)


with st.spinner("Scanning experiment logs..."):
    df = load_status()

if df.empty:
    st.info(f"No logs found in {LOGS_ROOT}.")
    st.stop()

# --- Filters ---
sections = ["All"] + sorted(df["Section"].unique().tolist())
selected_section = st.sidebar.selectbox(
    "Section",
    sections,
    format_func=lambda x: x.replace("_", " ").title() if x != "All" else "All",
)
only_incomplete = st.sidebar.checkbox("Show only incomplete combinations")

filtered = df if selected_section == "All" else df[df["Section"] == selected_section]

# --- Aggregate ---
grouped = (
    filtered.groupby(["Section", "Game", "Condition", "Model Size", "Pair", "Setup"])
    .agg(Played=("Completed", "count"), Completed=("Completed", "sum"))
    .reset_index()
)
grouped["% Done"] = (grouped["Completed"] / grouped["Played"] * 100).round(1)
grouped = grouped.sort_values("% Done")

if only_incomplete:
    grouped = grouped[grouped["Completed"] < grouped["Played"]]

# --- Top metrics ---
total_played = int(filtered.shape[0])
total_completed = int(filtered["Completed"].sum())
pct_done = round(total_completed / total_played * 100, 1) if total_played else 0
incomplete_combos = int((grouped["Completed"] < grouped["Played"]).sum())

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Played", f"{total_played:,}")
col2.metric("Total Completed", f"{total_completed:,}")
col3.metric("Overall % Done", f"{pct_done}%")
col4.metric("Incomplete Combinations", f"{incomplete_combos:,}")

st.markdown("---")

st.dataframe(grouped, use_container_width=True, hide_index=True)
