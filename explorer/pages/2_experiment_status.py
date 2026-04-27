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
            elif section_raw == "self_refine" and len(parts) >= 7:
                # self_refine/{game}_self_refine_v1/{size}/{pair}/{setup}/timestamp/file
                game_type = parts[1].replace("_self_refine_v1", "")
                model_size = parts[2]
                pair_tag = parts[3]
                setup_tag = parts[4]
                if setup_tag.endswith("_self_refineP1_self_refineP2"):
                    condition = "self_refine"
                elif setup_tag.endswith("_defaultP1_defaultP2"):
                    condition = "baseline"
                else:
                    continue
            else:
                continue

            if section_raw == "section_one":
                retry = "retry" if condition == "retry3" else "no_retry"
            else:
                retry = "no_retry"

            with open(gs_path) as f:
                data = json.load(f)
            last_state = data["game_state"][-1]
            completed = last_state.get("current_iteration") == "END"
            is_error = last_state.get("current_iteration") == "ERROR"
            error_type = last_state.get("error_type", "") if is_error else ""
            error_message = last_state.get("error_message", "") if is_error else ""

            records.append(
                {
                    "Section": section_raw,
                    "Game": game_type,
                    "Condition": condition,
                    "Retry": retry,
                    "Model Size": model_size,
                    "Pair": pair_tag,
                    "Setup": setup_tag,
                    "Completed": completed,
                    "Error Type": error_type,
                    "Error Message": error_message,
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

st.subheader("Grouped Status by Model Size")

per_cell = (
    filtered.groupby(["Model Size", "Game", "Section", "Condition", "Retry", "Pair"])
    .agg(Played=("Completed", "count"), Completed=("Completed", "sum"))
    .reset_index()
)
per_cell["Cell"] = per_cell["Completed"].astype(str) + "/" + per_cell["Played"].astype(str)

SECTION_MAP = {
    ("section_one", "no_retries"):  ("S1-Base",     "-"),
    ("section_one", "retry3"):      ("S1-Base",     "-"),
    ("section_two", "default"):     ("S2-Personas", "default"),
    ("section_two", "desperate"):   ("S2-Personas", "desperate"),
    ("section_two", "cunning"):     ("S2-Personas", "cunning"),
    ("self_refine", "baseline"):    ("Self-Refine", "baseline"),
    ("self_refine", "self_refine"): ("Self-Refine", "self_refine"),
}

CANONICAL_SIZES = ["very_small", "small", "medium"]
GAME_ORDER = ["buysell", "trading", "ultimatum"]

GROUPED_TABLE_CSS = """
<style>
.grouped-status-table { border-collapse: collapse; font-size: 0.85rem; margin-bottom: 1rem; }
.grouped-status-table th, .grouped-status-table td {
    border: 1px solid #e0e0e0; padding: 4px 10px; text-align: center; white-space: nowrap;
}
.grouped-status-table thead th { background: #f5f5f5; font-weight: 600; }
.grouped-status-table thead tr:first-child th { background: #e8eef6; font-size: 0.95rem; }
.grouped-status-table tbody th { text-align: left; font-weight: 500; background: #fafafa; }
.grouped-status-table tbody td:empty::before { content: "-"; color: #bbb; }
</style>
"""
st.markdown(GROUPED_TABLE_CSS, unsafe_allow_html=True)

for size in CANONICAL_SIZES:
    sub = per_cell[per_cell["Model Size"] == size]
    if sub.empty:
        continue
    st.markdown(f"### {size.replace('_', ' ').title()}")

    sub = sub.copy()
    labels = sub.apply(
        lambda r: SECTION_MAP.get((r["Section"], r["Condition"]), (r["Section"], r["Condition"])),
        axis=1,
    )
    sub["SectionLabel"] = [x[0] for x in labels]
    sub["SubLabel"] = [x[1] for x in labels]

    pivoted = sub.pivot_table(
        index="Pair",
        columns=["Game", "SectionLabel", "SubLabel", "Retry"],
        values="Cell",
        aggfunc="first",
    )
    pivoted.columns.names = ["Game", "Section", "Sub", "Retry"]

    present_games = [g for g in GAME_ORDER if g in pivoted.columns.get_level_values(0)]
    if present_games:
        pivoted = pivoted.reindex(columns=present_games, level=0)

    pivoted = pivoted.fillna("-")

    html = pivoted.to_html(classes="grouped-status-table", border=0, escape=False)
    st.markdown(html, unsafe_allow_html=True)

st.markdown("---")
st.subheader("Flat Status Table")
st.dataframe(grouped, use_container_width=True, hide_index=True)

# --- Error Breakdown (incomplete games only) ---
incomplete_df = filtered[~filtered["Completed"]]
if not incomplete_df.empty:
    st.markdown("---")
    st.subheader("Error Breakdown (Incomplete Games)")

    with_error = incomplete_df[incomplete_df["Error Type"] != ""]
    without_error = incomplete_df[incomplete_df["Error Type"] == ""]

    ec1, ec2 = st.columns(2)
    ec1.metric("With Captured Error", len(with_error))
    ec2.metric("Unknown (pre-fix runs)", len(without_error))

    # Error type counts grouped by section / game / error type
    display_df = incomplete_df.copy()
    display_df["Error Type"] = display_df["Error Type"].replace("", "Unknown")
    error_counts = (
        display_df.groupby(["Section", "Game", "Error Type"])
        .size()
        .reset_index(name="Count")
        .sort_values("Count", ascending=False)
    )
    st.dataframe(error_counts, use_container_width=True, hide_index=True)

    with st.expander("Show individual error details"):
        detail = display_df[["Section", "Game", "Pair", "Setup", "Error Type", "Error Message"]].copy()
        detail["Error Message"] = detail["Error Message"].str[:120]
        st.dataframe(detail, use_container_width=True, hide_index=True)
