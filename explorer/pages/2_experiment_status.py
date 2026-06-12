import sys
import os
import json
import re as _re

sys.path.append("../")
sys.path.append(".")

import streamlit as st
import pandas as pd
from pathlib import Path

from analysis.common import LOGS_ROOT, clean_name

st.set_page_config(page_title="Experiment Status", layout="wide")
st.title("Experiment Status")


_STRAT_SUFFIX_RE = __import__("re").compile(r"_([a-z_]+)P1_([a-z_]+)P2$")

_STRAT_TO_COND = {
    ("default",     "default"):     "base × base",
    ("self_refine", "self_refine"): "refine × refine",
    ("default",     "self_refine"): "base × refine",
    ("self_refine", "default"):     "refine × base",
}

_SR_GAMES = {
    "trading_self_refine_v1":          "trading",
    "buysell_self_refine_v1":          "buysell",
    "ultimatum_self_refine_v1":        "ultimatum",
    "trading_self_refine_v1_retry3":   "trading",
    "buysell_self_refine_v1_retry3":   "buysell",
    "ultimatum_self_refine_v1_retry3": "ultimatum",
}


def _sr_condition_dirs() -> list[tuple]:
    """
    Yield (game, size, pair_tag, condition, dir_path, is_retry) for every
    self-refine condition directory that exists on disk — including empty ones.
    """
    results = []
    sr_root = Path(LOGS_ROOT) / "self_refine"
    if not sr_root.exists():
        return results
    for exp_dir in sr_root.iterdir():
        game = _SR_GAMES.get(exp_dir.name)
        if game is None or not exp_dir.is_dir():
            continue
        is_retry = exp_dir.name.endswith("_retry3")
        for size_dir in exp_dir.iterdir():
            if not size_dir.is_dir():
                continue
            size = size_dir.name
            for sub in size_dir.iterdir():
                if not sub.is_dir():
                    continue
                # Trading/Ultimatum: sub = "{pair}_{stratP1}_{stratP2}"
                m = _STRAT_SUFFIX_RE.search(sub.name)
                if m:
                    cond = _STRAT_TO_COND.get((m.group(1), m.group(2)))
                    if cond:
                        pair = sub.name[: m.start()]
                        results.append((game, size, pair, cond, sub, is_retry))
                else:
                    # BuySell: sub = "{pair}", look one level deeper
                    pair = sub.name
                    for setup_dir in sub.iterdir():
                        if not setup_dir.is_dir():
                            continue
                        m2 = _STRAT_SUFFIX_RE.search(setup_dir.name)
                        if m2:
                            cond = _STRAT_TO_COND.get((m2.group(1), m2.group(2)))
                            if cond:
                                results.append((game, size, pair, cond, setup_dir, is_retry))
    return results


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
            is_retry = False
            if section_raw == "section_one" and len(parts) == 8:
                game_type = parts[1].replace("_section_one", "")
                condition = parts[2]
                model_size = parts[3]
                pair_tag = parts[4]
                setup_tag = parts[5]
            elif section_raw == "section_one" and len(parts) == 7:
                game_type = parts[1].replace("_section_one", "")
                condition = parts[2]
                model_size = parts[3]
                pair_tag = parts[4]
                setup_tag = "-"
            elif section_raw == "section_two" and len(parts) >= 7:
                exp_name = parts[1]
                is_retry = exp_name.endswith("_retry3")
                game_type = exp_name.replace("_retry3", "").replace("_section_two_personas", "")
                model_size = parts[2]
                pair_tag = parts[3]
                setup_tag = parts[4]
                condition = next(
                    (b for b in ["desperate", "cunning"] if setup_tag.endswith(f"_{b}")),
                    "default",
                )
            elif section_raw == "section_two" and len(parts) == 6:
                exp_name = parts[1]
                is_retry = exp_name.endswith("_retry3")
                game_type = exp_name.replace("_retry3", "").replace("_section_two_personas", "")
                model_size = parts[2]
                raw_pair = parts[3]
                setup_tag = "-"
                condition = next(
                    (b for b in ["desperate", "cunning", "default"] if raw_pair.endswith(f"_{b}")),
                    "default",
                )
                pair_tag = raw_pair
                for suffix in ["_desperate", "_cunning", "_default"]:
                    if pair_tag.endswith(suffix):
                        pair_tag = pair_tag[: -len(suffix)]
                        break
            elif section_raw == "self_refine":
                # Parse strategy suffix from whichever path level carries it.
                exp_name = parts[1]
                is_retry = exp_name.endswith("_retry3")
                game_type = exp_name.replace("_self_refine_v1_retry3", "").replace("_self_refine_v1", "")
                model_size = parts[2]
                p1 = p2 = None
                for piece in reversed(parts[3:-1]):
                    m = _STRAT_SUFFIX_RE.search(piece)
                    if m:
                        p1, p2 = m.group(1), m.group(2)
                        break
                condition = _STRAT_TO_COND.get((p1, p2)) if p1 else None
                if condition is None:
                    continue
                # Pair tag: buysell has an extra dir level, others embed it in dir name.
                if len(parts) >= 8:                      # buysell: …/pair/setup/ts/file
                    pair_tag = parts[3]
                    setup_tag = parts[4]
                else:                                    # trading/ultimatum: …/pair+strat/ts/file
                    raw = parts[3]
                    m2 = _STRAT_SUFFIX_RE.search(raw)
                    pair_tag = raw[: m2.start()] if m2 else raw
                    setup_tag = "-"
            elif section_raw == "negotiation_team" and len(parts) in (6, 7):
                exp_name = parts[1]
                is_p2 = exp_name.endswith("_p2")
                core = exp_name[: -len("_p2")] if is_p2 else exp_name
                game_type = core.split("_negotiation_team")[0]
                diversity = "hetero" if "hetero" in core else "homo"
                condition = f"{diversity}-{'p2' if is_p2 else 'p1'}"
                model_size = parts[2]
                pair_tag = parts[3]
                # buysell carries an extra setup level (…/pair/setup/ts/file)
                setup_tag = parts[4] if len(parts) == 7 else "-"
            else:
                continue

            if section_raw == "section_one":
                retry = "retry" if condition == "retry3" else "no_retry"
            elif section_raw in ("section_two", "self_refine"):
                retry = "retry" if is_retry else "no_retry"
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

    df = pd.DataFrame(records)

    # Inject placeholder rows (Played=0) for self-refine condition directories
    # that exist on disk but have no game_state.json files inside.
    existing_keys = set()
    if not df.empty:
        sr = df[df["Section"] == "self_refine"]
        for _, row in sr.iterrows():
            existing_keys.add((row["Game"], row["Model Size"], row["Pair"], row["Condition"], row["Retry"]))

    placeholders = []
    for game, size, pair, cond, _, is_retry in _sr_condition_dirs():
        retry_val = "retry" if is_retry else "no_retry"
        if (game, size, pair, cond, retry_val) not in existing_keys:
            placeholders.append({
                "Section": "self_refine",
                "Game": game,
                "Condition": cond,
                "Retry": retry_val,
                "Model Size": size,
                "Pair": pair,
                "Setup": "-",
                "Completed": False,
                "Error Type": "",
                "Error Message": "",
                "_planned_empty": True,
            })

    if placeholders:
        ph_df = pd.DataFrame(placeholders)
        df = pd.concat([df, ph_df], ignore_index=True)

    if "_planned_empty" not in df.columns:
        df["_planned_empty"] = False
    df["_planned_empty"] = df["_planned_empty"].fillna(False)

    return df


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

filtered     = df if selected_section == "All" else df[df["Section"] == selected_section]
# Real runs only (exclude planned-empty placeholders) for metrics and flat table
real_runs    = filtered[~filtered["_planned_empty"]]

# --- Aggregate (real runs only) ---
grouped = (
    real_runs.groupby(["Section", "Game", "Condition", "Model Size", "Pair", "Setup"])
    .agg(Played=("Completed", "count"), Completed=("Completed", "sum"))
    .reset_index()
)
grouped["% Done"] = (grouped["Completed"] / grouped["Played"] * 100).round(1)
grouped = grouped.sort_values("% Done")

if only_incomplete:
    grouped = grouped[grouped["Completed"] < grouped["Played"]]

# --- Top metrics ---
total_played    = int(real_runs.shape[0])
total_completed = int(real_runs["Completed"].sum())
pct_done        = round(total_completed / total_played * 100, 1) if total_played else 0
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
    ("section_one", "no_retries"):      ("S1-Base",     "-"),
    ("section_one", "retry3"):          ("S1-Base",     "-"),
    ("section_two", "default"):         ("S2-Personas", "default"),
    ("section_two", "desperate"):       ("S2-Personas", "desperate"),
    ("section_two", "cunning"):         ("S2-Personas", "cunning"),
    ("self_refine", "base × base"):     ("Self-Refine", "base × base"),
    ("self_refine", "refine × refine"): ("Self-Refine", "refine × refine"),
    ("self_refine", "base × refine"):   ("Self-Refine", "base × refine"),
    ("self_refine", "refine × base"):   ("Self-Refine", "refine × base"),
    ("negotiation_team", "homo-p1"):    ("Team", "homo P1"),
    ("negotiation_team", "hetero-p1"):  ("Team", "hetero P1"),
    ("negotiation_team", "homo-p2"):    ("Team", "homo P2"),
    ("negotiation_team", "hetero-p2"):  ("Team", "hetero P2"),
}

CANONICAL_SIZES = ["very_small", "small", "medium"]
GAME_ORDER = ["buysell", "trading", "ultimatum"]

def _is_self_play(pair: str) -> bool:
    parts = pair.split("_vs_", 1)
    return len(parts) == 2 and parts[0].strip() == parts[1].strip()


def _mark_self_play_rows(html: str, self_play_pairs: set) -> str:
    """Add class="self-play" to <tr> elements whose row-header matches a self-play pair."""
    return _re.sub(
        r"<tr>(\s*<th[^>]*>)([^<]*)(</th>)",
        lambda m: (
            f'<tr class="self-play">{m.group(1)}{m.group(2)}{m.group(3)}'
            if m.group(2) in self_play_pairs
            else m.group(0)
        ),
        html,
    )


def _self_play_row_style(row: pd.Series) -> list[str]:
    color = "background-color: #eef2f7" if _is_self_play(str(row.get("Pair", ""))) else ""
    return [color] * len(row)


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
.grouped-status-table tbody tr.self-play th,
.grouped-status-table tbody tr.self-play td { background-color: #eef2f7; }
</style>
"""
st.markdown(GROUPED_TABLE_CSS, unsafe_allow_html=True)

# Team negotiation gets its own table below (team-vs-single pairs don't fit the
# self/cross-play grid), so keep it out of the main grouped tables.
main_cells = per_cell[per_cell["Section"] != "negotiation_team"]
team_cells = per_cell[per_cell["Section"] == "negotiation_team"]

for size in CANONICAL_SIZES:
    sub = main_cells[main_cells["Model Size"] == size]
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

    self_play_pairs = {p for p in pivoted.index if _is_self_play(p)}
    html = pivoted.to_html(classes="grouped-status-table", border=0, escape=False)
    html = _mark_self_play_rows(html, self_play_pairs)
    st.markdown(html, unsafe_allow_html=True)

# --- Team Negotiation (separate table) ---
if not team_cells.empty:
    st.markdown("---")
    st.subheader("Team Negotiation Status")

    # The homo/hetero + P1/P2 condition is already implied by each pair, so
    # spreading it across columns leaves the grid almost empty. Instead group
    # rows by condition and keep one column per game.
    _TEAM_COND_ORDER = ["homo P1", "hetero P1", "homo P2", "hetero P2"]

    tc = team_cells.copy()
    labels = tc.apply(
        lambda r: SECTION_MAP.get((r["Section"], r["Condition"]), (r["Section"], r["Condition"])),
        axis=1,
    )
    tc["Team"] = [x[1] for x in labels]
    tc["Team"] = pd.Categorical(tc["Team"], categories=_TEAM_COND_ORDER, ordered=True)

    for size in CANONICAL_SIZES:
        sub = tc[tc["Model Size"] == size]
        if sub.empty:
            continue
        st.markdown(f"### {size.replace('_', ' ').title()}")

        pivoted = sub.pivot_table(
            index=["Team", "Pair"],
            columns="Game",
            values="Cell",
            aggfunc="first",
            observed=True,
        )

        present_games = [g for g in GAME_ORDER if g in pivoted.columns]
        if present_games:
            pivoted = pivoted.reindex(columns=present_games)

        pivoted = pivoted.fillna("-")
        html = pivoted.to_html(classes="grouped-status-table", border=0, escape=False)
        st.markdown(html, unsafe_allow_html=True)

st.markdown("---")
st.subheader("Flat Status Table")
st.dataframe(
    grouped.style.apply(_self_play_row_style, axis=1),
    use_container_width=True,
    hide_index=True,
)

# --- Error Breakdown (incomplete games only) ---
incomplete_df = real_runs[~real_runs["Completed"]]
if not incomplete_df.empty:
    st.markdown("---")
    st.subheader("Error Breakdown (Incomplete Games)")

    with_error = incomplete_df[incomplete_df["Error Type"] != ""]

    st.metric("With Captured Error", len(with_error))

    # Error type counts grouped by section / game / error type
    display_df = incomplete_df.copy()
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
