import sys
import os
import json
import warnings
from pathlib import Path

sys.path.append("../")
sys.path.append(".")

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from pandas.errors import SettingWithCopyWarning

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
warnings.filterwarnings("ignore")

sns.set_context("notebook")
sns.set_palette("colorblind")

from analysis.common import LOGS_ROOT, clean_name

st.set_page_config(page_title="Self-Refine Analysis", layout="wide")
st.title("Self-Refine Analysis")

SELF_REFINE_ROOT = Path(LOGS_ROOT) / "self_refine"
if not SELF_REFINE_ROOT.exists():
    st.error(f"No data in {SELF_REFINE_ROOT}.")
    st.stop()

GAMES = {
    "trading_self_refine_v1": "trading",
    "buysell_self_refine_v1": "buysell",
    "ultimatum_self_refine_v1": "ultimatum",
}
STRATEGIES = ["baseline", "self_check", "self_refine"]
STRATEGY_COLORS = {"baseline": "#7f7f7f", "self_check": "#1f77b4", "self_refine": "#d62728"}


def resource_total(res: dict) -> float:
    return sum(res["_value"].values())


def parse_run_path(game_state_path: Path) -> dict:
    parts = game_state_path.parts
    idx = parts.index("self_refine")
    experiment = parts[idx + 1]
    size = parts[idx + 2]
    strategy = None
    for piece in reversed(parts[idx + 3 : -1]):
        for s in STRATEGIES:
            if piece.endswith("_" + s) or piece == s:
                strategy = s
                break
        if strategy:
            break
    run_id = parts[-2]
    pair_raw = parts[idx + 3]
    pair = pair_raw
    for s in STRATEGIES:
        if pair.endswith("_" + s):
            pair = pair[: -(len(s) + 1)]
    return dict(experiment=experiment, size=size, pair=pair, strategy=strategy, run_id=run_id)


def extract_outcome(game_prefix: str, data: dict) -> dict:
    gs = data["game_state"]
    last = gs[-1]
    completed = last.get("current_iteration") == "END"
    turn_states = [s for s in gs if s.get("current_iteration") not in ("START", "END")]
    out = dict(
        game=game_prefix,
        completed=completed,
        num_turns=len(turn_states),
        total_parse_retries=data.get("total_parse_retries", 0),
        model_1=data["players"][0].get("model_id", data["players"][0].get("model")),
        model_2=data["players"][1].get("model_id", data["players"][1].get("model")),
    )
    if not completed:
        return out
    summary = last.get("summary", {})
    out["final_response"] = summary.get("final_response")
    out["deal"] = summary.get("final_response") == "ACCEPT"
    if game_prefix == "buysell":
        oc = summary.get("player_outcome", [None, None])
        out["outcome_1"] = oc[0]
        out["outcome_2"] = oc[1]
        out["joint_welfare"] = (oc[0] + oc[1]) if out["deal"] and oc[0] is not None else 0
    else:
        init = summary.get("initial_resources")
        final = summary.get("final_resources")
        if init and final:
            d1 = resource_total(final[0]) - resource_total(init[0])
            d2 = resource_total(final[1]) - resource_total(init[1])
            out["outcome_1"] = d1
            out["outcome_2"] = d2
            out["joint_welfare"] = d1 + d2
    return out


@st.cache_data(ttl=600)
def load_all_runs():
    rows = []
    for exp_dir, game_prefix in GAMES.items():
        for gs_path in (SELF_REFINE_ROOT / exp_dir).rglob("game_state.json"):
            try:
                meta = parse_run_path(gs_path)
                if meta["strategy"] is None:
                    continue
                with open(gs_path) as f:
                    data = json.load(f)
                out = extract_outcome(game_prefix, data)
                rows.append({**meta, **out, "run_dir": str(gs_path.parent)})
            except Exception:
                continue
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["model_short"] = df["model_1"].map(clean_name)
    return df


df = load_all_runs()
if df.empty:
    st.warning("No self-refine runs found.")
    st.stop()

st.sidebar.header("Filters")
sizes = sorted(df["size"].unique())
size = st.sidebar.selectbox("Model size", sizes, index=0)
df_size = df[df["size"] == size]

tab_overview, tab_outcomes, tab_loop = st.tabs(
    ["Overview", "Outcomes (baseline vs self-refine)", "Loop drilldown"]
)


with tab_overview:
    st.subheader(f"Runs available — {size}")
    counts = (
        df_size.groupby(["game", "model_short", "strategy"])
        .size()
        .unstack("strategy", fill_value=0)
    )
    st.dataframe(counts, use_container_width=True)

    cols = st.columns(3)
    cols[0].metric("Total runs", len(df_size))
    cols[1].metric("Models", df_size["model_short"].nunique())
    cols[2].metric("Strategies", df_size["strategy"].nunique())

    st.caption(
        "Self-refine discards the feedback/refine loop from agent state by design. "
        "Per-turn refine traces are written to `refine_trace_iter_{N}_turn_{T}.json` "
        "starting from runs executed after this logging was added — existing runs "
        "won't have them. The Loop drilldown tab shows traces when they exist."
    )


with tab_outcomes:
    st.subheader("Completion rate (game ended in agreement or ran full iterations)")
    comp = (
        df_size.groupby(["game", "model_short", "strategy"])["completed"]
        .mean()
        .reset_index()
    )
    game_cols = st.columns(len(comp["game"].unique()))
    for col, game in zip(game_cols, sorted(comp["game"].unique())):
        with col:
            sub = comp[comp["game"] == game]
            fig, ax = plt.subplots(figsize=(4.5, 2.8))
            sns.barplot(
                data=sub,
                x="model_short",
                y="completed",
                hue="strategy",
                hue_order=[s for s in STRATEGIES if s in sub["strategy"].unique()],
                palette=STRATEGY_COLORS,
                ax=ax,
            )
            ax.set_ylim(0, 1.05)
            ax.set_ylabel("Completion rate")
            ax.set_xlabel("")
            ax.set_title(f"{game}")
            ax.tick_params(axis="x", rotation=30, labelsize=8)
            ax.legend(fontsize=8)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    st.subheader("Payoff for player 1 (self-play, so symmetric in expectation)")
    payoff_df = df_size[df_size["completed"] & df_size["outcome_1"].notna()]
    ylabel_map = {
        "trading": "Resource delta (P1)",
        "buysell": "Profit (P1)",
        "ultimatum": "Dollars (P1)",
    }
    game_cols = st.columns(len(payoff_df["game"].unique()))
    for col, game in zip(game_cols, sorted(payoff_df["game"].unique())):
        with col:
            sub = payoff_df[payoff_df["game"] == game]
            fig, ax = plt.subplots(figsize=(4.5, 2.8))
            sns.barplot(
                data=sub,
                x="model_short",
                y="outcome_1",
                hue="strategy",
                hue_order=[s for s in STRATEGIES if s in sub["strategy"].unique()],
                palette=STRATEGY_COLORS,
                errorbar=("ci", 95),
                ax=ax,
            )
            ax.set_ylabel(ylabel_map.get(game, "Outcome"))
            ax.set_xlabel("")
            ax.set_title(f"{game}")
            ax.tick_params(axis="x", rotation=30, labelsize=8)
            ax.legend(fontsize=8)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    st.subheader("Turns to completion (lower = faster agreement)")
    turns_df = df_size[df_size["completed"]]
    game_cols = st.columns(len(turns_df["game"].unique()))
    for col, game in zip(game_cols, sorted(turns_df["game"].unique())):
        with col:
            sub = turns_df[turns_df["game"] == game]
            fig, ax = plt.subplots(figsize=(4.5, 2.8))
            sns.barplot(
                data=sub,
                x="model_short",
                y="num_turns",
                hue="strategy",
                hue_order=[s for s in STRATEGIES if s in sub["strategy"].unique()],
                palette=STRATEGY_COLORS,
                errorbar=("ci", 95),
                ax=ax,
            )
            ax.set_ylabel("Turns")
            ax.set_xlabel("")
            ax.set_title(f"{game}")
            ax.tick_params(axis="x", rotation=30, labelsize=8)
            ax.legend(fontsize=8)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    with st.expander("Mann-Whitney U: baseline vs self_refine (per game × model)"):
        from scipy.stats import mannwhitneyu

        test_rows = []
        pdf = df_size[df_size["completed"]].dropna(subset=["outcome_1"])
        for game in pdf["game"].unique():
            for m in pdf["model_short"].unique():
                b = pdf[(pdf["game"] == game) & (pdf["model_short"] == m) & (pdf["strategy"] == "baseline")]["outcome_1"].values
                s = pdf[(pdf["game"] == game) & (pdf["model_short"] == m) & (pdf["strategy"] == "self_refine")]["outcome_1"].values
                if len(b) >= 3 and len(s) >= 3:
                    stat, p = mannwhitneyu(b, s, alternative="two-sided")
                    test_rows.append({
                        "game": game,
                        "model": m,
                        "n_baseline": len(b),
                        "n_refine": len(s),
                        "median_baseline": float(np.median(b)),
                        "median_refine": float(np.median(s)),
                        "p_value": p,
                    })
        if test_rows:
            test_df = pd.DataFrame(test_rows)
            n_tests = len(test_df)
            test_df["bonferroni_p"] = (test_df["p_value"] * n_tests).clip(upper=1.0)
            test_df["significant_0.05"] = test_df["bonferroni_p"] < 0.05
            st.dataframe(test_df.round(4), use_container_width=True)
            st.caption(f"n={n_tests} tests; Bonferroni-corrected p < 0.05 = significant.")
        else:
            st.info("Not enough runs per cell (min 3 each) for Mann-Whitney U.")


with tab_loop:
    st.subheader("Loop drilldown")
    refine_df = df_size[df_size["strategy"] == "self_refine"].copy()

    def has_traces(run_dir):
        return any(Path(run_dir).glob("refine_trace_iter_*.json"))

    refine_df["has_trace"] = refine_df["run_dir"].map(has_traces)
    n_with = int(refine_df["has_trace"].sum())
    n_total = len(refine_df)
    st.caption(
        f"{n_with} / {n_total} self-refine runs have per-turn trace files. "
        f"Re-run experiments after the logging change to populate the rest."
    )

    traced = refine_df[refine_df["has_trace"]]
    if traced.empty:
        st.info("No refine trace files found yet. Run new self-refine experiments to populate them.")
        st.stop()

    game_pick = st.selectbox("Game", sorted(traced["game"].unique()))
    model_pick = st.selectbox(
        "Model",
        sorted(traced[traced["game"] == game_pick]["model_short"].unique()),
    )
    pool = traced[(traced["game"] == game_pick) & (traced["model_short"] == model_pick)]
    run_pick = st.selectbox(
        "Run",
        pool["run_dir"].tolist(),
        format_func=lambda p: Path(p).name,
    )

    run_dir = Path(run_pick)
    trace_files = sorted(
        run_dir.glob("refine_trace_iter_*.json"),
        key=lambda p: tuple(int(x) for x in p.stem.split("_")[3::2]),
    )
    st.write(f"{len(trace_files)} traced turns in this run.")

    for tf in trace_files:
        with open(tf) as f:
            tr = json.load(f)
        parts = tf.stem.split("_")
        iter_n = parts[3]
        turn_n = parts[5]
        with st.expander(f"Iter {iter_n}, turn {turn_n}  ({len(tr['iterations'])} refine steps)"):
            st.markdown("**Initial draft**")
            st.code(tr["initial_draft"], language="text")
            for i, step in enumerate(tr["iterations"], start=1):
                st.markdown(f"**Feedback {i}**")
                st.code(step["feedback"], language="text")
                st.markdown(f"**Refined {i}**")
                st.code(step["refined"], language="text")
            st.markdown("**Final (same as last refined)**")
            st.code(tr["final"], language="text")
