import sys
import os
import json
import re
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
from analysis.self_refine_process import load_trace_rows as _proc_rows, load_trace_summary as _proc_summary

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

STRATEGY_SUFFIX_RE = re.compile(r"_([a-z_]+)P1_([a-z_]+)P2$")

COND_ORDER = ["base × base", "refine × refine", "base × refine", "refine × base"]
COND_COLORS = {
    "base × base":     "#7f7f7f",
    "refine × refine": "#d62728",
    "base × refine":   "#1f77b4",
    "refine × base":   "#ff7f0e",
}
YLABEL_MAP = {"trading": "Resource delta", "buysell": "Profit", "ultimatum": "Dollars"}


def cond_label(p1: str, p2: str) -> str:
    s = {"default": "base", "self_refine": "refine"}
    return f"{s.get(p1, p1)} × {s.get(p2, p2)}"


def resource_total(res: dict) -> float:
    return sum(res["_value"].values())


def _synth_strategy(p1: str, p2: str) -> str:
    """Map a (p1, p2) strategy pair to a single label used by existing charts."""
    if p1 == p2 == "default":
        return "baseline"
    if p1 == p2:
        return p1
    return f"{p1}P1_{p2}P2"


def _strip_strategy_suffix(piece: str) -> tuple[str, str | None, str | None]:
    """Return (piece_without_suffix, p1_strategy, p2_strategy)."""
    m = STRATEGY_SUFFIX_RE.search(piece)
    if m:
        return piece[: m.start()], m.group(1), m.group(2)
    return piece, None, None


def parse_run_path(game_state_path: Path) -> dict:
    parts = game_state_path.parts
    idx = parts.index("self_refine")
    experiment = parts[idx + 1]
    size = parts[idx + 2]

    p1_strategy, p2_strategy = None, None
    for piece in reversed(parts[idx + 3 : -1]):
        _, p1, p2 = _strip_strategy_suffix(piece)
        if p1:
            p1_strategy, p2_strategy = p1, p2
            break

    strategy = _synth_strategy(p1_strategy, p2_strategy) if p1_strategy else None

    run_id = parts[-2]
    pair, _, _ = _strip_strategy_suffix(parts[idx + 3])
    return dict(
        experiment=experiment,
        size=size,
        pair=pair,
        strategy=strategy,
        p1_strategy=p1_strategy,
        p2_strategy=p2_strategy,
        run_id=run_id,
    )


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

df["cond"] = df.apply(
    lambda r: cond_label(r["p1_strategy"], r["p2_strategy"])
    if pd.notna(r["p1_strategy"]) else None,
    axis=1,
)

st.sidebar.header("Filters")
sizes = sorted(df["size"].unique())
size = st.sidebar.selectbox("Model size", sizes, index=0)
df_size = df[df["size"] == size]

tab_overview, tab_outcomes, tab_cross, tab_loop, tab_process = st.tabs(
    ["Overview", "Outcomes (symmetric)", "Cross-play", "Loop drilldown", "Process Analysis"]
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

    st.subheader("Payoff for player 1 (self-play)")
    payoff_df = df_size[df_size["completed"] & df_size["outcome_1"].notna()]
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
            ax.set_ylabel(YLABEL_MAP.get(game, "Outcome") + " (P1)")
            ax.set_xlabel("")
            ax.set_title(f"{game}")
            ax.tick_params(axis="x", rotation=30, labelsize=8)
            ax.legend(fontsize=8)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    st.subheader("Turns to completion")
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


with tab_cross:
    st.subheader("Cross-play: asymmetric strategy conditions")

    df_asym = df[df["p1_strategy"] != df["p2_strategy"]].dropna(subset=["p1_strategy", "p2_strategy"])
    if df_asym.empty:
        st.info("No cross-play data yet. Run experiments with different p1_strategy and p2_strategy per player.")
    else:
        # Use the sidebar size filter — for medium, all 4 conditions are present for all games
        df_cp = df_size.dropna(subset=["cond"]).copy()

        if df_cp.empty:
            st.info(f"No strategy data for size group **{size}**. Try **medium** which has all four conditions.")
        else:
            avail_conds = [c for c in COND_ORDER if c in df_cp["cond"].unique()]
            palette = {c: COND_COLORS[c] for c in avail_conds}

            games_here = sorted(df_cp["game"].unique())

            # ── Section 1: Per-player payoff across all conditions ───────────
            st.markdown("#### Per-player payoff by strategy condition")
            st.caption("Mean ± 95 % CI. *base* = default (one-shot), *refine* = self-refine loop.")

            completed_cp = df_cp[df_cp["completed"]].dropna(subset=["outcome_1"])

            if not completed_cp.empty:
                for game in games_here:
                    sub_g = completed_cp[completed_cp["game"] == game]
                    if sub_g.empty:
                        continue
                    ylabel = YLABEL_MAP.get(game, "Outcome")
                    st.markdown(f"**{game}**")
                    fig, axes = plt.subplots(1, 2, figsize=(11, 3.2), sharey=False)
                    for ax, (col, title) in zip(axes, [("outcome_1", "P1"), ("outcome_2", "P2")]):
                        sub = sub_g.dropna(subset=[col])
                        if sub.empty:
                            ax.set_visible(False)
                            continue
                        sns.barplot(
                            data=sub, x="model_short", y=col,
                            hue="cond", hue_order=avail_conds, palette=palette,
                            errorbar=("ci", 95), ax=ax,
                        )
                        ax.set_title(f"{title} payoff")
                        ax.set_ylabel(ylabel)
                        ax.set_xlabel("")
                        ax.tick_params(axis="x", rotation=30, labelsize=8)
                        ax.legend(fontsize=7, title="Condition")
                    fig.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)

            # ── Section 2: Δ vs base×base ────────────────────────────────────
            baseline_cond = "base × base"
            if baseline_cond in avail_conds:
                st.markdown("#### Advantage Δ vs base × base")
                st.caption(
                    "Δ = mean payoff in condition − mean payoff in *base × base*. "
                    "Positive Δ P1 under **refine × base** means P1 gains by refining when the opponent doesn't."
                )
                non_base = [c for c in avail_conds if c != baseline_cond]
                for game in games_here:
                    sub_g = completed_cp[completed_cp["game"] == game].dropna(subset=["outcome_1"])
                    if sub_g.empty:
                        continue
                    means = (
                        sub_g.groupby(["model_short", "cond"])[["outcome_1", "outcome_2"]]
                        .mean()
                        .reset_index()
                    )
                    base_m = (
                        means[means["cond"] == baseline_cond]
                        [["model_short", "outcome_1", "outcome_2"]]
                        .rename(columns={"outcome_1": "b1", "outcome_2": "b2"})
                    )
                    delta = means.merge(base_m, on="model_short", how="inner")
                    delta["Δ P1"] = delta["outcome_1"] - delta["b1"]
                    delta["Δ P2"] = delta["outcome_2"] - delta["b2"]
                    delta = delta[delta["cond"].isin(non_base)]
                    if delta.empty:
                        continue

                    ylabel = YLABEL_MAP.get(game, "Outcome")
                    st.markdown(f"**{game}**")
                    fig, axes = plt.subplots(1, 2, figsize=(11, 3.2), sharey=True)
                    non_base_palette = {c: COND_COLORS[c] for c in non_base}
                    for ax, col in zip(axes, ["Δ P1", "Δ P2"]):
                        sns.barplot(
                            data=delta, x="model_short", y=col,
                            hue="cond", hue_order=non_base, palette=non_base_palette,
                            ax=ax,
                        )
                        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
                        ax.set_title(col)
                        ax.set_ylabel(f"Δ {ylabel}")
                        ax.set_xlabel("")
                        ax.tick_params(axis="x", rotation=30, labelsize=8)
                        ax.legend(fontsize=7, title="Condition")
                    fig.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)
            else:
                st.info(
                    "No *base × base* data for this size group — can't compute Δ. "
                    "Switch to **medium** where all four conditions are available."
                )

            # ── Section 3: Joint welfare ─────────────────────────────────────
            st.markdown("#### Joint welfare by condition")
            welfare_cp = df_cp[df_cp["completed"]].dropna(subset=["joint_welfare"])
            if not welfare_cp.empty:
                fig, axes = plt.subplots(1, len(games_here), figsize=(5 * len(games_here), 3.2))
                if len(games_here) == 1:
                    axes = [axes]
                for ax, game in zip(axes, games_here):
                    sub = welfare_cp[welfare_cp["game"] == game]
                    if sub.empty:
                        ax.set_visible(False)
                        continue
                    sns.barplot(
                        data=sub, x="model_short", y="joint_welfare",
                        hue="cond", hue_order=avail_conds, palette=palette,
                        errorbar=("ci", 95), ax=ax,
                    )
                    ax.set_title(game)
                    ax.set_ylabel("Joint welfare")
                    ax.set_xlabel("")
                    ax.tick_params(axis="x", rotation=30, labelsize=8)
                    ax.legend(fontsize=7, title="Condition")
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

            # ── Section 4: Completion rate ───────────────────────────────────
            st.markdown("#### Completion rate by condition")
            comp_cp = df_cp.groupby(["game", "model_short", "cond"])["completed"].mean().reset_index()
            if not comp_cp.empty:
                fig, axes = plt.subplots(1, len(games_here), figsize=(5 * len(games_here), 3.2))
                if len(games_here) == 1:
                    axes = [axes]
                for ax, game in zip(axes, games_here):
                    sub = comp_cp[comp_cp["game"] == game]
                    sns.barplot(
                        data=sub, x="model_short", y="completed",
                        hue="cond", hue_order=avail_conds, palette=palette,
                        ax=ax,
                    )
                    ax.set_ylim(0, 1.05)
                    ax.set_title(game)
                    ax.set_ylabel("Completion rate")
                    ax.set_xlabel("")
                    ax.tick_params(axis="x", rotation=30, labelsize=8)
                    ax.legend(fontsize=7, title="Condition")
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

            # ── Section 5: Raw data ──────────────────────────────────────────
            with st.expander("Raw data"):
                show_cols = [c for c in [
                    "game", "model_short", "cond", "p1_strategy", "p2_strategy",
                    "size", "completed", "outcome_1", "outcome_2", "joint_welfare", "num_turns",
                ] if c in df_cp.columns]
                st.dataframe(df_cp[show_cols].round(2), use_container_width=True)


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


# ── Process Analysis tab ───────────────────────────────────────────────────
DIRECTION_COLORS = {
    "more_selfish":  "#d62728",
    "unchanged":     "#7f7f7f",
    "more_generous": "#1f77b4",
    "no_proposal":   "#bcbd22",
}
DIRECTION_LABELS = {
    "more_selfish":  "More selfish",
    "unchanged":     "Unchanged",
    "more_generous": "More generous",
    "no_proposal":   "No proposal (accept)",
}
OFFER_YLABELS = {
    "ultimatum": "Amount proposer keeps (out of 100)",
    "buysell":   "ZUP price in trade",
    "trading":   "Net resources gained by proposer",
}

with tab_process:
    st.subheader("What happens inside the refinement loop?")
    st.caption(
        "Analyses are limited to turns where the agent actually proposes a trade "
        "(i.e. ACCEPT turns are excluded from offer drift). "
        "Higher *ambition delta* = proposer kept more value for themselves after refinement."
    )

    df_rows_all = _proc_rows()
    df_sum_all  = _proc_summary()

    if df_rows_all.empty or df_sum_all.empty:
        st.info("No refine trace files found. Run self-refine experiments to populate them.")
    else:
        df_rows_s = df_rows_all[df_rows_all["size"] == size].copy()
        df_sum_s  = df_sum_all[df_sum_all["size"] == size].copy()

        # ── Overview metrics ─────────────────────────────────────────────
        proposal_rows = df_sum_s[df_sum_s["direction"] != "no_proposal"]
        n_traces      = len(df_sum_s)
        n_proposals   = len(proposal_rows)
        pct_proposals = n_proposals / n_traces * 100 if n_traces else 0

        direction_counts = proposal_rows["direction"].value_counts()
        pct_selfish  = direction_counts.get("more_selfish",  0) / n_proposals * 100 if n_proposals else 0
        pct_generous = direction_counts.get("more_generous", 0) / n_proposals * 100 if n_proposals else 0
        pct_unch     = direction_counts.get("unchanged",     0) / n_proposals * 100 if n_proposals else 0
        mean_delta   = proposal_rows["ambition_delta"].mean() if n_proposals else float("nan")

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Traces analysed",    n_traces)
        c2.metric("Proposal turns",     f"{pct_proposals:.0f}%")
        c3.metric("More selfish",       f"{pct_selfish:.0f}%")
        c4.metric("More generous",      f"{pct_generous:.0f}%")
        c5.metric("Mean ambition Δ",    f"{mean_delta:+.1f}" if not np.isnan(mean_delta) else "—")

        st.divider()

        # ── Section 1: Conservative vs Ambitious per game × model ────────
        st.markdown("### Conservative vs Ambitious: direction of offer drift")
        st.caption(
            "For each turn where the agent proposes a trade, does self-refine make "
            "the final offer *more selfish* (keeps more value) or *more generous*? "
            "Ultimatum: lower dollar offer = more selfish. "
            "BuySell (seller): higher price = more selfish; (buyer): lower price = more selfish. "
            "Trading: higher net resource gain = more selfish."
        )

        prop_s = proposal_rows.copy()
        if prop_s.empty:
            st.info("No proposal turns with parseable offers for this size.")
        else:
            dir_order = ["more_selfish", "unchanged", "more_generous"]
            avail_dir = [d for d in dir_order if d in prop_s["direction"].unique()]
            games_here = sorted(prop_s["game"].unique())

            fig, axes = plt.subplots(1, len(games_here), figsize=(5 * len(games_here), 3.6))
            if len(games_here) == 1:
                axes = [axes]

            for ax, game in zip(axes, games_here):
                sub = prop_s[prop_s["game"] == game]
                models = sorted(sub["model_short"].dropna().unique())
                bar_w = 0.6

                bottoms = np.zeros(len(models))
                for direction in dir_order:
                    vals = []
                    for m in models:
                        m_sub = sub[sub["model_short"] == m]
                        tot = len(m_sub)
                        cnt = (m_sub["direction"] == direction).sum()
                        vals.append(cnt / tot * 100 if tot else 0)
                    ax.bar(
                        np.arange(len(models)), vals, bar_w,
                        bottom=bottoms,
                        color=DIRECTION_COLORS.get(direction, "#999"),
                        label=DIRECTION_LABELS.get(direction, direction),
                    )
                    bottoms += np.array(vals)

                ax.set_xticks(np.arange(len(models)))
                ax.set_xticklabels(models, rotation=30, ha="right", fontsize=8)
                ax.set_ylabel("% of proposal turns")
                ax.set_ylim(0, 105)
                ax.set_title(game)
                if game == games_here[0]:
                    ax.legend(fontsize=7, loc="upper right")

            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        st.divider()

        # ── Section 2: Offer trajectory across refinement steps ──────────
        st.markdown("### Offer trajectory: how values shift step by step")
        st.caption(
            "Mean offer value at each refinement step across all traced turns. "
            "Step 0 = initial draft; step 1 = after first critique; step 2 = after second critique. "
            "Bands show ±1 SEM."
        )

        traj = df_rows_s.dropna(subset=["offer_value"]).copy()
        if traj.empty:
            st.info("No parseable offer values for this size.")
        else:
            games_traj = sorted(traj["game"].unique())
            fig, axes = plt.subplots(1, len(games_traj), figsize=(5 * len(games_traj), 3.6))
            if len(games_traj) == 1:
                axes = [axes]

            for ax, game in zip(axes, games_traj):
                sub = traj[traj["game"] == game]
                models = sorted(sub["model_short"].dropna().unique())
                palette = sns.color_palette("colorblind", len(models))

                for model, color in zip(models, palette):
                    msub = sub[sub["model_short"] == model]
                    agg = (
                        msub.groupby("version_idx")["offer_value"]
                        .agg(mean="mean", sem="sem", count="count")
                        .reset_index()
                    )
                    agg = agg.sort_values("version_idx")
                    ax.plot(agg["version_idx"], agg["mean"], marker="o", color=color, label=model)
                    ax.fill_between(
                        agg["version_idx"],
                        agg["mean"] - agg["sem"],
                        agg["mean"] + agg["sem"],
                        alpha=0.2, color=color,
                    )

                # x-tick labels
                ver_labels = {0: "initial", 1: "iter 1", 2: "iter 2", 3: "iter 3"}
                ax.set_xticks(sorted(sub["version_idx"].unique()))
                ax.set_xticklabels(
                    [ver_labels.get(v, f"iter {v}") for v in sorted(sub["version_idx"].unique())],
                    fontsize=8,
                )
                ax.set_title(game)
                ax.set_ylabel(OFFER_YLABELS.get(game, "Offer value"))
                ax.set_xlabel("Refinement step")
                ax.legend(fontsize=7)

            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        st.divider()

        # ── Section 3: Action flip analysis ──────────────────────────────
        st.markdown("### Action flips: does refinement change the decision type?")
        st.caption(
            "How often does the `<player answer>` change between initial draft and final version? "
            "E.g. initial NONE → final ACCEPT means the agent switched to accepting mid-loop."
        )

        action_s = df_sum_s.dropna(subset=["action_changed"]).copy()
        if action_s.empty:
            st.info("No action data available for this size.")
        else:
            # Flip rate per game × model
            flip_rate = (
                action_s.groupby(["game", "model_short"])["action_changed"]
                .mean()
                .reset_index()
                .rename(columns={"action_changed": "flip_rate"})
            )
            games_act = sorted(flip_rate["game"].unique())
            fig, axes = plt.subplots(1, len(games_act), figsize=(5 * len(games_act), 3.2))
            if len(games_act) == 1:
                axes = [axes]

            for ax, game in zip(axes, games_act):
                sub = flip_rate[flip_rate["game"] == game]
                sns.barplot(
                    data=sub, x="model_short", y="flip_rate",
                    color="#2ca02c", ax=ax,
                )
                ax.set_ylim(0, 1.05)
                ax.set_ylabel("Fraction of turns with action flip")
                ax.set_xlabel("")
                ax.set_title(game)
                ax.tick_params(axis="x", rotation=30, labelsize=8)

            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            # Transition matrix
            with st.expander("Action transition table (initial → final)"):
                trans = (
                    action_s[action_s["action_changed"]]
                    .groupby(["game", "initial_action", "final_action"])
                    .size()
                    .reset_index(name="count")
                )
                if trans.empty:
                    st.info("No action flips observed.")
                else:
                    st.dataframe(trans, use_container_width=True)

        st.divider()

        # ── Section 4: Monotonicity ───────────────────────────────────────
        st.markdown("### Monotonicity: does the offer converge consistently?")
        st.caption(
            "A turn is *monotone* if the offer moved in one consistent direction across all "
            "refinement steps (e.g. always decreasing, or always increasing). "
            "Only turns with ≥3 versions (initial + ≥2 iterations) are included."
        )

        mono_s = df_sum_s.dropna(subset=["is_monotone"]).copy()
        if mono_s.empty:
            st.info("No turns with ≥3 versions found for this size (need at least 2 refinement iterations).")
        else:
            mono_rate = (
                mono_s.groupby(["game", "model_short"])["is_monotone"]
                .mean()
                .reset_index()
                .rename(columns={"is_monotone": "monotone_rate"})
            )
            games_mono = sorted(mono_rate["game"].unique())
            fig, axes = plt.subplots(1, len(games_mono), figsize=(5 * len(games_mono), 3.2))
            if len(games_mono) == 1:
                axes = [axes]

            for ax, game in zip(axes, games_mono):
                sub = mono_rate[mono_rate["game"] == game]
                sns.barplot(
                    data=sub, x="model_short", y="monotone_rate",
                    color="#9467bd", ax=ax,
                )
                ax.set_ylim(0, 1.05)
                ax.set_ylabel("Fraction of turns (monotone)")
                ax.set_xlabel("")
                ax.set_title(game)
                ax.tick_params(axis="x", rotation=30, labelsize=8)

            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        # ── Raw data expander ─────────────────────────────────────────────
        with st.expander("Raw summary data"):
            show = [c for c in [
                "game", "model_short", "size", "p1_strategy", "p2_strategy",
                "iter_n", "turn_n", "initial_offer", "final_offer",
                "offer_delta", "ambition_delta", "direction",
                "is_monotone", "action_changed", "initial_action", "final_action",
            ] if c in df_sum_s.columns]
            st.dataframe(df_sum_s[show].round(2), use_container_width=True)
