import sys

sys.path.append("../")
sys.path.append(".")

import os
import json
import string
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from utils import text_formatting, from_timestamp_str
from analysis.common import LOGS_ROOT, clean_name

st.set_page_config(page_title="Team Negotiation", layout="wide")

TEAM_ROOT = Path(LOGS_ROOT) / "negotiation_team"
_LABELS = string.ascii_uppercase

EXPLORER_DIR = os.path.dirname(os.path.dirname(__file__))
RED_AVATAR = os.path.join(EXPLORER_DIR, "red_robot.svg")
BLUE_AVATAR = os.path.join(EXPLORER_DIR, "blue_robot.svg")


# ----------------------------------------------------------------------
# Loaders
# ----------------------------------------------------------------------
def _team_player_index(game_state):
    """Index of the NegotiationTeamAgent player (fallback model == 'team')."""
    for i, p in enumerate(game_state.get("players", [])):
        if p.get("class") == "NegotiationTeamAgent" or p.get("model") == "team":
            return i
    return 0


@st.cache_data
def load_team_runs():
    rows = []
    if not TEAM_ROOT.exists():
        return pd.DataFrame()
    for gs_path in sorted(TEAM_ROOT.rglob("game_state.json")):
        rel = gs_path.relative_to(TEAM_ROOT).parts
        if len(rel) < 4:
            continue
        experiment = rel[0]
        size = rel[1]
        run_id = rel[-2]
        matchup = "/".join(rel[2:-2])
        try:
            with open(gs_path) as f:
                gs = json.load(f)
        except Exception:
            continue
        turns = gs.get("game_state", [])
        is_complete = bool(turns) and turns[-1].get("current_iteration") == "END"
        team_idx = _team_player_index(gs)
        team_label, _, opp_label = matchup.partition("_vs_")
        rows.append(
            {
                "experiment": experiment,
                "size": size,
                "matchup": matchup,
                "team_label": team_label or matchup,
                "opponent_label": opp_label,
                "run_id": run_id,
                "is_complete": is_complete,
                "team_idx": team_idx,
                "run_dir": str(gs_path.parent),
            }
        )
    return pd.DataFrame(rows)


def load_deliberation_traces(run_dir):
    """Return {(iter_n, turn_n): trace} for all deliberation_trace files in a run."""
    traces = {}
    for tp in sorted(Path(run_dir).glob("deliberation_trace_iter_*_turn_*.json")):
        parts = tp.stem.split("_")
        if len(parts) < 6:
            continue
        try:
            iter_n = int(parts[3])
            turn_n = int(parts[5])
            with open(tp) as f:
                traces[(iter_n, turn_n)] = json.load(f)
        except Exception:
            continue
    return traces


def build_team_trace_list(game_state, traces, team_idx):
    """Ordered list of traces matched to the team player's assistant messages."""
    out = []
    for state in game_state.get("game_state", []):
        turn = state.get("turn")
        iteration = state.get("current_iteration")
        if turn != team_idx or not isinstance(iteration, int):
            continue
        out.append(traces.get((iteration, turn)))
    return out


# ----------------------------------------------------------------------
# Rich deliberation rendering
# ----------------------------------------------------------------------
def _draft_columns(drafts, label):
    cols = st.columns(len(drafts))
    for k, (col, draft) in enumerate(zip(cols, drafts)):
        with col:
            st.caption(f"{label} {k + 1}")
            st.code(text_formatting(draft or "", False), language="text")


def _render_borda(trace):
    slate = trace.get("slate", [])
    rankings = trace.get("rankings", [])
    scores = trace.get("borda_scores", [])
    winner = trace.get("winner_index", 0)
    n = len(slate)
    if n == 0:
        return
    cand_labels = [_LABELS[k] for k in range(n)]

    # Voting table: rows = members + Borda score row, cols = candidate labels.
    data = {}
    for k, lab in enumerate(cand_labels):
        col_vals = []
        for r in rankings:
            col_vals.append(str(r.index(k) + 1) if k in r else "—")
        col_vals.append(str(scores[k]) if k < len(scores) else "—")
        data[f"Candidate {lab}"] = col_vals
    index = [f"Member {i + 1} rank" for i in range(len(rankings))] + ["Borda score"]
    df = pd.DataFrame(data, index=index)

    win_col = f"Candidate {_LABELS[winner]}" if winner < n else None

    def _highlight(col):
        return [
            "background-color: #d8f5d8; font-weight: 600" if col.name == win_col else ""
            for _ in col
        ]

    st.markdown("**Rankings & Borda tally** (rank 1 = best for the team)")
    st.dataframe(df.style.apply(_highlight, axis=0), use_container_width=True)

    # Borda bar chart with the winning candidate highlighted.
    if scores:
        fig, ax = plt.subplots(figsize=(max(3, n), 2.4))
        colors = ["#2e9e3f" if k == winner else "#9fb3c8" for k in range(len(scores))]
        ax.bar([_LABELS[k] for k in range(len(scores))], scores, color=colors)
        ax.set_ylabel("Borda points")
        ax.set_xlabel("Candidate")
        ax.set_title("Consensus vote")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        st.pyplot(fig)
        plt.close(fig)


def render_deliberation(trace):
    if not trace:
        return
    drafts = trace.get("drafts", [])
    rounds = trace.get("discussion_rounds", [])
    n_members = len(drafts)
    title = f"🧩 Team Deliberation — {n_members} members, {len(rounds)} discussion round{'s' if len(rounds) != 1 else ''}"
    with st.expander(title):
        if drafts:
            st.markdown("### Phase 1 — Independent drafts")
            _draft_columns(drafts, "Member")

        if rounds:
            st.markdown("### Phase 2 — Discussion rounds")
            for r_idx, revised in enumerate(rounds, start=1):
                st.markdown(f"**Round {r_idx}**")
                _draft_columns(revised, "Member")

        st.markdown("### Phase 3 — Consensus (Borda)")
        _render_borda(trace)

        winner = trace.get("winner_index", 0)
        final = trace.get("final", "")
        win_label = _LABELS[winner] if isinstance(winner, int) and winner < 26 else "?"
        st.markdown(f"**Winning move — Candidate {win_label} (team's emitted move)**")
        st.success(text_formatting(final or "", False))


# ----------------------------------------------------------------------
# Page
# ----------------------------------------------------------------------
st.title("Team Negotiation")

runs = load_team_runs()
if runs.empty:
    st.info(f"No team negotiation runs found under `{TEAM_ROOT}`.")
    st.stop()

# Sidebar cascading filters.
experiments = sorted(runs["experiment"].unique())
experiment = st.sidebar.selectbox("Experiment", experiments)
runs_e = runs[runs["experiment"] == experiment]

sizes = sorted(runs_e["size"].unique())
size = st.sidebar.selectbox("Model size", sizes)
runs_es = runs_e[runs_e["size"] == size]

matchups = sorted(runs_es["matchup"].unique())
matchup = st.sidebar.selectbox("Matchup", ["All"] + matchups)
filtered = runs_es if matchup == "All" else runs_es[runs_es["matchup"] == matchup]

tab_overview, tab_conv = st.tabs(["Overview", "Conversation & Phases"])

# ---- Tab 1: Overview --------------------------------------------------
with tab_overview:
    st.subheader(f"{experiment} · {size}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Runs", len(filtered))
    completion = filtered["is_complete"].mean() if len(filtered) else 0.0
    c2.metric("Completion rate", f"{completion * 100:.0f}%")

    # Aggregate consensus stats over all deliberation traces in the filtered set.
    unanimous = 0
    n_turns = 0
    margins = []
    for _, row in filtered.iterrows():
        for trace in load_deliberation_traces(row["run_dir"]).values():
            n_turns += 1
            winner = trace.get("winner_index", 0)
            rankings = [r for r in trace.get("rankings", []) if r]
            if rankings and all(r[0] == winner for r in rankings):
                unanimous += 1
            scores = sorted(trace.get("borda_scores", []), reverse=True)
            if len(scores) >= 2 and scores[0] > 0:
                margins.append((scores[0] - scores[1]) / scores[0])
    c3.metric(
        "Unanimity rate",
        f"{(unanimous / n_turns * 100):.0f}%" if n_turns else "—",
        help="Share of team turns where every member ranked the winning candidate first.",
    )

    m1, m2 = st.columns(2)
    m1.metric("Team deliberation turns", n_turns)
    m2.metric(
        "Mean Borda margin",
        f"{(sum(margins) / len(margins) * 100):.0f}%" if margins else "—",
        help="Winner minus runner-up Borda score, normalized by the winner's score.",
    )

    st.markdown("**Runs per matchup**")
    per_matchup = (
        filtered.groupby("matchup")
        .agg(runs=("run_id", "count"), completed=("is_complete", "sum"))
        .reset_index()
    )
    st.dataframe(per_matchup, use_container_width=True, hide_index=True)

# ---- Tab 2: Conversation & Phases ------------------------------------
with tab_conv:
    if filtered.empty:
        st.warning("No runs match the current filters.")
    else:
        opts = filtered.copy()
        opts["label"] = opts.apply(
            lambda r: f"{'✓' if r.is_complete else '✗'} {r.matchup} · "
            f"{from_timestamp_str(r.run_id)} · {r.run_id}",
            axis=1,
        )
        choice = st.selectbox("Select a run", list(opts["label"]))
        row = opts[opts["label"] == choice].iloc[0]
        run_dir = row["run_dir"]

        with open(os.path.join(run_dir, "game_state.json")) as f:
            game_state = json.load(f)

        players = game_state.get("players", [])
        team_idx = int(row["team_idx"])
        opp_idx = 1 - team_idx if len(players) == 2 else team_idx
        team_p = players[team_idx]
        opp_p = players[opp_idx] if len(players) > opp_idx else {}

        # Run header.
        specs = team_p.get("member_specs", [])
        members = ", ".join(clean_name(s.get("id", "?")) for s in specs) or "—"
        h1, h2, h3 = st.columns(3)
        h1.markdown(f"**Team members**\n\n{members}")
        h2.markdown(f"**Discussion rounds**\n\n{team_p.get('discussion_rounds', '—')}")
        h3.markdown(
            f"**Opponent**\n\n{clean_name(opp_p.get('model_id', opp_p.get('model', '—')))}"
        )

        # System prompts.
        cs1, cs2 = st.columns(2)
        with cs1:
            with st.expander(f"System prompt — {team_p.get('agent_name', 'Team')}"):
                conv = team_p.get("conversation", [])
                if conv:
                    st.write(text_formatting(conv[0]["content"], True))
        with cs2:
            with st.expander(f"System prompt — {opp_p.get('agent_name', 'Opponent')}"):
                conv = opp_p.get("conversation", [])
                if conv:
                    st.write(text_formatting(conv[0]["content"], True))

        st.markdown("---")

        # Deliberation traces matched to the team's assistant messages.
        traces = load_deliberation_traces(run_dir)
        team_traces = build_team_trace_list(game_state, traces, team_idx)

        team_name = team_p.get("agent_name", "Team")
        opp_name = opp_p.get("agent_name", "Opponent")
        team_conv = [m for m in team_p.get("conversation", []) if m["role"] == "assistant"]
        opp_conv = [m for m in opp_p.get("conversation", []) if m["role"] == "assistant"]

        team_avatar = RED_AVATAR if team_idx == 0 else BLUE_AVATAR
        opp_avatar = BLUE_AVATAR if team_idx == 0 else RED_AVATAR

        # Interleave in play order: lower-indexed player speaks first each round.
        first_is_team = team_idx <= opp_idx
        for i in range(max(len(team_conv), len(opp_conv))):
            def _render_team(i=i):
                if i < len(team_conv):
                    with st.chat_message(team_name, avatar=team_avatar):
                        st.write(text_formatting(team_conv[i]["content"], False))
                        if i < len(team_traces):
                            render_deliberation(team_traces[i])

            def _render_opp(i=i):
                if i < len(opp_conv):
                    with st.chat_message(opp_name, avatar=opp_avatar):
                        thinking = opp_conv[i].get("thinking")
                        if thinking:
                            with st.expander("💭 Chain of Thought"):
                                st.markdown(thinking)
                        st.write(text_formatting(opp_conv[i]["content"], False))

            if first_is_team:
                _render_team()
                _render_opp()
            else:
                _render_opp()
                _render_team()
