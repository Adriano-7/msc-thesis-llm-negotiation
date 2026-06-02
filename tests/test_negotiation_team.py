"""Unit tests for NegotiationTeamAgent — no GPU / no model weights.

Members are replaced with a scripted ``FakeMember`` (via monkeypatching the
factory the team uses), so the three-phase deliberation, Borda tally, trace
shape and conversation hygiene are exercised deterministically.

Run:  pytest tests/test_negotiation_team.py -q
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import ratbench.utils as utils
from ratbench.agents.agents import Agent
from ratbench.agents.negotiation_team_agent import NegotiationTeamAgent
from ratbench.constants import AGENT_ONE


class FakeMember(Agent):
    """Agent whose chat() replays a fixed script of responses, one per think()."""

    def __init__(self, agent_name, script):
        super().__init__(agent_name=agent_name)
        self.prompt_entity_initializer = "system"
        self.conversation = []
        self._script = list(script)
        self._i = 0

    def chat(self):
        resp = self._script[self._i]
        self._i += 1
        return resp

    def update_conversation_tracking(self, role, message):
        self.conversation.append({"role": role, "content": message})


class StrictMember(FakeMember):
    """Like FakeMember, but enforces user/assistant alternation on every chat()
    the way a chat template (e.g. Gemma's) does — so a regression that adds two
    consecutive same-role turns fails loudly instead of silently."""

    def chat(self):
        roles = [m["role"] for m in self.conversation if m["role"] != "system"]
        for a, b in zip(roles, roles[1:]):
            if a == b:
                raise AssertionError(
                    f"Conversation roles must alternate; saw consecutive {a!r}: {roles}"
                )
        return super().chat()


# Per-member script with discussion_rounds=2 is:
#   [phase1_draft, round1_revision, round2_revision(=slate entry), phase3_ranking]
DRAFTS = ["MOVE_A", "MOVE_B", "ACCEPT_MOVE"]


def _scripts(rankings):
    """Build three member scripts where revisions keep the draft unchanged
    (so the slate equals DRAFTS) and Phase 3 returns the given ranking text."""
    return [
        [DRAFTS[i], DRAFTS[i], DRAFTS[i], rankings[i]] for i in range(3)
    ]


def _make_team(monkeypatch, rankings, discussion_rounds=2, member_cls=FakeMember):
    script_iter = iter(_scripts(rankings))

    def fake_factory(name, agent_name, strategy="default", **kw):
        return member_cls(agent_name, next(script_iter))

    monkeypatch.setattr(utils, "factory_agent", fake_factory)
    specs = [{"id": "fake/m0"}, {"id": "fake/m1"}, {"id": "fake/m2"}]
    return NegotiationTeamAgent(AGENT_ONE, specs, discussion_rounds=discussion_rounds)


def test_borda_picks_majority_favourite(monkeypatch):
    # All members rank B (index 1) first → B wins.
    team = _make_team(monkeypatch, ["<ranking> B > A > C </ranking>"] * 3)
    team.init_agent("SYSTEM PROMPT", " ROLE")
    final = team.step("opponent move")

    assert final == "MOVE_B"
    tr = team._last_deliberation_trace
    assert tr["winner_index"] == 1
    # B: 2 pts ×3 members = 6; A: 3; C: 0
    assert tr["borda_scores"] == [3, 6, 0]


def test_accept_can_win_from_slate(monkeypatch):
    # "Accept" is just a draft (index 2); ranking it first makes the team accept.
    team = _make_team(monkeypatch, ["<ranking> C > A > B </ranking>"] * 3)
    team.init_agent("SYSTEM PROMPT", " ROLE")
    final = team.step("opponent move")

    assert final == "ACCEPT_MOVE"
    assert team._last_deliberation_trace["winner_index"] == 2


def test_conversation_hygiene(monkeypatch):
    # After a turn, each member's history ends [..., user(turn), assistant(final)]
    # with no deliberation residue (no discussion / ranking prompts).
    team = _make_team(monkeypatch, ["<ranking> B > A > C </ranking>"] * 3)
    team.init_agent("SYSTEM PROMPT", " ROLE")
    team.step("opponent move")

    for m in team.members:
        assert m.conversation[-1] == {"role": "assistant", "content": "MOVE_B"}
        assert m.conversation[-2] == {"role": "user", "content": "opponent move"}
        blob = " ".join(e["content"] for e in m.conversation)
        assert "Rank ALL" not in blob
        assert "deliberating" not in blob
    # Team's public transcript mirrors the same final move.
    assert team.conversation[-1] == {"role": "assistant", "content": "MOVE_B"}


def test_trace_structure(monkeypatch):
    team = _make_team(monkeypatch, ["<ranking> A > B > C </ranking>"] * 3)
    team.init_agent("SYSTEM PROMPT", " ROLE")
    team.step("opponent move")

    tr = team._last_deliberation_trace
    assert tr["drafts"] == DRAFTS
    assert len(tr["discussion_rounds"]) == 2
    assert all(len(r) == 3 for r in tr["discussion_rounds"])
    assert tr["slate"] == DRAFTS
    assert len(tr["rankings"]) == 3
    assert len(tr["borda_scores"]) == 3
    assert tr["final"] == DRAFTS[tr["winner_index"]]


def test_reason_laundering_replaces_only_reason(monkeypatch):
    # Slate drafts carry a <reason> that references the deliberation, plus other
    # tags. The winning member gets one extra scripted chat() returning a clean
    # <reason>; laundering must swap only the reason and preserve every other tag.
    moves = [
        f"<my name> RED </my name><reason> raw reason {i}, see Candidate B </reason>"
        f"<player answer> NONE </player answer>"
        f"<newly proposed trade> RED Gives X: {i} </newly proposed trade>"
        for i in range(3)
    ]
    laundered = "<reason> clean first-person reasoning </reason>"
    # Per member: [draft, rev1, rev2 (=slate), ranking]; winner (B, idx 1) also
    # gets a 5th entry consumed by the laundering call. Extra trailing entries on
    # non-winners are never popped.
    script_iter = iter(
        [moves[i], moves[i], moves[i], "<ranking> B > A > C </ranking>", laundered]
        for i in range(3)
    )

    def fake_factory(name, agent_name, strategy="default", **kw):
        return FakeMember(agent_name, next(script_iter))

    monkeypatch.setattr(utils, "factory_agent", fake_factory)
    specs = [{"id": "fake/m0"}, {"id": "fake/m1"}, {"id": "fake/m2"}]
    team = NegotiationTeamAgent(AGENT_ONE, specs, discussion_rounds=2)
    team.init_agent("SYSTEM PROMPT", " ROLE")
    final = team.step("opponent move")

    tr = team._last_deliberation_trace
    assert tr["winner_index"] == 1
    # Reason is the laundered text; the raw (pre-launder) move is kept in the trace.
    assert tr["final_raw"] == moves[1]
    assert "clean first-person reasoning" in final
    assert "raw reason" not in final and "Candidate B" not in final
    # Every other tag survives byte-for-byte from the winning draft.
    assert "<player answer> NONE </player answer>" in final
    assert "<newly proposed trade> RED Gives X: 1 </newly proposed trade>" in final
    assert "<my name> RED </my name>" in final
    # Only the winner consumed the 5th (laundering) script entry.
    assert team.members[1]._i == 5
    assert team.members[0]._i == 4 and team.members[2]._i == 4


def test_member_roles_alternate_during_deliberation(monkeypatch):
    # Regression: every member's history must keep user/assistant alternating
    # through all three phases (Gemma's chat template rejects otherwise).
    team = _make_team(
        monkeypatch, ["<ranking> A > B > C </ranking>"] * 3, member_cls=StrictMember
    )
    team.init_agent("SYSTEM PROMPT", " ROLE")
    team.step("opponent move")  # raises AssertionError if alternation breaks


# ── pure-function tests (no team construction needed) ──────────────────

def test_parse_ranking_well_formed():
    assert NegotiationTeamAgent._parse_ranking("<ranking> C > A > B </ranking>", 3) == [2, 0, 1]


def test_parse_ranking_noisy_and_partial():
    # No tags, lowercase, extra words, duplicate, missing C → still recovers order.
    out = NegotiationTeamAgent._parse_ranking("i think b then a are best", 3)
    assert out == [1, 0]


def test_parse_ranking_unparseable():
    assert NegotiationTeamAgent._parse_ranking("no labels here", 3) == []


def test_borda_drops_empty_votes():
    # Every vote unparseable → deterministic fallback to candidate 0, zero scores.
    assert NegotiationTeamAgent._borda([[], [], []], 3) == (0, [0, 0, 0])


def test_borda_tally_and_tiebreak():
    # Votes: [A>B>C], [B>A>C] → A=2+1=3, B=1+2=3, C=0. Tie A/B → lowest index A.
    winner, scores = NegotiationTeamAgent._borda([[0, 1, 2], [1, 0, 2]], 3)
    assert scores == [3, 3, 0]
    assert winner == 0
