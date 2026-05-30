"""Deliberating negotiation team agent.

A single negotiating party made of N homogeneous-role members that, on every
turn, run a three-phase deliberation and emit one move:

    Phase 1 — Independent drafts.   Each member drafts a complete move.
    Phase 2 — Discussion rounds ×R. Each member sees the others' (grouped)
              drafts and revises its own.
    Phase 3 — Consensus (Borda).    Each member ranks the slate of revised
              drafts; a Borda count selects the team's move.

The deliberation is entirely internal: the opponent only ever sees the final
move, exactly like the private-reasoning design of NegotiationArena. "Accept
the opponent's offer" is just one possible draft, so a single Borda ranking
decides accept-vs-counter simultaneously (no separate accept vote, no member
confidence) — see ``negotiation_team_design_note.md``.

This mirrors the save/restore discipline of ``SelfRefineAgent``: each phase
runs on the members' scratch conversations, which are rolled back so member
context does not grow with deliberation; only the agreed final move is
committed to the shared history. The full deliberation is stored in
``self._last_deliberation_trace`` and the game engine persists it to
``deliberation_trace_iter_{i}_turn_{t}.json``.
"""

import re
import time
import string
from copy import deepcopy

from ratbench.agents.agents import Agent
from ratbench.utils import get_tag_contents


# Labels used to present the slate of candidate moves to the members.
_LABELS = string.ascii_uppercase


class NegotiationTeamAgent(Agent):
    """Round-table consensus team standing in for a single NegotiationArena party."""

    def __init__(
        self,
        agent_name,
        member_specs,
        discussion_rounds=2,
        max_new_tokens=1024,
        **kwargs,
    ):
        super().__init__(agent_name=agent_name)  # validates AGENT_ONE / AGENT_TWO
        # Local import keeps the factory <-> agent dependency from forming a cycle.
        from ratbench.utils import factory_agent

        if not member_specs:
            raise ValueError("NegotiationTeamAgent requires at least one member spec.")

        self.run_epoch_time_ms = str(round(time.time() * 1000))
        self.conversation = []
        self.prompt_entity_initializer = "system"
        self.discussion_rounds = discussion_rounds
        self.model = "team"
        self.member_specs = member_specs  # kept for logging / reproducibility

        self.members = [
            factory_agent(
                spec["id"],
                agent_name=agent_name,
                strategy="default",
                quantization=spec.get("quantization"),
                model_type=spec.get("model_type", "llm"),
                enable_thinking=spec.get("enable_thinking"),
                max_new_tokens=max_new_tokens,
            )
            for spec in member_specs
        ]
        self._last_deliberation_trace = None

    # ------------------------------------------------------------------
    # Agent interface — the team keeps its own public transcript and mirrors
    # team-level events (incoming move, final move, parse-retry errors) to
    # every member so each member shares the same external history.
    # ------------------------------------------------------------------

    def init_agent(self, system_prompt, role):
        for m in self.members:
            m.init_agent(system_prompt, role)
        # Members are identical at init; adopt one as the team's public transcript.
        self.conversation = deepcopy(self.members[0].conversation)

    def update_conversation_tracking(self, role, message):
        self.conversation.append({"role": role, "content": message})
        for m in self.members:
            m.update_conversation_tracking(role, message)

    def chat(self):
        raise NotImplementedError(
            "NegotiationTeamAgent acts via think(); chat() is never called directly."
        )

    # ------------------------------------------------------------------
    # Deliberation
    # ------------------------------------------------------------------

    def think(self):
        # ── Phase 1 — independent drafts ──────────────────────────────
        saved = [deepcopy(m.conversation) for m in self.members]
        drafts = [m.think() for m in self.members]
        for m, s in zip(self.members, saved):
            m.conversation = s

        # ── Phase 2 — discussion rounds (×R) ──────────────────────────
        current = list(drafts)
        rounds_trace = []
        for _ in range(self.discussion_rounds):
            revised = []
            for i, m in enumerate(self.members):
                s = deepcopy(m.conversation)
                m.update_conversation_tracking(
                    "user", self._discussion_prompt(current, own=i)
                )
                revised.append(m.think())
                m.conversation = s
            rounds_trace.append(list(revised))
            current = revised

        # ── Phase 3 — per-member ranking + Borda over the slate ───────
        slate = current
        rankings = []
        for m in self.members:
            s = deepcopy(m.conversation)
            m.update_conversation_tracking("user", self._ranking_prompt(slate))
            rankings.append(self._parse_ranking(m.think(), len(slate)))
            m.conversation = s
        winner_idx, borda_scores = self._borda(rankings, len(slate))
        final = slate[winner_idx]

        self._last_deliberation_trace = {
            "drafts": drafts,
            "discussion_rounds": rounds_trace,
            "slate": slate,
            "rankings": rankings,
            "borda_scores": borda_scores,
            "winner_index": winner_idx,
            "final": final,
        }

        # Commit only the agreed move to the team and to every member.
        self.update_conversation_tracking("assistant", final)
        return final

    # ------------------------------------------------------------------
    # Internal deliberation prompts (never seen by the opponent)
    # ------------------------------------------------------------------

    def _discussion_prompt(self, drafts, own):
        """Show member *own* its draft plus the grouped peer drafts and ask it
        to revise. Identical drafts are aggregated rather than concatenated so
        agreement is visible as support, not repetition."""
        groups = self._group_peers(drafts, own)
        peer_block = "\n\n".join(
            f"--- Proposal {_LABELS[k]} (supported by {count} "
            f"member{'s' if count != 1 else ''}) ---\n{text}"
            for k, (text, count) in enumerate(groups)
        )
        return (
            "Your team is deliberating privately before sending one move. This "
            "discussion is never shown to the other player.\n\n"
            "Your current proposal:\n"
            f"{drafts[own]}\n\n"
            "Your teammates independently proposed:\n"
            f"{peer_block}\n\n"
            "Weigh these against your own. If a teammate's reasoning is "
            "stronger for the team's payoff, adopt or merge it; if yours is "
            "stronger, keep it. Then output your (possibly revised) full move "
            "in exactly the required response format — all tags in order, as "
            "specified at the start of the game. Output only that move."
        )

    def _ranking_prompt(self, slate):
        """Ask a member to rank the whole slate best→worst for the team."""
        slate_block = "\n\n".join(
            f"--- Candidate {_LABELS[k]} ---\n{text}" for k, text in enumerate(slate)
        )
        labels = ", ".join(_LABELS[k] for k in range(len(slate)))
        return (
            "Your team must pick one of the following candidate moves to send. "
            "This vote is private.\n\n"
            f"{slate_block}\n\n"
            f"Rank ALL candidates ({labels}) from best to worst for your team's "
            "payoff in this negotiation, accounting for how likely the other "
            "player is to accept. Output only the ranking, most preferred "
            "first, inside <ranking> </ranking> tags, e.g. "
            f"<ranking> {' > '.join(_LABELS[k] for k in range(len(slate)))} </ranking>."
        )

    # ------------------------------------------------------------------
    # Borda helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _group_peers(drafts, own):
        """Collapse exact-equal peer drafts into (text, support_count) groups,
        excluding the member's own draft. Preserves first-seen order."""
        groups = []
        index = {}
        for i, text in enumerate(drafts):
            if i == own:
                continue
            key = text.strip()
            if key in index:
                t, c = groups[index[key]]
                groups[index[key]] = (t, c + 1)
            else:
                index[key] = len(groups)
                groups.append((text, 1))
        return groups

    @staticmethod
    def _parse_ranking(text, slate_len):
        """Extract an ordered list of candidate indices from a member's vote.

        Robust to small-model noise: reads the <ranking> block if present
        (else the whole text), takes valid candidate letters in order of first
        appearance. Returns [] when nothing parseable is found (that vote is
        dropped from the tally)."""
        try:
            body = get_tag_contents(text, "ranking")
        except ValueError:
            body = text
        valid = {_LABELS[k]: k for k in range(slate_len)}
        order = []
        seen = set()
        # Only standalone single-letter tokens count as labels, so prose words
        # ("best", "A move") and embedded letters ("MOVE_A") are not misread.
        for ch in re.findall(r"\b([A-Za-z])\b", body):
            ch = ch.upper()
            if ch in valid and ch not in seen:
                seen.add(ch)
                order.append(valid[ch])
        return order

    @staticmethod
    def _borda(rankings, slate_len):
        """Tally Borda points over member rankings.

        A candidate ranked at position k (0 = best) in a vote earns
        ``slate_len - 1 - k`` points; candidates a vote omits earn 0 (tied
        last). Winner is the highest total, deterministic lowest-index
        tie-break. Falls back to candidate 0 if every vote was unparseable."""
        scores = [0] * slate_len
        counted = 0
        for ranking in rankings:
            if not ranking:
                continue
            counted += 1
            for pos, idx in enumerate(ranking):
                scores[idx] += slate_len - 1 - pos
        if counted == 0:
            return 0, scores
        winner = max(range(slate_len), key=lambda i: scores[i])
        return winner, scores

    # ------------------------------------------------------------------
    # Lean serialization — log the team's public transcript only, not the N
    # member transcripts (which would bloat game_state.json on every turn).
    # Mid-game resume is not supported for teams; runs start fresh.
    # ------------------------------------------------------------------

    def get_state(self):
        return {
            "class": self.__class__.__name__,
            "conversation": deepcopy(self.conversation),
            "run_epoch_time_ms": self.run_epoch_time_ms,
            "agent_name": self.agent_name,
            "member_specs": deepcopy(self.member_specs),
            "discussion_rounds": self.discussion_rounds,
        }
