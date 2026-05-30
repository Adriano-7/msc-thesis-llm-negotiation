# Negotiation Team — Design Note

*Working note to solidify the idea before implementation. A reference to keep next to the code.*

## What we're building

A **deliberating negotiation team** inside NegotiationArena. The team is a single negotiating party made of **N members with homogeneous roles** — every member is a symmetric generalist (no proposer/critic split) — facing **one opponent agent** across the framework's multi-turn scenarios.

Conceptually the team is a **deliberation ensemble**: N reasoners converging on the best move for one shared team objective. In effect this is ReConcile's round-table consensus transplanted from static single-answer reasoning into a **dynamic, stateful, adversarial**
negotiation, run on **small open-weight models** (Gemma, Qwen, Ministral, ~4–27B) instead of frontier APIs.

Why it's not just a re-run of ReConcile — four things change, and each is a reason the original findings might *not* transfer:
1. The "answer" is a **structured move**, not a label.
2. The **state evolves** turn to turn.
3. There's an **adversary** doing theory-of-mind on our output.
4. The models are **small and open-weight** (weaker at confidence, instruction-following).

## Per-turn loop (the three phases)

On each turn the team must act:

**Phase 1 — Independent drafts.** Every member independently drafts a candidate move +
reasoning, conditioned on the current game state and dialogue so far. A draft is a complete
move, and may be **either a counter-offer or "accept the opponent's current offer"** —
accepting is one possible move, not a separate decision.

**Phase 2 — Discussion rounds (×R).** Each member sees the others' drafts (grouped so
similar moves aggregate rather than concatenate) and revises its own draft + reasoning.
Repeats until convergence or R is reached. All internal — **the opponent never sees the
deliberation**, only the final move. This preserves a stable external persona across turns
and matches NegotiationArena's private-reasoning design.

**Phase 3 — Consensus (Borda).** The N revised drafts form a **slate**, and a **Borda count**
over that slate selects the team's move. Because "accept" is one of the drafts, a single
Borda ranking decides simultaneously whether to accept and, if not, which counter-offer to
send. The winning move is emitted; the opponent replies; the new state re-enters the loop.

## Two deliberate decisions (and why)

**Accept is folded into the slate — no separate accept/reject vote.**
In this setting the two are the *same* decision: whether to accept is only meaningful
relative to what the team believes it could get by countering. A separate referendum on
acceptance would be parasitic on the offer decision and could contradict it. One slate,
one ranking, no contradiction.

**No member confidence.**
Confidence-weighting was central to ReConcile because its tasks had a *truth value* that
calibrated confidence could track. A negotiation move has no truth value — only a payoff
against an opponent whose reservation point is hidden — so "confidence in a move" measures
the wrong thing. It also imports a known failure: small models are poorly calibrated and
overconfident, and in negotiation that signal is further contaminated by social stance
(an aggressive draft and a timid one may both report high confidence). Dropping it makes
Borda more faithful to the team objective and removes a fragile field to emit/parse.
*Principled successor:* if we later add explicit private member preferences, weight by a
member's **actual utility** for an offer, not self-reported certainty.

## The spine: heterogeneity

The diversity question is the **main axis**, not a side experiment. Run the identical
pipeline as:
- **Homogeneous team** — N instances of one model.
- **Heterogeneous team** — N different families.

Ask: *when does deliberation beat a single model, as a function of internal diversity?*
Both source papers predict diversity is the active ingredient (ReConcile's "w/o Multiple
Models" ablation lost 6.8 points; the negotiation-teams literature found elaborate
strategies only pay off when members are dissimilar), and this hasn't been tested for
sub-30B open-weight models in negotiation. **Validate the manipulation directly** by
measuring Phase-1 draft diversity (BERTScore-style), so a null result is diagnosable as
"drafts weren't diverse enough" vs "diversity doesn't help."

## Lineage

- **ReConcile** (Chen et al., 2023) — round-table consensus, multi-model, discussion +
  voting. Source of the propose → discuss → consensus loop.
- **NegotiationArena** (Bianchi et al., 2024) — the stateful, adversarial, multi-turn
  environment; structured moves; private reasoning.
- **Sánchez-Anguix et al. (2016)** — agent-based negotiation teams; intra-team strategies
  (Representative / voting / Borda / unanimity); the mediator-as-non-stakeholder idea; the
  diversity-matters finding. (Min/avg member-welfare metrics stay dormant here because
  members share one goal — they reactivate only if we add explicit private preferences.)
