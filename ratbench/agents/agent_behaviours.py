from abc import ABC
from copy import deepcopy
from ratbench.agents.agents import Agent
from ratbench.constants import REASONING_TAG


class SelfCheckingAgent(Agent, ABC):
    def think(self):
        # do one step of thinking
        super().think()
        # print("reflecting!")
        # prompt agent to check proposal
        self.update_conversation_tracking("user", "Double check your proposal.")
        # think again
        return super().think()


class ReasoningAgent(Agent, ABC):
    def init_agent(self, system_prompt, role):
        system_prompt = (
            system_prompt
            + f"\nReason succinctly step by step about your response within <{REASONING_TAG}> [add reasoning] </{REASONING_TAG}> tags."
        )
        super().init_agent(system_prompt, role)


class SelfRefineAgent(Agent, ABC):
    """Self-Refine loop (Madaan et al., 2023).

    Generate → feedback → refine, iterated ``max_refine_iters`` times.
    Only the final refined answer is retained in ``self.conversation``;
    the feedback/refine exchanges happen on a scratch copy and are
    discarded, so outer-loop context does not grow with iterations.
    """

    max_refine_iters = 2

    feedback_prompt = (
        "Before finalizing your response, critique it along these axes. "
        "Be specific and actionable — \"make it better\" is not useful. "
        "If a dimension is fine, say OK and move on. Do not rewrite the "
        "response yet.\n\n"
        "1. Format: Does your response contain all required tags in the "
        "correct order, with valid content (e.g., a parseable <proposed "
        "trade>)? Note any missing or malformed tag.\n"
        "2. Payoff alignment: Given your <my goals> (or <my valuation>), "
        "does the proposed trade actually advance your payoff? Would a "
        "slightly different split give you more utility while still being "
        "plausibly acceptable? Quantify if possible.\n"
        "3. Opponent plausibility: Given what the other player has said "
        "or offered so far, is your proposal likely to be accepted, or "
        "will it almost certainly be rejected? A rejected proposal wastes "
        "your turn.\n"
        "4. Consistency: Is your <reason> consistent with your <player "
        "answer> and <newly proposed trade>? Does your <message> "
        "contradict your actual move?\n"
        "5. Rule compliance: Are you respecting the proposal budget and "
        "turn constraints stated in the rules?\n\n"
        "Emit your critique inside <critique> … </critique> tags. Include "
        "at most one concrete, actionable change per axis (e.g. \"increase "
        "X by 2\", \"switch from ACCEPT to propose\", \"remove the "
        "contradictory claim in <message>\") — not vague suggestions."
    )

    refine_prompt = (
        "Now rewrite your full response, incorporating the critique. Keep "
        "what was fine; change only what the critique flagged. Emit the "
        "complete response in the required format (all tags in order, as "
        "specified at the start of the game), not a diff. Do not include "
        "the critique in the final answer."
    )

    def think(self):
        saved = deepcopy(self.conversation)
        y = super().think()
        for _ in range(self.max_refine_iters):
            self.update_conversation_tracking("user", self.feedback_prompt)
            fb = self.chat()
            self.update_conversation_tracking("assistant", fb)
            self.update_conversation_tracking("user", self.refine_prompt)
            y = self.chat()
            self.update_conversation_tracking("assistant", y)
        self.conversation = saved
        self.update_conversation_tracking("assistant", y)
        return y
