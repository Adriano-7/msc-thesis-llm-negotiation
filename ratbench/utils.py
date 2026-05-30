import os
import copy
from ratbench.agents import ChatGPTAgent, ClaudeAgent

def normalize_model(entry):
    """Normalize a YAML model entry to ``{"id": str, "quantization": str|None}``.

    Accepts either a plain string (``"org/model"``) or a dict
    (``{"id": "org/model", "quantization": "4bit"}``).
    """
    if isinstance(entry, str):
        return {
            "id": entry,
            "quantization": None,
            "model_type": "llm",
            "enable_thinking": None,
            "rate_limit_delay": None,
        }
    return {
        "id": entry["id"],
        "quantization": entry.get("quantization"),
        "model_type": entry.get("model_type", "llm"),
        "enable_thinking": entry.get("enable_thinking"),
        "rate_limit_delay": entry.get("rate_limit_delay"),
    }

def factory_agent(name, agent_name, strategy="default", **kwargs):
    """
    Factory to create agents by short name or HuggingFace model_id.

    Short names (backwards compatible):
        "gpt-4", "gpt-3.5", "claude-2", "claude-2.1"

    HuggingFace models:
        Any string containing "/" is treated as a HF model_id:
        "Qwen/Qwen2.5-7B-Instruct", "deepseek-ai/DeepSeek-V2-Lite-Chat", etc.

    strategy (HF only):
        "default"     — plain one-shot generation
        "self_check"  — one extra double-check pass (SelfCheckingAgent)
        "self_refine" — Self-Refine feedback/refine loop (Madaan et al. 2023)
    """
    # ── Legacy API-based agents ──
    if name == "gpt-4":
        return ChatGPTAgent(agent_name=agent_name, model="gpt-4-1106-preview")
    elif name == "gpt-3.5":
        return ChatGPTAgent(agent_name=agent_name, model="gpt-3.5-turbo-1106")
    elif name == "claude-2":
        return ClaudeAgent(agent_name=agent_name, model="claude-2")
    elif name == "claude-2.1":
        return ClaudeAgent(agent_name=agent_name, model="claude-2.1")

    # ── Hosted iaedu chatbot API ──
    elif name == "gpt-4o-iaedu":
        from ratbench.agents.iaedu_agent import (
            IaEduAgent,
            SelfCheckingIaEduAgent,
            SelfRefineIaEduAgent,
        )
        strategy_to_cls = {
            "default": IaEduAgent,
            "self_check": SelfCheckingIaEduAgent,
            "self_refine": SelfRefineIaEduAgent,
        }
        if strategy not in strategy_to_cls:
            raise ValueError(
                f"Unknown strategy: {strategy!r}. "
                f"Expected one of: {list(strategy_to_cls)}"
            )
        # Drop HF-only kwargs the runner forwards unconditionally.
        hf_only = {"quantization", "model_type", "enable_thinking"}
        clean_kwargs = {k: v for k, v in kwargs.items() if k not in hf_only}
        return strategy_to_cls[strategy](agent_name=agent_name, **clean_kwargs)

    # ── HuggingFace open-weight models ──
    elif "/" in name:
        # Local imports keep torch out of the import graph until needed.
        from ratbench.agents.hf_agent import (
            HuggingFaceAgent,
            SelfCheckingHuggingFaceAgent,
            SelfRefineHuggingFaceAgent,
        )
        strategy_to_cls = {
            "default": HuggingFaceAgent,
            "self_check": SelfCheckingHuggingFaceAgent,
            "self_refine": SelfRefineHuggingFaceAgent,
        }
        if strategy not in strategy_to_cls:
            raise ValueError(
                f"Unknown strategy: {strategy!r}. "
                f"Expected one of: {list(strategy_to_cls)}"
            )
        cls = strategy_to_cls[strategy]
        return cls(agent_name=agent_name, model_id=name, **kwargs)

    else:
        raise ValueError(
            f"Unknown agent: {name}. Use a short name (gpt-4, claude-2, ...) "
            f"or a HuggingFace model_id (org/model-name)."
        )

def build_party(spec, agent_name, strategy="default", **defaults):
    """Build a negotiating party for one player slot.

    A party is either a single agent or a deliberating team:

    - If *spec* carries a ``"team"`` block, returns a ``NegotiationTeamAgent``
      whose members are built from ``spec["team"]["members"]``. The team's
      members all share *agent_name* so their conversation shape matches the
      slot (P1 vs P2 differ — see ``HuggingFaceAgent.init_agent``).
    - Otherwise *spec* is a normalized model dict and this delegates to
      :func:`factory_agent`, forwarding the per-model fields.

    ``strategy`` and ``defaults`` (e.g. ``rate_limit_delay``) apply to the
    single-agent path; teams always run their members with the default
    strategy.
    """
    team = spec.get("team") if isinstance(spec, dict) else None
    if team:
        from ratbench.agents.negotiation_team_agent import NegotiationTeamAgent

        return NegotiationTeamAgent(
            agent_name=agent_name,
            member_specs=team["members"],
            discussion_rounds=team.get("discussion_rounds", 2),
        )

    return factory_agent(
        spec["id"],
        agent_name=agent_name,
        strategy=strategy,
        quantization=spec.get("quantization"),
        model_type=spec.get("model_type", "llm"),
        enable_thinking=spec.get("enable_thinking"),
        rate_limit_delay=spec.get("rate_limit_delay"),
        **defaults,
    )


def get_tag_contents(response, interest_tag):
    start_index, end_index, length = get_tag_indices(response, interest_tag)
    if start_index == -1 or end_index == -1:
        raise ValueError(
            f"Missing required tag: <{interest_tag}>...</{interest_tag}>. "
            f"Make sure your response contains both the opening <{interest_tag}> "
            f"and closing </{interest_tag}> tags."
        )
    contents = response[start_index + length : end_index].lstrip(" ").rstrip(" ")
    return copy.deepcopy(contents)


def get_tag_indices(response, interest_tag):
    start_index = response.find(f"<{interest_tag}>")
    end_index = response.find(f"</{interest_tag}>")
    return start_index, end_index, len(f"<{interest_tag}>")


def text_to_dict(s):
    try:
        return {k: int(v) for k, v in (item.split(": ") for item in s.split(", "))}
    except (ValueError, IndexError) as exc:
        raise ValueError(
            f"Could not parse resource string '{s}'. "
            f"Expected format: 'KEY: VALUE, KEY: VALUE' (e.g. 'X: 1' or 'ZUP: 50')."
        ) from exc


def get_next_filename(prefix, folder="."):
    prefix = prefix + "_"
    if not os.path.exists(folder):
        return prefix[:-1]
    # List all files with the given prefix in the current directory
    files = [file for file in os.listdir(folder) if file.startswith(prefix)]

    # Extract the numeric part of the file names and find the maximum
    numbers = [
        int(file[len(prefix) :]) for file in files if file[len(prefix) :].isdigit()
    ]

    # Determine the next integer in the sequence
    next_number = max(numbers, default=0) + 1

    # Generate the next file name
    next_filename = f"{prefix}{next_number}"

    return next_filename