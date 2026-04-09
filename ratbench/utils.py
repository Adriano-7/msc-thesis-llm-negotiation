import os
import copy
from ratbench.agents import ChatGPTAgent, ClaudeAgent

def normalize_model(entry):
    """Normalize a YAML model entry to ``{"id": str, "quantization": str|None}``.

    Accepts either a plain string (``"org/model"``) or a dict
    (``{"id": "org/model", "quantization": "4bit"}``).
    """
    if isinstance(entry, str):
        return {"id": entry, "quantization": None, "model_type": "llm", "enable_thinking": None}
    return {
        "id": entry["id"],
        "quantization": entry.get("quantization"),
        "model_type": entry.get("model_type", "llm"),
        "enable_thinking": entry.get("enable_thinking"),
    }

# Lazy import to avoid loading torch when not needed
_HF_AGENT = None

def _get_hf_agent_class():
    global _HF_AGENT
    if _HF_AGENT is None:
        from ratbench.agents.hf_agent import HuggingFaceAgent
        _HF_AGENT = HuggingFaceAgent
    return _HF_AGENT


def factory_agent(name, agent_name, **kwargs):
    """
    Factory to create agents by short name or HuggingFace model_id.

    Short names (backwards compatible):
        "gpt-4", "gpt-3.5", "claude-2", "claude-2.1"

    HuggingFace models (new):
        Any string containing "/" is treated as a HF model_id:
        "Qwen/Qwen2.5-7B-Instruct", "deepseek-ai/DeepSeek-V2-Lite-Chat", etc.
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

    # ── HuggingFace open-weight models ──
    elif "/" in name:
        HFAgent = _get_hf_agent_class()
        return HFAgent(agent_name=agent_name, model_id=name, **kwargs)

    else:
        raise ValueError(
            f"Unknown agent: {name}. Use a short name (gpt-4, claude-2, ...) "
            f"or a HuggingFace model_id (org/model-name)."
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