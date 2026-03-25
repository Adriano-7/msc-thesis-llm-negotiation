"""
Unified HuggingFace Agent for open-weight models.

Works with any model that supports transformers' chat template:
  - Qwen (Qwen/Qwen2.5-7B-Instruct, Qwen/Qwen3-8B, etc.)
  - DeepSeek (deepseek-ai/DeepSeek-V2-Lite-Chat, etc.)
  - Mistral (mistralai/Mistral-7B-Instruct-v0.3, etc.)

Usage:
    agent = HuggingFaceAgent(
        agent_name=AGENT_ONE,
        model_id="Qwen/Qwen2.5-7B-Instruct",
    )
"""

import torch
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText,  AutoTokenizer, BitsAndBytesConfig 
from ratbench.agents.agents import Agent
import time
from ratbench.constants import AGENT_ONE, AGENT_TWO

# ---------------------------------------------------------------------------
# Global model cache – the model is loaded into VRAM only ONCE per process,
# even when two agents share the same model_id (which is the typical case).
# ---------------------------------------------------------------------------
_SHARED_MODELS: dict = {}  # (model_id, quantization) -> (model, tokenizer)


def evict_unused_models(keep_keys: set):
    """Remove cached models not in *keep_keys* and free their VRAM.

    Args:
        keep_keys: set of ``(model_id, quantization)`` tuples to keep.
    """
    to_remove = [k for k in _SHARED_MODELS if k not in keep_keys]
    for key in to_remove:
        print(f"[HuggingFaceAgent] Evicting {key[0]} from VRAM")
        del _SHARED_MODELS[key]
    if to_remove:
        import gc
        gc.collect()
        torch.cuda.empty_cache()


def _load_model(model_id: str, quantization=None, dtype=torch.bfloat16, device_map="auto"):
    """Load or retrieve a cached (model, tokenizer) pair.

    Args:
        quantization: None (no quantization), "4bit", or "8bit".
    """
    cache_key = (model_id, quantization)
    if cache_key not in _SHARED_MODELS:
        quant_label = f" [{quantization}]" if quantization else ""
        print(f"\n[HuggingFaceAgent] Loading {model_id}{quant_label} … (one-time)")
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

        load_kwargs = dict(
            dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
        )

        if quantization == "4bit":
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif quantization == "8bit":
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
            )

        try:
            model = AutoModelForImageTextToText.from_pretrained(model_id, **load_kwargs)
        except ValueError:
            # Not a vision model — fall back to text-only
            model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)

        _SHARED_MODELS[cache_key] = (model, tokenizer)
    return _SHARED_MODELS[cache_key]


class HuggingFaceAgent(Agent):
    """Drop-in replacement for QwenAgent that works with *any* HF chat model."""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-7B-Instruct",
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        quantization: str = None,
        # These are passed through but not used for loading:
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.run_epoch_time_ms = str(round(time.time() * 1000))
        self.model_id = model_id
        self.conversation: list = []
        self.prompt_entity_initializer = "system"
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.do_sample = do_sample

        # Load (or reuse) model
        self.model, self.tokenizer = _load_model(self.model_id, quantization=quantization)

    # ------------------------------------------------------------------
    # Agent interface
    # ------------------------------------------------------------------

    def init_agent(self, system_prompt: str, role: str):
        self.conversation = []
        if AGENT_ONE in self.agent_name:
            self.conversation.append({"role": "system", "content": system_prompt})
            self.conversation.append({"role": "user", "content": role})
        elif AGENT_TWO in self.agent_name:
            self.conversation.append(
                {"role": "system", "content": system_prompt + role}
            )
        else:
            raise ValueError(f"Agent name must contain {AGENT_ONE} or {AGENT_TWO}")

    def chat(self) -> str:
        messages = [{"role": m["role"], "content": m["content"]} for m in self.conversation]
        
        if "gemma" in self.model_id.lower() and messages and messages[0]["role"] == "system":
            sys_content = messages.pop(0)["content"]
            
            if messages and messages[0]["role"] == "user":
                messages[0]["content"] = f"{sys_content}\n\n{messages[0]['content']}"
            else:
                messages.insert(0, {"role": "user", "content": sys_content})

        text = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        gen_kwargs = dict(
            max_new_tokens=self.max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        if self.do_sample:
            gen_kwargs["temperature"] = self.temperature
            gen_kwargs["top_p"] = self.top_p
            gen_kwargs["do_sample"] = True

        output_ids = self.model.generate(**inputs, **gen_kwargs)

        new_tokens = output_ids[0][inputs.input_ids.shape[1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def update_conversation_tracking(self, role: str, message: str):
        self.conversation.append({"role": role, "content": message})

    # ------------------------------------------------------------------
    # Serialisation helpers  (deepcopy, get_state)
    # ------------------------------------------------------------------

    def __deepcopy__(self, memo):
        from copy import deepcopy

        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k in ("model", "tokenizer"):
                # Store class name string instead of the heavy object
                setattr(result, k, type(v).__name__ if v is not None else "str")
            else:
                setattr(result, k, deepcopy(v, memo))
        return result