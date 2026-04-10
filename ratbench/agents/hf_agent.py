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

import re
import torch
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText,  AutoTokenizer, BitsAndBytesConfig
from ratbench.agents.agents import Agent
import time
from ratbench.constants import AGENT_ONE, AGENT_TWO

# Matches <think>…</think> blocks, or a leading block where <think> was
# swallowed by skip_special_tokens (text starts mid-thought, ends with </think>).
_THINK_FULL_RE = re.compile(r"<think>(.*?)</think>\s*", flags=re.DOTALL)
_THINK_LEADING_RE = re.compile(r"^(.*?)</think>\s*", flags=re.DOTALL)

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


def _load_model(model_id: str, quantization=None, model_type="llm", dtype=torch.bfloat16, device_map="auto"):
    """Load or retrieve a cached (model, tokenizer) pair.

    Args:
        quantization: None (no quantization), "4bit", or "8bit".
        model_type: "llm" (text-only, default) or "vlm" (vision-language).
    """
    cache_key = (model_id, quantization)
    if cache_key not in _SHARED_MODELS:
        quant_label = f" [{quantization}]" if quantization else ""
        print(f"\n[HuggingFaceAgent] Loading {model_id}{quant_label} … (one-time)")
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

        load_kwargs = dict(
            device_map=device_map,
            trust_remote_code=True,
        )

        if quantization == "4bit":
            # Don't pass dtype — BnB handles weight loading internally;
            # bnb_4bit_compute_dtype sets the dtype used for computation.
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
        else:
            load_kwargs["dtype"] = dtype

        AutoClass = AutoModelForImageTextToText if model_type == "vlm" else AutoModelForCausalLM
        model = AutoClass.from_pretrained(model_id, **load_kwargs)

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
        model_type: str = "llm",
        enable_thinking: bool = None,
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
        self.enable_thinking = enable_thinking
        self._quantization = quantization
        self._model_type = model_type
        self._last_thinking_content = None

        # Auto-increase token budget for thinking mode
        if self.enable_thinking and self.max_new_tokens <= 4096:
            self.max_new_tokens = 16384

        # Model is loaded lazily on first chat() call
        self.model = model_id
        self._model = None
        self._tokenizer = None

    def _ensure_loaded(self):
        if self._model is None:
            self._model, self._tokenizer = _load_model(
                self.model_id, quantization=self._quantization, model_type=self._model_type
            )

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
        self._ensure_loaded()
        messages = [{"role": m["role"], "content": m["content"]} for m in self.conversation]
        
        if "gemma" in self.model_id.lower() and messages and messages[0]["role"] == "system":
            sys_content = messages.pop(0)["content"]
            
            if messages and messages[0]["role"] == "user":
                messages[0]["content"] = f"{sys_content}\n\n{messages[0]['content']}"
            else:
                messages.insert(0, {"role": "user", "content": sys_content})

        chat_template_kwargs = dict(
            tokenize=False,
            add_generation_prompt=True,
        )
        if self.enable_thinking is not None:
            chat_template_kwargs["enable_thinking"] = self.enable_thinking

        text = self._tokenizer.apply_chat_template(
            messages,
            **chat_template_kwargs,
        )
        inputs = self._tokenizer([text], return_tensors="pt").to(self._model.device)

        gen_kwargs = dict(
            max_new_tokens=self.max_new_tokens,
            pad_token_id=self._tokenizer.eos_token_id,
        )
        if self.do_sample:
            gen_kwargs["temperature"] = self.temperature
            gen_kwargs["top_p"] = self.top_p
            gen_kwargs["do_sample"] = True

        output_ids = self._model.generate(**inputs, **gen_kwargs)

        new_tokens = output_ids[0][inputs.input_ids.shape[1] :]

        # Extract thinking content 
        thinking, response = self._split_thinking(new_tokens)
        if thinking is None and self.enable_thinking:
            raise RuntimeError(
                f"Thinking truncated: no </think> found in {len(new_tokens)} tokens "
                f"(max_new_tokens={self.max_new_tokens}). "
                f"Increase max_new_tokens to give the model room to finish reasoning."
            )
        self._last_thinking_content = thinking
        return response

    def _split_thinking(self, tokens):
        """Separate thinking content from the response.

        Returns (thinking_text, response_text).  thinking_text is None when
        the model did not produce a thinking block.

        Strategy:
        1. Look for the </think> *token* in the output token ids and split
           there.  This is the only reliable method for models like Qwen3.5
           where <think> and </think> are special tokens that
           skip_special_tokens=True silently removes.
        2. Fall back to regex on the decoded text for models that emit
           <think>...</think> as regular text tokens.
        """
        # --- Token-level split (primary) --------------------------------
        think_end_id = self._tokenizer.convert_tokens_to_ids("</think>")
        unk_id = getattr(self._tokenizer, "unk_token_id", None)
        if think_end_id is not None and think_end_id != unk_id:
            mask = (tokens == think_end_id)
            if mask.any():
                idx = mask.nonzero(as_tuple=True)[0][0].item()
                thinking = self._tokenizer.decode(
                    tokens[:idx], skip_special_tokens=True
                ).strip()
                response = self._tokenizer.decode(
                    tokens[idx + 1 :], skip_special_tokens=True
                ).strip()
                return thinking or None, response

        # --- Regex fallback (for models using plain-text tags) ----------
        raw = self._tokenizer.decode(tokens, skip_special_tokens=True)
        think_match = _THINK_FULL_RE.search(raw)
        if not think_match:
            think_match = _THINK_LEADING_RE.match(raw)
        if think_match:
            thinking = think_match.group(1).strip()
            response = (raw[: think_match.start()] + raw[think_match.end() :]).lstrip()
            return thinking or None, response

        return None, raw

    def update_conversation_tracking(self, role: str, message: str):
        entry = {"role": role, "content": message}
        if role == "assistant" and self._last_thinking_content:
            entry["thinking"] = self._last_thinking_content
            self._last_thinking_content = None
        self.conversation.append(entry)

    # ------------------------------------------------------------------
    # Serialisation helpers  (deepcopy, get_state)
    # ------------------------------------------------------------------

    def __deepcopy__(self, memo):
        from copy import deepcopy

        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k in ("_model", "_tokenizer"):
                # These are the heavy objects, they shouldn't be deepcopied
                setattr(result, k, None)
            else:
                setattr(result, k, deepcopy(v, memo))
        return result