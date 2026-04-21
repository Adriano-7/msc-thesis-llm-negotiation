import json
import os
import re
import secrets
import sys
import time
from copy import deepcopy

import requests

from ratbench.agents.agents import Agent
from ratbench.agents.agent_behaviours import SelfCheckingAgent, SelfRefineAgent
from ratbench.constants import AGENT_ONE, AGENT_TWO


class IaEduAgent(Agent):
    """Hosted chatbot agent backed by the iaedu streaming API.

    The API exposes a single `message` string field per call, so the full
    conversation (system + user/assistant turns) is flattened into that field
    on every turn — mirroring how ChatGPTAgent re-sends full history.
    """

    DEFAULT_ENDPOINT = (
        "https://api.iaedu.pt/agent-chat//api/v1/agent/"
        "cmamvd3n40000c801qeacoad2/stream"
    )

    def __init__(
        self,
        model="gpt-4o-iaedu",
        endpoint=None,
        api_key=None,
        channel_id=None,
        timeout=120,
        rate_limit_delay=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.run_epoch_time_ms = str(round(time.time() * 1000))
        self.model = model
        self.endpoint = endpoint or os.environ.get("IAEDU_ENDPOINT", self.DEFAULT_ENDPOINT)
        self.api_key = api_key or os.environ.get("IAEDU_API_KEY")
        self.channel_id = channel_id or os.environ.get("IAEDU_CHANNEL_ID")
        if not self.api_key or not self.channel_id:
            raise RuntimeError(
                "IaEduAgent requires IAEDU_API_KEY and IAEDU_CHANNEL_ID "
                "(set as env vars or passed as kwargs)."
            )
        self.thread_id = secrets.token_urlsafe(16)
        self.conversation = []
        self.prompt_entity_initializer = "system"
        self.timeout = timeout
        self.rate_limit_delay = float(rate_limit_delay) if rate_limit_delay else 0.0
        self._debug_stream = os.environ.get("IAEDU_DEBUG_STREAM") == "1"

    def init_agent(self, system_prompt, role):
        if AGENT_ONE in self.agent_name:
            self.update_conversation_tracking(
                self.prompt_entity_initializer, system_prompt
            )
            self.update_conversation_tracking("user", role)
        elif AGENT_TWO in self.agent_name:
            self.update_conversation_tracking(
                self.prompt_entity_initializer, system_prompt + role
            )
        else:
            raise ValueError(f"Agent name must contain {AGENT_ONE} or {AGENT_TWO}")

    def update_conversation_tracking(self, role, message):
        self.conversation.append({"role": role, "content": message})

    def chat(self):
        if self.rate_limit_delay:
            time.sleep(self.rate_limit_delay)
        data = {
            "channel_id": self.channel_id,
            "thread_id": self.thread_id,
            "user_info": "{}",
            "message": self._flatten_conversation(),
        }
        headers = {"x-api-key": self.api_key}
        resp = requests.post(
            self.endpoint,
            files={k: (None, v) for k, v in data.items()},
            headers=headers,
            timeout=self.timeout,
            stream=True,
        )
        resp.raise_for_status()
        return self._consume_stream(resp)

    def _flatten_conversation(self):
        parts = []
        for msg in self.conversation:
            role = msg["role"].upper()
            parts.append(f"[{role}]\n{msg['content']}")
        parts.append("[ASSISTANT]\n")
        return "\n\n".join(parts)

    def _consume_stream(self, resp):
        chunks = []
        parsed_any = False
        for raw in resp.iter_lines(decode_unicode=True):
            if not raw:
                continue
            if self._debug_stream:
                sys.stderr.write(f"[iaedu stream] {raw}\n")
            line = raw.lstrip()
            if line.startswith("data:"):
                line = line[len("data:"):].lstrip()
            if not line or line == "[DONE]":
                continue
            try:
                payload = json.loads(line)
            except ValueError:
                chunks.append(line)
                continue
            parsed_any = True
            self._check_for_error(payload)
            chunks.append(self._extract_text(payload))
        text = "".join(c for c in chunks if c).strip()
        if not text and not parsed_any:
            text = resp.text.strip()
        self._check_text_for_error(text)
        return text

    @staticmethod
    def _check_for_error(payload):
        if not isinstance(payload, dict):
            return
        err = payload.get("error")
        if err:
            msg = err if isinstance(err, str) else json.dumps(err)
            raise RuntimeError(f"iaedu API error: {msg}")
        status = payload.get("status")
        if isinstance(status, str) and status.lower() == "error":
            msg = payload.get("message") or payload.get("error") or json.dumps(payload)
            raise RuntimeError(f"iaedu API error: {msg}")

    _INLINE_ERROR_RE = re.compile(r"rate limit|\((?:429|5\d\d)\)", re.IGNORECASE)

    @classmethod
    def _check_text_for_error(cls, text):
        # Catch error messages the API streams inline without an `error` key
        # (e.g. "Rate limit reached (429)" showing up as plain content).
        if text and cls._INLINE_ERROR_RE.search(text):
            raise RuntimeError(f"iaedu API error (inline): {text[:200]}")

    @staticmethod
    def _extract_text(payload):
        if isinstance(payload, str):
            return payload
        if not isinstance(payload, dict):
            return ""
        for key in ("content", "delta", "text", "message", "token", "output"):
            val = payload.get(key)
            if isinstance(val, str):
                return val
            if isinstance(val, dict):
                nested = IaEduAgent._extract_text(val)
                if nested:
                    return nested
        choices = payload.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                for key in ("delta", "message"):
                    sub = first.get(key)
                    if isinstance(sub, dict):
                        nested = IaEduAgent._extract_text(sub)
                        if nested:
                            return nested
                if isinstance(first.get("text"), str):
                    return first["text"]
        return ""

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result


class SelfCheckingIaEduAgent(IaEduAgent, SelfCheckingAgent):
    pass


class SelfRefineIaEduAgent(IaEduAgent, SelfRefineAgent):
    pass
