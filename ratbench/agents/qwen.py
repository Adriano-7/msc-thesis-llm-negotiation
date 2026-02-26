# ratbench/agents/qwen.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from ratbench.agents.agents import Agent
import time
from ratbench.constants import AGENT_ONE, AGENT_TWO
import os

class QwenAgent(Agent):
    def __init__(
        self,
        model_id="Qwen/Qwen2.5-7B-Instruct", # Note: Qwen3 is not out yet, assuming Qwen2.5 or 2 based on your link structure, but using the variable for flexibility
        max_new_tokens=400,
        temperature=0.7,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.run_epoch_time_ms = str(round(time.time() * 1000))
        self.model_id = model_id
        self.conversation = []
        self.prompt_entity_initializer = "system"
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        # Load Model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype="auto",
            device_map="auto"
        )

    def __deepcopy__(self, memo):
        from copy import deepcopy
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k in ("model", "tokenizer"):
                v = v.__class__.__name__
            setattr(result, k, deepcopy(v, memo))
        return result

    def init_agent(self, system_prompt, role):
        if AGENT_ONE in self.agent_name:
            self.conversation.append({"role": "system", "content": system_prompt})
            self.conversation.append({"role": "user", "content": role})
        elif AGENT_TWO in self.agent_name:
            self.conversation.append({"role": "system", "content": system_prompt + role})
        else:
            raise ValueError("No Player 1 or Player 2 in role")

    def chat(self):
        text = self.tokenizer.apply_chat_template(
            self.conversation,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature
        )
        
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response

    def update_conversation_tracking(self, role, message):
        self.conversation.append({"role": role, "content": message})