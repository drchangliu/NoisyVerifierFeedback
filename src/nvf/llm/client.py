from __future__ import annotations

import json
from dataclasses import dataclass

import anthropic
import openai
import requests


@dataclass
class LLMResponse:
    content: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    model: str = ""
    stop_reason: str = ""  # provider's stop/done reason; "length"/"max_tokens" = truncated


class LLMClient:
    """Unified LLM client for Claude, OpenAI, and Ollama models."""

    # Approximate per-token pricing (USD per 1K tokens)
    PRICING = {
        "claude-sonnet-4-20250514": {"input": 0.003, "output": 0.015},
        "claude-sonnet-4-6": {"input": 0.003, "output": 0.015},
        "claude-haiku-4-5-20251001": {"input": 0.0008, "output": 0.004},
        "claude-opus-4-20250514": {"input": 0.015, "output": 0.075},
        "claude-opus-4-8": {"input": 0.005, "output": 0.025},
        "claude-fable-5": {"input": 0.010, "output": 0.050},
        "gpt-4o": {"input": 0.0025, "output": 0.01},
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    }

    # Models that reject temperature/top_p/top_k with a 400 (Opus 4.7+ and the
    # Claude 5 family). Seed variation for these cohorts comes from the
    # provider's nondeterministic serving, not a temperature setting.
    NO_SAMPLING_PARAMS = ("claude-opus-4-7", "claude-opus-4-8",
                          "claude-fable-5", "claude-sonnet-5", "claude-mythos")

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        temperature: float = 0.2,
        ollama_base_url: str | None = None,
        max_tokens: int = 4096,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        if ollama_base_url:
            self._provider = "ollama"
            self._ollama_base_url = ollama_base_url.rstrip("/")
        elif model.startswith("claude"):
            self._anthropic = anthropic.Anthropic()
            self._provider = "anthropic"
        else:
            self._openai = openai.OpenAI()
            self._provider = "openai"

    def generate(self, messages: list[dict]) -> LLMResponse:
        if self._provider == "anthropic":
            return self._generate_anthropic(messages)
        elif self._provider == "ollama":
            return self._generate_ollama(messages)
        else:
            return self._generate_openai(messages)

    def _generate_anthropic(self, messages: list[dict]) -> LLMResponse:
        system = None
        user_messages = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                user_messages.append(m)

        kwargs = {"model": self.model, "max_tokens": self.max_tokens, "messages": user_messages}
        if system:
            kwargs["system"] = system
        if self.temperature is not None and not self.model.startswith(self.NO_SAMPLING_PARAMS):
            kwargs["temperature"] = self.temperature

        response = self._anthropic.messages.create(**kwargs)

        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        cost = self._compute_cost(input_tokens, output_tokens)

        # Fable 5 responses lead with thinking blocks (empty text) before the
        # text block, and a refusal has an empty content list -- so join the
        # text blocks rather than assume content[0] is text.
        text = "".join(
            b.text for b in response.content if getattr(b, "type", "") == "text"
        )

        return LLMResponse(
            content=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            model=self.model,
            stop_reason=response.stop_reason or "",
        )

    def _generate_openai(self, messages: list[dict]) -> LLMResponse:
        response = self._openai.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        input_tokens = response.usage.prompt_tokens if response.usage else 0
        output_tokens = response.usage.completion_tokens if response.usage else 0
        cost = self._compute_cost(input_tokens, output_tokens)

        return LLMResponse(
            content=response.choices[0].message.content or "",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            model=self.model,
            stop_reason=response.choices[0].finish_reason or "",
        )

    def _generate_ollama(self, messages: list[dict]) -> LLMResponse:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }

        # Verbose reasoning models can exceed a 300s read on long generations;
        # a longer wait changes no result, only avoids a premature kill.
        # Override with OLLAMA_READ_TIMEOUT (seconds).
        import os
        read_timeout = float(os.environ.get("OLLAMA_READ_TIMEOUT", "300"))
        resp = requests.post(
            f"{self._ollama_base_url}/api/chat",
            json=payload,
            timeout=read_timeout,
        )
        resp.raise_for_status()
        data = resp.json()

        content = data.get("message", {}).get("content", "")
        eval_count = data.get("eval_count", 0)
        prompt_eval_count = data.get("prompt_eval_count", 0)

        return LLMResponse(
            content=content,
            input_tokens=prompt_eval_count,
            output_tokens=eval_count,
            cost_usd=0.0,  # Local models are free
            model=self.model,
            stop_reason=data.get("done_reason", "") or "",
        )

    def _compute_cost(self, input_tokens: int, output_tokens: int) -> float:
        pricing = self.PRICING.get(self.model, {"input": 0.003, "output": 0.015})
        return (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1000
