"""Unified LLM API layer.

Replaces the per-vendor classes scattered across PRE (OpenAI/Claude/Baidu/GLM)
and Auto-PRE (+ local transformers models) with a single client that talks to
any OpenAI-compatible endpoint. Every modern provider (OpenAI, Anthropic via
proxy, vLLM, Together, DeepSeek, local servers, ...) exposes this interface, so
one class covers them all. The user supplies only base_url + api_key + model.
"""

from __future__ import annotations

import math
import time
import traceback
from dataclasses import dataclass, field
from typing import Optional

from openai import OpenAI


@dataclass
class APIConfig:
    """Configuration for a single model endpoint, parsed from YAML."""

    model: str
    api_key: str
    base_url: Optional[str] = None
    # "evaluatee" generates task answers, "evaluator" judges, "both" does both.
    role: str = "both"
    max_tries: int = 5
    temperature: float = 0.0
    extra: dict = field(default_factory=dict)

    @property
    def model_name(self) -> str:
        # Stable identifier used in checkpoint filenames and result keys.
        return self.extra.get("name", self.model)

    @property
    def is_evaluatee(self) -> bool:
        return self.role in ("evaluatee", "both")

    @property
    def is_evaluator(self) -> bool:
        return self.role in ("evaluator", "both")


class LLMClient:
    """Thin, retrying wrapper around an OpenAI-compatible chat endpoint."""

    def __init__(self, config: APIConfig) -> None:
        self.config = config
        self.model_name = config.model_name
        self._client = OpenAI(api_key=config.api_key, base_url=config.base_url)

    # -- plain chat ---------------------------------------------------------
    def chat(self, prompt: str, system: str = "You are a helpful assistant.") -> Optional[str]:
        """Return the model's text reply, or None after exhausting retries."""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        for attempt in range(self.config.max_tries):
            try:
                resp = self._client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    temperature=self.config.temperature,
                )
                content = resp.choices[0].message.content
                if content is not None:
                    return content.strip()
            except Exception:
                traceback.print_exc()
                time.sleep(min(2 ** attempt, 30))
        return None

    # -- chat with token logprobs (for CalibraEval debiasing) ---------------
    def chat_with_logprobs(
        self, prompt: str, target_tokens=("A", "B"), top_logprobs: int = 5
    ) -> Optional[dict]:
        """Return normalized probabilities over ``target_tokens`` at the first
        generated token, plus the raw text.

        Returns None if the endpoint does not support logprobs (so callers can
        cleanly fall back to text-only judging). Output shape::

            {"text": str, "probs": {"A": 0.7, "B": 0.3}}
        """
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        for attempt in range(self.config.max_tries):
            try:
                resp = self._client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    temperature=self.config.temperature,
                    logprobs=True,
                    top_logprobs=top_logprobs,
                )
                choice = resp.choices[0]
                if choice.logprobs is None or not choice.logprobs.content:
                    return None
                # Probability mass on each target token at the first position.
                first = choice.logprobs.content[0].top_logprobs
                raw = {t: 0.0 for t in target_tokens}
                for cand in first:
                    tok = cand.token.strip()
                    for t in target_tokens:
                        if tok == t:
                            raw[t] = math.exp(cand.logprob)
                total = sum(raw.values())
                if total <= 0:
                    return None
                probs = {t: raw[t] / total for t in target_tokens}
                text = (choice.message.content or "").strip()
                return {"text": text, "probs": probs}
            except Exception:
                traceback.print_exc()
                time.sleep(min(2 ** attempt, 30))
        return None


def build_clients(api_configs):
    """Instantiate LLMClient objects from a list of dict / APIConfig."""
    clients = []
    for cfg in api_configs:
        if isinstance(cfg, APIConfig):
            clients.append(LLMClient(cfg))
        else:
            known = {"model", "api_key", "base_url", "role", "max_tries", "temperature"}
            extra = {k: v for k, v in cfg.items() if k not in known}
            clients.append(
                LLMClient(
                    APIConfig(
                        model=cfg["model"],
                        api_key=cfg["api_key"],
                        base_url=cfg.get("base_url"),
                        role=cfg.get("role", "both"),
                        max_tries=int(cfg.get("max_tries", 5)),
                        temperature=float(cfg.get("temperature", 0.0)),
                        extra=extra,
                    )
                )
            )
    return clients
