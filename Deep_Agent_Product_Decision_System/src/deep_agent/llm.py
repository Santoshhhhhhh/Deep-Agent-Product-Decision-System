"""LLM abstraction with two backends.

- ``anthropic``: real Claude API calls (structured JSON output enforced
  by prompt + Pydantic re-validation with bounded retry).
- ``mock``: a deterministic heuristic engine that emulates each agent's
  reasoning, so the full multi-agent pipeline and the evaluation harness
  run offline with zero API cost. This is also what unit tests use.

Every structured call goes through ``structured_call`` which validates
the response against a Pydantic model and retries with the validation
error appended — the same bounded self-correction pattern the critic
loop uses at the graph level.
"""
from __future__ import annotations

import json
import os
import re
from typing import Type, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

DEFAULT_MODEL = os.environ.get("DEEP_AGENT_MODEL", "claude-sonnet-4-5")


class LLMError(RuntimeError):
    pass


class LLMClient:
    """mode: 'mock' (default, offline) or 'anthropic' (needs ANTHROPIC_API_KEY)."""

    def __init__(self, mode: str | None = None, model: str = DEFAULT_MODEL) -> None:
        self.mode = mode or os.environ.get("DEEP_AGENT_LLM", "mock")
        self.model = model
        self._client = None
        if self.mode == "anthropic":
            try:
                import anthropic  # lazy import
            except ImportError as e:
                raise LLMError("pip install anthropic to use anthropic mode") from e
            self._client = anthropic.Anthropic()

    # ------------------------------------------------------------------ API
    def complete(self, system: str, user: str, max_tokens: int = 1500) -> str:
        if self.mode == "anthropic":
            resp = self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return "".join(b.text for b in resp.content if b.type == "text")
        return self._mock_complete(system, user)

    def structured_call(
        self,
        system: str,
        user: str,
        response_model: Type[T],
        max_retries: int = 2,
    ) -> T:
        """Call the LLM and validate JSON output against ``response_model``.

        On validation failure, re-prompts with the error message appended
        (bounded self-correction at the call level).
        """
        schema = json.dumps(response_model.model_json_schema(), indent=2)
        sys_prompt = (
            f"{system}\n\nRespond ONLY with a JSON object matching this schema "
            f"(no markdown fences, no prose):\n{schema}"
        )
        prompt = user
        last_err: Exception | None = None
        for _ in range(max_retries + 1):
            raw = self.complete(sys_prompt, prompt)
            try:
                return response_model.model_validate_json(_extract_json(raw))
            except (ValidationError, ValueError) as e:
                last_err = e
                prompt = (
                    f"{user}\n\nYour previous output failed validation:\n{e}\n"
                    "Return corrected JSON only."
                )
        raise LLMError(f"Structured call failed after retries: {last_err}")

    # ----------------------------------------------------------------- mock
    def _mock_complete(self, system: str, user: str) -> str:
        """Deterministic emulation keyed off the agent role in the system prompt."""
        from .mock_brain import mock_route  # lazy to avoid cycles
        return mock_route(system, user)


def _extract_json(text: str) -> str:
    """Tolerate models that wrap JSON in markdown fences or prose."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text
