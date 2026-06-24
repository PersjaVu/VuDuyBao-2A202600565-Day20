"""LLM client abstraction.

Production note: agents depend on this interface instead of importing an SDK directly.
Retry, timeout, token logging, and cost estimation live here -- not inside agents.

This client makes **real** LLM calls only. It speaks the OpenAI Chat Completions API, which
also covers OpenAI-compatible gateways like **OpenRouter** (just a different ``base_url`` and
key). There is no offline/mock fallback: if no key is configured, ``complete`` raises so the
failure is loud and obvious instead of silently returning fake text. Tests inject a fake
client via constructor dependency injection rather than relying on a built-in mock.
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass
from typing import Any, cast

from tenacity import retry, stop_after_attempt, wait_exponential

from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.errors import LabError
from multi_agent_research_lab.observability.tracing import trace_span

logger = logging.getLogger(__name__)

# Approximate USD pricing per 1M tokens. Free OpenRouter models cost 0; the default OpenAI
# pricing is kept as a fallback so cost reporting still works for paid models.
_PRICE_PER_1M = {
    "default": {"input": 0.15, "output": 0.60},
    "free": {"input": 0.0, "output": 0.0},
}


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


def _estimate_tokens(text: str) -> int:
    """Cheap token estimate (~4 chars/token), used only if the provider omits usage."""

    return max(1, len(text) // 4)


def _estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    table = _PRICE_PER_1M["free"] if ":free" in model else _PRICE_PER_1M["default"]
    return (input_tokens * table["input"] + output_tokens * table["output"]) / 1_000_000


class LLMClient:
    """Real LLM client for OpenAI / OpenRouter (OpenAI-compatible)."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    @property
    def provider(self) -> str:
        if self._settings.openrouter_api_key:
            return "openrouter"
        if self._settings.openai_api_key:
            return "openai"
        return "none"

    @property
    def is_live(self) -> bool:
        return self.provider != "none"

    @property
    def model(self) -> str:
        if self.provider == "openrouter":
            return self._settings.openrouter_model
        return self._settings.openai_model

    def _credentials(self) -> tuple[str, str | None]:
        """Return (api_key, base_url) for the active provider."""

        if self.provider == "openrouter":
            return self._settings.openrouter_api_key or "", self._settings.openrouter_base_url
        return self._settings.openai_api_key or "", None

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Return a real model completion with retry, timeout, and cost accounting."""

        if not self.is_live:
            raise LabError(
                "No LLM credentials configured. Set OPENROUTER_API_KEY (recommended) or "
                "OPENAI_API_KEY in your environment / .env, and install the 'llm' extra."
            )
        return self._complete_live(system_prompt, user_prompt)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.5, max=4), reraise=True)
    def _complete_live(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        try:
            openai_mod = cast(Any, importlib.import_module("openai"))
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise LabError(
                "The 'openai' package is required for live calls. Install with: "
                'pip install -e ".[llm]"'
            ) from exc

        api_key, base_url = self._credentials()
        client = openai_mod.OpenAI(api_key=api_key, base_url=base_url)
        with trace_span(
            f"llm:{self.model}",
            {"provider": self.provider, "system": system_prompt[:200], "user": user_prompt[:200]},
            run_type="llm",
        ):
            completion = client.chat.completions.create(
                model=self.model,
                temperature=0.2,
                timeout=self._settings.timeout_seconds,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
        content = completion.choices[0].message.content or ""
        usage = getattr(completion, "usage", None)
        in_tok = getattr(usage, "prompt_tokens", None) or _estimate_tokens(
            system_prompt + user_prompt
        )
        out_tok = getattr(usage, "completion_tokens", None) or _estimate_tokens(content)
        logger.info(
            "llm call provider=%s model=%s tokens=%s/%s",
            self.provider,
            self.model,
            in_tok,
            out_tok,
        )
        return LLMResponse(
            content=content,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=_estimate_cost(self.model, in_tok, out_tok),
        )
