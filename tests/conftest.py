"""Shared test fixtures.

Production code makes real LLM calls (no built-in mock). Tests stay fast and deterministic by
injecting this ``FakeLLM`` test-double through the agents'/workflow's constructor (dependency
injection), so no network or API key is needed to test the orchestration logic.
"""

from __future__ import annotations

import pytest

from multi_agent_research_lab.core.schemas import SourceDocument
from multi_agent_research_lab.services.llm_client import LLMResponse


class FakeSearch:
    """Deterministic stand-in for ``SearchClient`` (no network) used only in tests."""

    def __init__(self, count: int = 5) -> None:
        self.count = count

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        n = min(self.count, max_results)
        return [
            SourceDocument(
                title=f"Doc {i + 1}",
                url=f"https://example.test/{i + 1}",
                snippet=f"Snippet {i + 1} about {query}.",
            )
            for i in range(n)
        ]


class FakeLLM:
    """Deterministic stand-in for ``LLMClient`` used only in tests."""

    def __init__(self, body: str | None = None) -> None:
        self.calls = 0
        self.body = body or (
            "Synthesized test answer. " * 20 + "Key points cited [1] [2] [3] [4] [5]."
        )

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        self.calls += 1
        return LLMResponse(content=self.body, input_tokens=100, output_tokens=80, cost_usd=0.0)


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def fake_search() -> FakeSearch:
    return FakeSearch()
