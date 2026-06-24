"""Service client tests."""

import pytest

from multi_agent_research_lab.core.config import Settings
from multi_agent_research_lab.core.errors import LabError
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient


def test_llm_requires_credentials() -> None:
    client = LLMClient(Settings(OPENROUTER_API_KEY=None, OPENAI_API_KEY=None))
    assert client.is_live is False
    assert client.provider == "none"
    with pytest.raises(LabError):
        client.complete("system", "user")


def test_llm_prefers_openrouter() -> None:
    client = LLMClient(Settings(OPENROUTER_API_KEY="key", OPENAI_API_KEY="other"))
    assert client.provider == "openrouter"
    assert client.is_live is True
    assert client.model == "openai/gpt-oss-120b:free"


def test_llm_falls_back_to_openai_when_no_openrouter() -> None:
    client = LLMClient(Settings(OPENROUTER_API_KEY=None, OPENAI_API_KEY="key"))
    assert client.provider == "openai"
    assert client.model == "gpt-4o-mini"


def test_search_provider_selection() -> None:
    assert SearchClient(Settings(TAVILY_API_KEY="k")).provider == "tavily"
    assert SearchClient(Settings(TAVILY_API_KEY=None)).provider == "duckduckgo"


def test_search_rejects_bad_max_results() -> None:
    client = SearchClient(Settings(TAVILY_API_KEY=None))
    with pytest.raises(ValueError):
        client.search("query", max_results=0)


def test_search_offline_fallback_is_capped() -> None:
    # The last-resort corpus is exercised directly (no network).
    client = SearchClient(Settings(TAVILY_API_KEY=None))
    docs = client._search_offline("multi-agent systems benchmark", max_results=3)
    assert len(docs) == 3
    assert all(d.snippet for d in docs)
