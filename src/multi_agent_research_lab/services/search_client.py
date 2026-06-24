"""Search client abstraction for the ResearcherAgent.

Search backends, in priority order:

* **Tavily** -- used when ``TAVILY_API_KEY`` is set (imported lazily).
* **DuckDuckGo** (``ddgs``) -- a keyless *real* web search, the default.
* **Local corpus** -- a small synthetic fallback used only if both real backends are
  unavailable (no network / package missing), so a transient outage never hard-fails a run.
  The assignment explicitly permits a local mock source for search.

Tests inject a fake searcher via dependency injection to stay deterministic and offline.
"""

from __future__ import annotations

import importlib
import logging
from typing import Any, cast

from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.errors import LabError
from multi_agent_research_lab.core.schemas import SourceDocument

logger = logging.getLogger(__name__)


class SearchClient:
    """Provider-agnostic real search client (Tavily or DuckDuckGo)."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    @property
    def provider(self) -> str:
        return "tavily" if self._settings.tavily_api_key else "duckduckgo"

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Return real documents relevant to ``query`` (capped at ``max_results``)."""

        if max_results < 1:
            raise ValueError("max_results must be >= 1")

        if self._settings.tavily_api_key:
            try:
                return self._search_tavily(query, max_results)
            except Exception as exc:  # pragma: no cover - network path
                logger.warning("Tavily failed (%s); falling back to DuckDuckGo", exc)
        try:
            return self._search_ddg(query, max_results)
        except Exception as exc:  # pragma: no cover - network path
            logger.warning("Web search failed (%s); using local corpus fallback", exc)
            return self._search_offline(query, max_results)

    def _search_ddg(self, query: str, max_results: int) -> list[SourceDocument]:
        ddgs_mod = cast(Any, importlib.import_module("ddgs"))
        docs: list[SourceDocument] = []
        with ddgs_mod.DDGS() as ddg:
            for item in ddg.text(query, max_results=max_results):
                docs.append(
                    SourceDocument(
                        title=item.get("title", "Untitled"),
                        url=item.get("href"),
                        snippet=item.get("body", ""),
                        metadata={"source": "duckduckgo"},
                    )
                )
        if not docs:
            raise LabError(f"web search returned no results for query: {query!r}")
        return docs[:max_results]

    def _search_offline(self, query: str, max_results: int) -> list[SourceDocument]:
        """Last-resort synthetic corpus (only when real backends are unavailable)."""

        keywords = [w for w in query.replace(",", " ").split() if len(w) > 3][:4]
        topic = " ".join(keywords) or query
        templates = [
            (f"Survey: {topic}", "https://example.org/survey", f"A survey of {topic}."),
            (f"Guide to {topic}", "https://example.org/guide", f"A practitioner guide to {topic}."),
            (f"Case study: {topic}", "https://example.org/case", f"A field report on {topic}."),
            (f"Critique of {topic}", "https://example.org/critique", f"A review of {topic}."),
            (f"Docs for {topic}", "https://example.org/docs", f"Reference docs for {topic}."),
        ]
        return [
            SourceDocument(
                title=t, url=u, snippet=s, metadata={"rank": i + 1, "source": "offline-corpus"}
            )
            for i, (t, u, s) in enumerate(templates)
        ][:max_results]

    def _search_tavily(  # pragma: no cover - network path
        self, query: str, max_results: int
    ) -> list[SourceDocument]:
        tavily_mod = cast(Any, importlib.import_module("tavily"))
        client = tavily_mod.TavilyClient(api_key=self._settings.tavily_api_key)
        response = client.search(query=query, max_results=max_results)
        docs: list[SourceDocument] = []
        for item in response.get("results", [])[:max_results]:
            docs.append(
                SourceDocument(
                    title=item.get("title", "Untitled"),
                    url=item.get("url"),
                    snippet=item.get("content", ""),
                    metadata={"score": item.get("score"), "source": "tavily"},
                )
            )
        if not docs:
            raise LabError(f"Tavily returned no results for query: {query!r}")
        return docs
