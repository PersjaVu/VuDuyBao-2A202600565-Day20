"""Researcher agent.

Responsibility: gather sources for the query, filter/deduplicate them, capture citations,
and write concise research notes the analyst can build on.
"""

from __future__ import annotations

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient

_SYSTEM_PROMPT = (
    "You are a meticulous Researcher agent. Collect relevant, credible sources and write "
    "tight, factual research notes. Always attribute facts to numbered sources."
)


class ResearcherAgent(BaseAgent):
    """Collects sources and creates concise research notes."""

    name = "researcher"

    def __init__(self, search: SearchClient | None = None, llm: LLMClient | None = None) -> None:
        self.search = search or SearchClient()
        self.llm = llm or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate ``state.sources`` and ``state.research_notes``."""

        results = self.search.search(state.request.query, max_results=state.request.max_sources)

        # Filter: drop empty snippets and deduplicate by URL/title.
        seen: set[str] = set()
        filtered = []
        for doc in results:
            key = (doc.url or doc.title).strip().lower()
            if not doc.snippet.strip() or key in seen:
                continue
            seen.add(key)
            filtered.append(doc)
        state.sources = filtered

        # Build a numbered evidence digest the writer can cite as [1], [2], ...
        evidence = "\n".join(
            f"[{i + 1}] {doc.title} ({doc.url or 'no-url'}): {doc.snippet}"
            for i, doc in enumerate(filtered)
        )
        user_prompt = (
            f"Query: {state.request.query}\n"
            f"Audience: {state.request.audience}\n\n"
            f"Sources:\n{evidence}\n\n"
            "Write 4-6 bullet research notes. Tag every bullet with its source id like [1]."
        )
        response = self.llm.complete(_SYSTEM_PROMPT, user_prompt)
        state.record_usage(response.input_tokens, response.output_tokens, response.cost_usd)

        # The research notes are the LLM's own output, grounded in the gathered sources.
        state.research_notes = response.content

        state.agent_results.append(
            AgentResult(
                agent=AgentName.RESEARCHER,
                content=state.research_notes,
                metadata={"num_sources": len(filtered)},
            )
        )
        state.add_trace_event("researcher", {"num_sources": len(filtered)})
        return state
