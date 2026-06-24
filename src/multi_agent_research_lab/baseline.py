"""Single-agent baseline.

One agent does the whole job in a single pass: it (optionally) searches, then asks the LLM
to research, analyze, and write in one shot. This is the control group the multi-agent
workflow is benchmarked against.
"""

from __future__ import annotations

from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient
from multi_agent_research_lab.utils.text import count_citation_markers

_SYSTEM_PROMPT = (
    "You are a single research assistant. In one pass, gather context, analyze it, and write "
    "a clear, well-cited answer for the stated audience. Cite sources inline as [n]."
)


def run_single_agent(
    state: ResearchState,
    *,
    settings: Settings | None = None,
    llm: LLMClient | None = None,
    search: SearchClient | None = None,
) -> ResearchState:
    """Execute the single-agent baseline and populate ``state.final_answer``."""

    settings = settings or get_settings()
    llm = llm or LLMClient(settings)
    search = search or SearchClient(settings)

    with trace_span("baseline", {"query": state.request.query}):
        state.sources = search.search(state.request.query, max_results=state.request.max_sources)
        evidence = "\n".join(
            f"[{i + 1}] {doc.title}: {doc.snippet}" for i, doc in enumerate(state.sources)
        )
        user_prompt = (
            f"Query: {state.request.query}\nAudience: {state.request.audience}\n\n"
            f"Sources:\n{evidence}\n\nWrite the full answer with inline [n] citations."
        )
        response = llm.complete(_SYSTEM_PROMPT, user_prompt)
        state.record_usage(response.input_tokens, response.output_tokens, response.cost_usd)

        citations = [
            f"[{i + 1}] {doc.title} — {doc.url or 'no-url'}" for i, doc in enumerate(state.sources)
        ]
        sources_block = "\n".join(citations) if citations else "- (none)"
        state.final_answer = f"{response.content.strip()}\n\n### Sources\n{sources_block}"

        # Coverage measured from the single agent's own citations.
        state.total_claims = len(state.sources)
        state.cited_claims = count_citation_markers(response.content, len(state.sources))
        state.agent_results.append(
            AgentResult(
                agent=AgentName.WRITER,
                content=state.final_answer,
                metadata={"mode": "single"},
            )
        )
    return state
