"""Writer agent.

Responsibility: synthesize research + analysis into a clear final answer for the target
audience, with an inline citation map back to the gathered sources. It records how many
claims it actually cited so the benchmark can compute citation coverage.
"""

from __future__ import annotations

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.utils.text import count_citation_markers

_SYSTEM_PROMPT = (
    "You are a precise Writer agent. Synthesize the research and analysis into a clear, "
    "well-structured answer for the stated audience. Cite sources inline as [n] and end with "
    "a 'Sources' list."
)


class WriterAgent(BaseAgent):
    """Produces the final answer from research and analysis notes."""

    name = "writer"

    def __init__(self, llm: LLMClient | None = None) -> None:
        self.llm = llm or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate ``state.final_answer`` and set ``state.cited_claims``."""

        context = "\n\n".join(
            part
            for part in (state.research_notes, state.analysis_notes)
            if part
        )
        user_prompt = (
            f"Query: {state.request.query}\n"
            f"Audience: {state.request.audience}\n\n"
            f"{context}\n\n"
            "Write the final answer (~300-500 words). Use inline [n] citations and a Sources list."
        )
        response = self.llm.complete(_SYSTEM_PROMPT, user_prompt)
        state.record_usage(response.input_tokens, response.output_tokens, response.cost_usd)

        # The body is the LLM's answer; we append a canonical Sources map so every [n] marker
        # resolves to an actual gathered document (verifiable citations, not invented ones).
        citations = [
            f"[{i + 1}] {doc.title} — {doc.url or 'no-url'}"
            for i, doc in enumerate(state.sources)
        ]
        sources_block = "\n".join(citations) if citations else "- (no sources)"
        state.final_answer = f"{response.content.strip()}\n\n### Sources\n{sources_block}"

        # Coverage = how many of the available sources the LLM actually cited in its answer.
        state.cited_claims = count_citation_markers(response.content, len(state.sources))

        state.agent_results.append(
            AgentResult(
                agent=AgentName.WRITER,
                content=state.final_answer,
                metadata={"cited_claims": state.cited_claims},
            )
        )
        state.add_trace_event(
            "writer",
            {"answer_chars": len(state.final_answer), "cited_claims": state.cited_claims},
        )
        return state
