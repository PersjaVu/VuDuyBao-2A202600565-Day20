"""Analyst agent.

Responsibility: turn raw research notes into structured insight -- extract key claims,
compare viewpoints, and flag weak/uncited evidence. It also records ``total_claims`` so the
benchmark can measure citation coverage downstream.
"""

from __future__ import annotations

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

_SYSTEM_PROMPT = (
    "You are a rigorous Analyst agent. Extract the key claims from research notes, compare "
    "competing viewpoints, and explicitly flag any claim with weak or missing evidence."
)


class AnalystAgent(BaseAgent):
    """Turns research notes into structured insights."""

    name = "analyst"

    def __init__(self, llm: LLMClient | None = None) -> None:
        self.llm = llm or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate ``state.analysis_notes`` and set ``state.total_claims``."""

        if not state.research_notes:
            raise ValueError("AnalystAgent requires research_notes before it can run")

        user_prompt = (
            f"Query: {state.request.query}\n\n"
            f"{state.research_notes}\n\n"
            "Produce: (a) 3-5 key claims each tagged with its source id, "
            "(b) one line on agreements/disagreements across sources, "
            "(c) a short 'weak evidence' flag list."
        )
        response = self.llm.complete(_SYSTEM_PROMPT, user_prompt)
        state.record_usage(response.input_tokens, response.output_tokens, response.cost_usd)

        # The analysis is the LLM's own structured output. ``total_claims`` is the number of
        # gathered sources the writer can cite -- the denominator for citation coverage.
        state.analysis_notes = response.content
        state.total_claims = len(state.sources)

        state.agent_results.append(
            AgentResult(
                agent=AgentName.ANALYST,
                content=state.analysis_notes,
                metadata={"total_claims": state.total_claims},
            )
        )
        state.add_trace_event("analyst", {"total_claims": state.total_claims})
        return state
