"""Shared state for the multi-agent workflow.

The state is the single source of truth handed off between agents. It is intentionally
rich so that any agent (or a human debugging a trace) can reconstruct *what happened*,
*who did it*, *how much it cost*, and *what is still missing*.
"""

from typing import Any

from pydantic import BaseModel, Field

from multi_agent_research_lab.core.schemas import AgentResult, ResearchQuery, SourceDocument


class ResearchState(BaseModel):
    """Single source of truth passed through the workflow."""

    request: ResearchQuery

    # --- routing / control plane -------------------------------------------------
    iteration: int = 0
    route_history: list[str] = Field(default_factory=list)
    next_agent: str | None = None
    failures: dict[str, int] = Field(default_factory=dict)

    # --- worker outputs ----------------------------------------------------------
    sources: list[SourceDocument] = Field(default_factory=list)
    research_notes: str | None = None
    analysis_notes: str | None = None
    final_answer: str | None = None
    critic_review: str | None = None

    # --- evaluation signals ------------------------------------------------------
    total_claims: int = 0
    cited_claims: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    estimated_cost_usd: float = 0.0

    # --- bookkeeping -------------------------------------------------------------
    agent_results: list[AgentResult] = Field(default_factory=list)
    trace: list[dict[str, Any]] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

    def record_route(self, route: str) -> None:
        """Record the chosen route and advance the iteration counter."""

        self.route_history.append(route)
        self.iteration += 1

    def record_failure(self, agent: str) -> int:
        """Increment and return the failure count for an agent."""

        self.failures[agent] = self.failures.get(agent, 0) + 1
        return self.failures[agent]

    def record_usage(
        self,
        input_tokens: int | None,
        output_tokens: int | None,
        cost_usd: float | None,
    ) -> None:
        """Accumulate token usage and estimated cost from an LLM call."""

        self.total_input_tokens += input_tokens or 0
        self.total_output_tokens += output_tokens or 0
        self.estimated_cost_usd += cost_usd or 0.0

    def add_trace_event(self, name: str, payload: dict[str, Any]) -> None:
        self.trace.append({"name": name, "payload": payload})

    @property
    def citation_coverage(self) -> float:
        """Fraction of key claims that are backed by a cited source (0..1)."""

        if self.total_claims <= 0:
            return 0.0
        return min(1.0, self.cited_claims / self.total_claims)

    @property
    def is_complete(self) -> bool:
        """True once a final answer exists."""

        return bool(self.final_answer)
