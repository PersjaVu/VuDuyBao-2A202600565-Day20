"""Critic agent (bonus).

Responsibility: an optional fact-check / safety review pass over the final answer. It checks
citation coverage, looks for obvious hallucination markers, and appends a structured review.
It is non-destructive: it annotates rather than rewrites the answer.
"""

from __future__ import annotations

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

_SYSTEM_PROMPT = (
    "You are a skeptical Critic agent. Verify that claims are cited, flag unsupported "
    "statements, and rate the answer's reliability."
)

# Words that often signal fabricated specificity without a citation.
_HALLUCINATION_MARKERS = ("guarantee", "100%", "always", "never fails", "proven fact")


class CriticAgent(BaseAgent):
    """Optional fact-checking and safety-review agent."""

    name = "critic"

    def __init__(self, llm: LLMClient | None = None) -> None:
        self.llm = llm or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Validate the final answer and append findings to ``state.critic_review``."""

        if not state.final_answer:
            raise ValueError("CriticAgent requires a final_answer to review")

        answer = state.final_answer
        coverage = state.citation_coverage
        has_sources_section = "Sources" in answer
        flagged = [m for m in _HALLUCINATION_MARKERS if m in answer.lower()]

        user_prompt = (
            f"Review this answer for the query '{state.request.query}'. "
            "Check citation coverage and unsupported claims.\n\n" + answer
        )
        response = self.llm.complete(_SYSTEM_PROMPT, user_prompt)
        state.record_usage(response.input_tokens, response.output_tokens, response.cost_usd)

        verdict = "PASS" if coverage >= 0.5 and has_sources_section and not flagged else "REVIEW"
        markers = (
            f"Hallucination markers: {', '.join(flagged)}"
            if flagged
            else "Hallucination markers: none detected"
        )
        # Deterministic checks (auditable) + the LLM's qualitative review.
        review_lines = [
            f"Verdict: {verdict}",
            f"Citation coverage: {coverage:.0%} ({state.cited_claims}/{state.total_claims} claims)",
            f"Sources section present: {has_sources_section}",
            markers,
            "",
            "LLM review:",
            response.content.strip(),
        ]
        state.critic_review = "\n".join(review_lines)

        state.agent_results.append(
            AgentResult(
                agent=AgentName.CRITIC,
                content=state.critic_review,
                metadata={"verdict": verdict, "coverage": coverage},
            )
        )
        state.add_trace_event("critic", {"verdict": verdict, "coverage": coverage})
        return state
