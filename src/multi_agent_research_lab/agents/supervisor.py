"""Supervisor / router agent.

The supervisor owns the *control plane*: it inspects the shared state and decides which
worker should run next, or whether the workflow is done. It never writes worker content
itself -- keeping orchestration and work cleanly separated.
"""

from __future__ import annotations

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.state import ResearchState

# Routes the supervisor can emit.
ROUTE_RESEARCHER = "researcher"
ROUTE_ANALYST = "analyst"
ROUTE_WRITER = "writer"
ROUTE_CRITIC = "critic"
ROUTE_DONE = "done"

# An agent that fails this many times is skipped (fallback) instead of retried forever.
_MAX_AGENT_FAILURES = 2


class SupervisorAgent(BaseAgent):
    """Decides which worker should run next and when to stop."""

    name = "supervisor"

    def __init__(self, max_iterations: int = 6, enable_critic: bool = True) -> None:
        self.max_iterations = max_iterations
        self.enable_critic = enable_critic

    def decide(self, state: ResearchState) -> str:
        """Pure routing policy -- returns the next route given the current state."""

        # Hard guardrail: never loop forever.
        if state.iteration >= self.max_iterations:
            return ROUTE_DONE

        def failed(agent: str) -> bool:
            return state.failures.get(agent, 0) >= _MAX_AGENT_FAILURES

        # 1. Need evidence first.
        if not state.research_notes and not failed(ROUTE_RESEARCHER):
            return ROUTE_RESEARCHER

        # 2. Turn evidence into structured analysis.
        if state.research_notes and not state.analysis_notes and not failed(ROUTE_ANALYST):
            return ROUTE_ANALYST

        # 3. Synthesize the final answer.
        if not state.final_answer and not failed(ROUTE_WRITER):
            return ROUTE_WRITER

        # 4. Optional bonus review pass once, after a draft exists.
        if (
            self.enable_critic
            and state.final_answer
            and state.critic_review is None
            and not failed(ROUTE_CRITIC)
        ):
            return ROUTE_CRITIC

        # 5. Nothing left to do (or everything that could fail has failed) -> stop.
        return ROUTE_DONE

    def run(self, state: ResearchState) -> ResearchState:
        """Compute the next route, record it on the state, and return it."""

        route = self.decide(state)
        state.next_agent = route
        state.add_trace_event(
            "route_decision",
            {
                "iteration": state.iteration,
                "next": route,
                "has_research": bool(state.research_notes),
                "has_analysis": bool(state.analysis_notes),
                "has_answer": bool(state.final_answer),
                "failures": dict(state.failures),
            },
        )
        return state
