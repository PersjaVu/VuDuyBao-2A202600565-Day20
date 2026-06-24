"""Multi-agent workflow orchestration.

Orchestration lives here; agent internals live in ``agents/``. The supervisor decides the
route, the workflow executes the chosen worker, and guardrails (max iterations, wall-clock
timeout, per-agent retry/fallback) keep the loop bounded.

A real ``LangGraph`` ``StateGraph`` is built in :meth:`build` *when langgraph is installed*
(imported lazily so it is never a hard dependency). The reproducible execution path in
:meth:`run` uses a built-in router engine that mirrors the same node/conditional-edge model,
so the lab runs end-to-end offline and in CI without the optional extra.
"""

from __future__ import annotations

import importlib
import logging
from time import perf_counter
from typing import Any, cast

from multi_agent_research_lab.agents.analyst import AnalystAgent
from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.agents.critic import CriticAgent
from multi_agent_research_lab.agents.researcher import ResearcherAgent
from multi_agent_research_lab.agents.supervisor import ROUTE_DONE, SupervisorAgent
from multi_agent_research_lab.agents.writer import WriterAgent
from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient

logger = logging.getLogger(__name__)


class MultiAgentWorkflow:
    """Builds and runs the multi-agent graph."""

    def __init__(
        self,
        settings: Settings | None = None,
        llm: LLMClient | None = None,
        search: SearchClient | None = None,
        enable_critic: bool = True,
    ) -> None:
        self.settings = settings or get_settings()
        self.llm = llm or LLMClient(self.settings)
        self.search = search or SearchClient(self.settings)
        self.enable_critic = enable_critic

        self.supervisor = SupervisorAgent(
            max_iterations=self.settings.max_iterations,
            enable_critic=enable_critic,
        )
        self.workers: dict[str, BaseAgent] = {
            "researcher": ResearcherAgent(self.search, self.llm),
            "analyst": AnalystAgent(self.llm),
            "writer": WriterAgent(self.llm),
            "critic": CriticAgent(self.llm),
        }

    def build(self) -> object:
        """Create a LangGraph ``StateGraph`` when available, else a portable spec.

        Returning a spec (rather than raising) keeps the workflow runnable without the
        optional ``llm`` extra while still exercising the LangGraph API when it is present.
        """

        try:
            langgraph_mod = cast(Any, importlib.import_module("langgraph.graph"))
        except Exception:  # pragma: no cover - exercised only without langgraph
            return {
                "engine": "builtin-router",
                "nodes": ["supervisor", *self.workers.keys()],
                "entry": "supervisor",
                "note": "Install the 'llm' extra to compile a real LangGraph StateGraph.",
            }

        graph = langgraph_mod.StateGraph(dict)  # pragma: no cover - requires langgraph

        def supervisor_node(payload: dict[str, Any]) -> dict[str, Any]:  # pragma: no cover
            state = ResearchState.model_validate(payload)
            return self.supervisor.run(state).model_dump()

        graph.add_node("supervisor", supervisor_node)
        for name, agent in self.workers.items():  # pragma: no cover - requires langgraph

            def make_node(a: BaseAgent) -> Any:
                def node(payload: dict[str, Any]) -> dict[str, Any]:
                    return a.run(ResearchState.model_validate(payload)).model_dump()

                return node

            graph.add_node(name, make_node(agent))
            graph.add_edge(name, "supervisor")

        graph.set_entry_point("supervisor")  # pragma: no cover - requires langgraph
        graph.add_conditional_edges(
            "supervisor",
            lambda payload: payload.get("next_agent", ROUTE_DONE),
            {**{n: n for n in self.workers}, ROUTE_DONE: langgraph_mod.END},
        )
        return graph.compile()

    def run(self, state: ResearchState) -> ResearchState:
        """Execute the routed workflow with full guardrails and return the final state."""

        deadline = perf_counter() + self.settings.timeout_seconds

        with trace_span("workflow", {"query": state.request.query}) as span:
            while True:
                self.supervisor.run(state)
                route = state.next_agent or ROUTE_DONE
                if route == ROUTE_DONE:
                    break

                # Guardrail: wall-clock timeout. Only abort if there is still work to do, so a
                # run that finishes naturally just after the deadline is not falsely flagged.
                if perf_counter() > deadline:
                    state.errors.append("timeout: workflow exceeded timeout_seconds")
                    state.add_trace_event("timeout", {"iteration": state.iteration})
                    break

                state.record_route(route)
                agent = self.workers[route]
                try:
                    with trace_span(f"agent:{route}", {"iteration": state.iteration}):
                        agent.run(state)
                except Exception as exc:
                    # Guardrail: per-agent retry/fallback. Record the failure; the
                    # supervisor will route around an agent that keeps failing.
                    count = state.record_failure(route)
                    msg = f"{route} failed (attempt {count}): {exc}"
                    state.errors.append(msg)
                    state.add_trace_event("agent_error", {"agent": route, "attempt": count})
                    logger.warning(msg)

            span["attributes"]["iterations"] = state.iteration

        self._validate(state)
        return state

    @staticmethod
    def _validate(state: ResearchState) -> None:
        """Guardrail: ensure the run produced a usable answer or a clear fallback."""

        if not state.final_answer:
            fallback = (
                "Unable to produce a complete answer within the configured guardrails. "
                f"Partial progress: sources={len(state.sources)}, "
                f"research={'yes' if state.research_notes else 'no'}, "
                f"analysis={'yes' if state.analysis_notes else 'no'}."
            )
            state.final_answer = fallback
            state.add_trace_event("validation_fallback", {"reason": "missing_final_answer"})
            if not state.errors:
                raise AgentExecutionError("workflow finished without a final answer")
