"""Agent unit tests (offline, deterministic).

The starter shipped these agents as ``TODO`` stubs that raised ``StudentTodoError``; now
that they are implemented, the tests assert real behavior instead.
"""

from multi_agent_research_lab.agents import (
    AnalystAgent,
    CriticAgent,
    ResearcherAgent,
    SupervisorAgent,
    WriterAgent,
)
from multi_agent_research_lab.agents.supervisor import (
    ROUTE_ANALYST,
    ROUTE_CRITIC,
    ROUTE_DONE,
    ROUTE_RESEARCHER,
    ROUTE_WRITER,
)
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState


def _state() -> ResearchState:
    return ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))


def test_supervisor_routes_through_pipeline() -> None:
    sup = SupervisorAgent(max_iterations=6)
    state = _state()

    assert sup.decide(state) == ROUTE_RESEARCHER
    state.research_notes = "notes"
    assert sup.decide(state) == ROUTE_ANALYST
    state.analysis_notes = "analysis"
    assert sup.decide(state) == ROUTE_WRITER
    state.final_answer = "answer"
    assert sup.decide(state) == ROUTE_CRITIC
    state.critic_review = "review"
    assert sup.decide(state) == ROUTE_DONE


def test_supervisor_enforces_max_iterations() -> None:
    sup = SupervisorAgent(max_iterations=2)
    state = _state()
    state.iteration = 2
    assert sup.decide(state) == ROUTE_DONE


def test_supervisor_falls_back_when_agent_keeps_failing() -> None:
    sup = SupervisorAgent(max_iterations=6)
    state = _state()
    state.failures["researcher"] = 2  # exceeded retry budget
    # With researcher unavailable and no research notes, it should not loop on researcher.
    assert sup.decide(state) != ROUTE_RESEARCHER


def test_worker_agents_pipeline_with_injected_llm(fake_llm, fake_search) -> None:
    state = _state()
    ResearcherAgent(search=fake_search, llm=fake_llm).run(state)
    assert state.sources
    assert state.research_notes

    AnalystAgent(llm=fake_llm).run(state)
    assert state.analysis_notes
    assert state.total_claims == len(state.sources)

    WriterAgent(llm=fake_llm).run(state)
    assert state.final_answer
    assert "Sources" in state.final_answer
    assert state.cited_claims == state.total_claims

    CriticAgent(llm=fake_llm).run(state)
    assert state.critic_review
    assert state.citation_coverage == 1.0
