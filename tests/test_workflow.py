"""End-to-end workflow and baseline tests (injected fake LLM, no network)."""

from multi_agent_research_lab.baseline import run_single_agent
from multi_agent_research_lab.core.config import Settings
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow


def _settings() -> Settings:
    return Settings(
        OPENROUTER_API_KEY=None, OPENAI_API_KEY=None, MAX_ITERATIONS=6, TIMEOUT_SECONDS=30
    )


def test_multi_agent_workflow_completes(fake_llm, fake_search) -> None:
    wf = MultiAgentWorkflow(settings=_settings(), llm=fake_llm, search=fake_search)
    state = ResearchState(request=ResearchQuery(query="Research multi-agent guardrails"))
    result = wf.run(state)

    assert result.is_complete
    assert result.final_answer
    assert result.route_history[0] == "researcher"
    assert "critic" in result.route_history
    assert result.estimated_cost_usd >= 0.0
    assert not result.errors


def test_workflow_without_critic_skips_critic(fake_llm, fake_search) -> None:
    wf = MultiAgentWorkflow(
        settings=_settings(), llm=fake_llm, search=fake_search, enable_critic=False
    )
    state = ResearchState(request=ResearchQuery(query="Compare agent architectures"))
    result = wf.run(state)
    assert "critic" not in result.route_history
    assert result.critic_review is None


def test_workflow_build_returns_object(fake_llm, fake_search) -> None:
    wf = MultiAgentWorkflow(settings=_settings(), llm=fake_llm, search=fake_search)
    built = wf.build()
    assert built is not None


def test_single_agent_baseline(fake_llm, fake_search) -> None:
    state = ResearchState(request=ResearchQuery(query="Summarize LLM agent guardrails"))
    result = run_single_agent(state, settings=_settings(), llm=fake_llm, search=fake_search)
    assert result.final_answer
    assert result.sources
