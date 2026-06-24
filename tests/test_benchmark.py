"""Benchmark + report tests (injected fake LLM)."""

from multi_agent_research_lab.core.config import Settings
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import run_benchmark, score_quality
from multi_agent_research_lab.evaluation.report import render_markdown_report
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow


def _settings() -> Settings:
    return Settings(
        OPENROUTER_API_KEY=None, OPENAI_API_KEY=None, MAX_ITERATIONS=6, TIMEOUT_SECONDS=30
    )


def test_run_benchmark_computes_metrics(fake_llm, fake_search) -> None:
    settings = _settings()

    def runner(query: str) -> ResearchState:
        return MultiAgentWorkflow(settings=settings, llm=fake_llm, search=fake_search).run(
            ResearchState(request=ResearchQuery(query=query))
        )

    state, metrics = run_benchmark("multi", "Research agent guardrails", runner)
    assert state.final_answer
    assert metrics.latency_seconds >= 0.0
    assert metrics.citation_coverage == 1.0
    assert metrics.error_rate == 0.0
    assert metrics.quality_score and metrics.quality_score > 5.0


def test_score_quality_rewards_citations_and_penalizes_errors() -> None:
    good = ResearchState(request=ResearchQuery(query="quality check please"))
    good.final_answer = "x" * 400 + "\n### Sources\n[1]"
    good.analysis_notes = "claims"
    good.total_claims = 2
    good.cited_claims = 2
    assert score_quality(good) >= 9.0

    bad = ResearchState(request=ResearchQuery(query="quality check please"))
    bad.errors.append("boom")
    assert score_quality(bad) < 3.0


def test_report_includes_new_columns(fake_llm, fake_search) -> None:
    settings = _settings()
    _, metrics = run_benchmark(
        "multi",
        "Research agent guardrails",
        lambda q: MultiAgentWorkflow(settings=settings, llm=fake_llm, search=fake_search).run(
            ResearchState(request=ResearchQuery(query=q))
        ),
    )
    report = render_markdown_report([metrics])
    assert "Citation cov." in report
    assert "Quality" in report
