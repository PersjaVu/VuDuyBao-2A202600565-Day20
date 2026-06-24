"""Benchmark single-agent vs multi-agent runs.

Measures latency, estimated cost, token usage, citation coverage, a heuristic quality
score, and error rate -- enough to compare approaches with numbers rather than vibes.
"""

from __future__ import annotations

from collections.abc import Callable
from time import perf_counter

from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState

Runner = Callable[[str], ResearchState]


def score_quality(state: ResearchState) -> float:
    """Heuristic 0-10 quality score from observable signals.

    Combines: produced an answer, answer substance, citation coverage, presence of a
    sources section, and absence of errors. Intended as an automatic proxy; peer review
    still provides the human 0-10 score in the rubric.
    """

    score = 0.0
    answer = state.final_answer or ""
    if answer:
        score += 2.0
    if len(answer) >= 300:
        score += 2.0
    score += 3.0 * state.citation_coverage
    if "Sources" in answer:
        score += 1.0
    if state.analysis_notes:
        score += 1.0
    if not state.errors:
        score += 1.0
    return round(min(10.0, score), 2)


def run_benchmark(
    run_name: str, query: str, runner: Runner
) -> tuple[ResearchState, BenchmarkMetrics]:
    """Run ``runner`` on ``query`` and return the state plus computed metrics."""

    started = perf_counter()
    try:
        state = runner(query)
    except Exception as exc:  # capture failures as an error-rate=1 result
        latency = perf_counter() - started
        metrics = BenchmarkMetrics(
            run_name=run_name,
            latency_seconds=latency,
            error_rate=1.0,
            quality_score=0.0,
            notes=f"run raised: {exc}",
        )
        return ResearchState.model_construct(), metrics
    latency = perf_counter() - started

    error_rate = 1.0 if state.errors else 0.0
    notes = "; ".join(state.errors[:2]) if state.errors else "ok"
    metrics = BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=latency,
        estimated_cost_usd=state.estimated_cost_usd,
        quality_score=score_quality(state),
        citation_coverage=state.citation_coverage,
        error_rate=error_rate,
        input_tokens=state.total_input_tokens,
        output_tokens=state.total_output_tokens,
        notes=notes,
    )
    return state, metrics
