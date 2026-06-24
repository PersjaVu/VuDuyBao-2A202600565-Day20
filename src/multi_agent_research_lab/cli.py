"""Command-line entrypoint for the lab."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from multi_agent_research_lab.baseline import run_single_agent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import run_benchmark
from multi_agent_research_lab.evaluation.report import render_markdown_report
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.observability.tracing import (
    configure_tracing,
    export_trace,
    flush_tracing,
)

app = typer.Typer(help="Multi-Agent Research Lab CLI")

# Windows terminals default to cp1252 and crash on non-ASCII model output; force UTF-8.
for _stream in (sys.stdout, sys.stderr):
    reconfigure = getattr(_stream, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(encoding="utf-8", errors="replace")

console = Console()


def _init() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    tracing = configure_tracing(settings)
    if settings.openrouter_api_key:
        llm = f"openrouter:{settings.openrouter_model}"
    elif settings.openai_api_key:
        llm = f"openai:{settings.openai_model}"
    else:
        llm = "NONE (set OPENROUTER_API_KEY in .env)"
        console.print("[yellow]No LLM credentials found — live calls will fail.[/yellow]")
    console.print(f"[dim]llm={llm} · tracing={tracing}[/dim]")


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run the single-agent baseline."""

    _init()
    state = ResearchState(request=ResearchQuery(query=query))
    run_single_agent(state)
    console.print(Panel.fit(state.final_answer or "(no answer)", title="Single-Agent Baseline"))
    flush_tracing()


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
    no_critic: Annotated[
        bool, typer.Option("--no-critic", help="Disable the critic agent")
    ] = False,
    trace_out: Annotated[
        str | None, typer.Option("--trace-out", help="Write trace JSON to this path")
    ] = None,
) -> None:
    """Run the multi-agent workflow."""

    _init()
    state = ResearchState(request=ResearchQuery(query=query))
    workflow = MultiAgentWorkflow(enable_critic=not no_critic)
    result = workflow.run(state)

    console.print(Panel.fit(result.final_answer or "(no answer)", title="Multi-Agent Answer"))
    console.print(f"[dim]route: {' -> '.join(result.route_history)} -> done[/dim]")
    if result.critic_review:
        console.print(Panel.fit(result.critic_review, title="Critic Review", style="cyan"))
    if result.errors:
        console.print(Panel.fit("\n".join(result.errors), title="Errors", style="red"))
    if trace_out:
        path = export_trace(result, trace_out)
        console.print(f"[green]trace written to {path}[/green]")
    flush_tracing()


@app.command()
def benchmark(
    query: Annotated[
        list[str] | None,
        typer.Option("--query", "-q", help="Query (repeatable); omit to use defaults"),
    ] = None,
    out: Annotated[
        str, typer.Option("--out", help="Markdown report output path")
    ] = "reports/benchmark_report.md",
) -> None:
    """Benchmark single-agent vs multi-agent and write a markdown report."""

    _init()
    queries = query or [
        "Research GraphRAG state-of-the-art and write a 500-word summary",
        "Compare single-agent and multi-agent workflows for customer support",
        "Summarize production guardrails for LLM agents",
    ]

    metrics = []
    for q in queries:
        _, single = run_benchmark(
            f"single | {q[:30]}",
            q,
            lambda x: run_single_agent(ResearchState(request=ResearchQuery(query=x))),
        )
        _, multi = run_benchmark(
            f"multi | {q[:30]}",
            q,
            lambda x: MultiAgentWorkflow().run(ResearchState(request=ResearchQuery(query=x))),
        )
        metrics.extend([single, multi])

    report = render_markdown_report(metrics, title="Benchmark Report: Single vs Multi-Agent")
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")

    table = Table(title="Benchmark summary")
    for col in ("Run", "Latency (s)", "Cost (USD)", "Citation", "Quality"):
        table.add_column(col)
    for m in metrics:
        table.add_row(
            m.run_name,
            f"{m.latency_seconds:.3f}",
            f"{m.estimated_cost_usd:.6f}" if m.estimated_cost_usd is not None else "",
            f"{m.citation_coverage:.0%}" if m.citation_coverage is not None else "",
            f"{m.quality_score:.1f}" if m.quality_score is not None else "",
        )
    console.print(table)
    console.print(f"[green]report written to {path}[/green]")
    flush_tracing()


if __name__ == "__main__":
    app()
