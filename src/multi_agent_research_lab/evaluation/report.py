"""Benchmark report rendering."""

from __future__ import annotations

from datetime import UTC, datetime

from multi_agent_research_lab.core.schemas import BenchmarkMetrics


def _fmt(value: float | int | None, spec: str = "") -> str:
    if value is None:
        return ""
    return format(value, spec) if spec else str(value)


def render_markdown_report(
    metrics: list[BenchmarkMetrics], *, title: str = "Benchmark Report"
) -> str:
    """Render benchmark metrics to a rich markdown report."""

    generated = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    header = (
        "| Run | Latency (s) | Cost (USD) | In tok | Out tok | "
        "Citation cov. | Quality (0-10) | Error rate | Notes |"
    )
    lines = [
        f"# {title}",
        "",
        f"_Generated: {generated}_",
        "",
        header,
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for m in metrics:
        cov = _fmt(m.citation_coverage, ".0%") if m.citation_coverage is not None else ""
        err = _fmt(m.error_rate, ".0%") if m.error_rate is not None else ""
        row = (
            f"| {m.run_name} | {_fmt(m.latency_seconds, '.3f')} | "
            f"{_fmt(m.estimated_cost_usd, '.6f')} | {_fmt(m.input_tokens)} | "
            f"{_fmt(m.output_tokens)} | {cov} | {_fmt(m.quality_score, '.1f')} | "
            f"{err} | {m.notes} |"
        )
        lines.append(row)

    lines += ["", "## How to read this", ""]
    lines += [
        "- **Latency** is wall-clock time per run.",
        "- **Cost** and **tokens** are estimated from the LLM client's usage accounting.",
        "- **Citation coverage** = cited claims / total claims.",
        "- **Quality** is the automatic heuristic score; pair it with peer-review scores.",
        "- **Error rate** is 1.0 if the run recorded any error, else 0.0.",
    ]
    return "\n".join(lines) + "\n"
