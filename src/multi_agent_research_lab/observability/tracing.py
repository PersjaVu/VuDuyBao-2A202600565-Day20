"""Tracing hooks.

Two layers:

* **LangSmith** (web dashboard) -- when ``LANGSMITH_API_KEY`` is set and the ``langsmith``
  package is installed, every span is sent to LangSmith as a nested run, so the whole
  Supervisor -> Researcher -> Analyst -> Writer -> Critic flow is viewable/screenshot-able at
  https://smith.langchain.com.
* **Local JSON** -- always available as a fallback artifact via :func:`export_trace`.

Both are provider-agnostic from the agents' point of view: agents just call
:func:`trace_span`; this module decides where the span goes.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from multi_agent_research_lab.core.config import Settings
    from multi_agent_research_lab.core.state import ResearchState

logger = logging.getLogger(__name__)

try:
    from langsmith import trace as _ls_trace

    _HAS_LANGSMITH = True
except ImportError:  # pragma: no cover - optional extra
    _HAS_LANGSMITH = False

# Toggled on by configure_tracing() once credentials are confirmed.
_LANGSMITH_ENABLED = False
# A reference to the active LangSmith client, captured from the first run, used to flush.
_LANGSMITH_CLIENT: Any = None


def configure_tracing(settings: Settings) -> str:
    """Detect/enable a tracing provider and return its name.

    Returns "langsmith" when runs will be sent to the web dashboard, else "local-json".
    """

    global _LANGSMITH_ENABLED

    if settings.langsmith_api_key and _HAS_LANGSMITH:
        os.environ.setdefault("LANGSMITH_API_KEY", settings.langsmith_api_key)
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ.setdefault("LANGSMITH_PROJECT", settings.langsmith_project)
        # Region-aware endpoint (US default; set to the EU/APAC host for those regions).
        os.environ.setdefault("LANGSMITH_ENDPOINT", settings.langsmith_endpoint)
        _LANGSMITH_ENABLED = True
        logger.info(
            "tracing -> LangSmith project=%s endpoint=%s",
            settings.langsmith_project,
            settings.langsmith_endpoint,
        )
        return "langsmith"

    _LANGSMITH_ENABLED = False
    if settings.langsmith_api_key and not _HAS_LANGSMITH:
        logger.warning("LANGSMITH_API_KEY set but 'langsmith' not installed; using local JSON")
    else:
        logger.info("tracing -> local JSON (set LANGSMITH_API_KEY to use the web dashboard)")
    return "local-json"


@contextmanager
def trace_span(
    name: str,
    attributes: dict[str, Any] | None = None,
    *,
    run_type: str = "chain",
) -> Iterator[dict[str, Any]]:
    """Span context manager. Sends to LangSmith when enabled; always times the block.

    Nested ``trace_span`` calls auto-parent via LangSmith's context vars, producing the run
    tree shown on the web dashboard.
    """

    started = perf_counter()
    span: dict[str, Any] = {"name": name, "attributes": attributes or {}, "duration_seconds": None}

    if _LANGSMITH_ENABLED and _HAS_LANGSMITH:
        global _LANGSMITH_CLIENT
        with _ls_trace(
            name=name, run_type=cast(Any, run_type), inputs=dict(attributes or {})
        ) as run:
            _LANGSMITH_CLIENT = getattr(run, "client", None) or _LANGSMITH_CLIENT
            try:
                yield span
            finally:
                span["duration_seconds"] = perf_counter() - started
                try:
                    run.end(outputs={"attributes": span["attributes"]})
                except Exception:  # pragma: no cover - never let tracing break a run
                    logger.debug("failed to close langsmith run for %s", name)
    else:
        try:
            yield span
        finally:
            span["duration_seconds"] = perf_counter() - started
            logger.debug("span %s took %.4fs", name, span["duration_seconds"])


def flush_tracing() -> None:
    """Block until queued LangSmith runs are uploaded (call before process exit)."""

    if _LANGSMITH_ENABLED and _LANGSMITH_CLIENT is not None:
        try:
            _LANGSMITH_CLIENT.flush()
        except Exception:  # pragma: no cover - best effort
            logger.debug("langsmith flush failed")


def export_trace(state: ResearchState, path: str | Path) -> Path:
    """Write the accumulated trace events to a JSON file and return the path."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "query": state.request.query,
        "iterations": state.iteration,
        "route_history": state.route_history,
        "errors": state.errors,
        "usage": {
            "input_tokens": state.total_input_tokens,
            "output_tokens": state.total_output_tokens,
            "estimated_cost_usd": state.estimated_cost_usd,
        },
        "events": state.trace,
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out
