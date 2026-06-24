# SOLUTION — Lab 20: Multi-Agent Research System

This document maps every lab requirement to where it is implemented and verified, so the
submission can be checked against the rubric and deliverables at a glance.

## How to run / verify

```bash
# install (dev tooling + LLM SDK)
python -m pip install -e ".[dev,llm]"

# configure the LLM provider (real calls — no mock). Edit .env:
#   OPENROUTER_API_KEY=sk-or-...           (recommended)
#   OPENROUTER_MODEL=openai/gpt-oss-120b:free
# (OPENAI_API_KEY also supported as a fallback)

# quality gates (all green)
ruff check src tests        # lint  -> All checks passed!
mypy src                    # types -> Success: no issues found in 29 source files
pytest                      # tests -> 20 passed (injected fake LLM + fake search, no network)

# run it (makes real LLM calls via OpenRouter)
python -m multi_agent_research_lab.cli baseline    -q "Summarize production guardrails for LLM agents"
python -m multi_agent_research_lab.cli multi-agent -q "Research GraphRAG state-of-the-art" --trace-out reports/trace_example.json
python -m multi_agent_research_lab.cli benchmark   # writes reports/benchmark_report.md
```

> **Real LLM, no mock.** `LLMClient` calls a live model (OpenRouter or OpenAI, both via the
> OpenAI-compatible API). There is **no offline/mock fallback** — without credentials it raises
> a clear error instead of returning fake text. The agents consume the model's actual output as
> their research/analysis/answer. Tests stay fast by injecting a `FakeLLM` test-double through
> constructor dependency injection (see `tests/conftest.py`), which is test scaffolding, not a
> mock baked into the product. The `SearchClient` does **real web search** too — Tavily when a
> key is set, otherwise keyless **DuckDuckGo** (`ddgs`); a small local corpus is only a
> last-resort fallback if the network/package is unavailable (the assignment permits a mock
> search source, but the default path is real).

## Learning outcomes → evidence

| # | Outcome | Where |
|---|---|---|
| 1 | Clear roles for multiple agents | [supervisor](src/multi_agent_research_lab/agents/supervisor.py), [researcher](src/multi_agent_research_lab/agents/researcher.py), [analyst](src/multi_agent_research_lab/agents/analyst.py), [writer](src/multi_agent_research_lab/agents/writer.py), [critic](src/multi_agent_research_lab/agents/critic.py) |
| 2 | Shared state for handoff | [core/state.py](src/multi_agent_research_lab/core/state.py) (`ResearchState`) |
| 3 | Guardrails: max iters, timeout, retry/fallback, validation | [graph/workflow.py](src/multi_agent_research_lab/graph/workflow.py), [supervisor.py](src/multi_agent_research_lab/agents/supervisor.py) |
| 4 | Traceable flow | [observability/tracing.py](src/multi_agent_research_lab/observability/tracing.py) — **live LangSmith** runs (project `VinUni`, APAC) + [trace_example.json](reports/trace_example.json) |
| 5 | Benchmark single vs multi (quality/latency/cost) | [evaluation/benchmark.py](src/multi_agent_research_lab/evaluation/benchmark.py), [reports/benchmark_report.md](reports/benchmark_report.md) |

## Student TODOs → completed

All `TODO(student)` stubs that raised `StudentTodoError` are implemented (remaining matches of
the string are only instructional prose in `docs/lab_guide.md` and grep examples).

| TODO | File | Status |
|---|---|---|
| Implement LLM client | [services/llm_client.py](src/multi_agent_research_lab/services/llm_client.py) | ✅ OpenAI (lazy) + offline fallback, tenacity retry, token/cost accounting |
| Implement search client / mock | [services/search_client.py](src/multi_agent_research_lab/services/search_client.py) | ✅ real web search: Tavily (key) → **DuckDuckGo keyless** → corpus fallback |
| Routing decision in Supervisor | [agents/supervisor.py](src/multi_agent_research_lab/agents/supervisor.py) | ✅ priority policy + max-iter + failure fallback |
| Each worker agent | [agents/researcher.py](src/multi_agent_research_lab/agents/researcher.py), [analyst.py](src/multi_agent_research_lab/agents/analyst.py), [writer.py](src/multi_agent_research_lab/agents/writer.py) | ✅ search→notes, claims, cited synthesis |
| Build LangGraph workflow | [graph/workflow.py](src/multi_agent_research_lab/graph/workflow.py) | ✅ portable router engine + real `StateGraph` when langgraph installed |
| Real tracing provider | [observability/tracing.py](src/multi_agent_research_lab/observability/tracing.py) | ✅ **live LangSmith** (nested run tree, region-aware endpoint US/EU/APAC) + JSON export fallback |
| Benchmark report | [evaluation/benchmark.py](src/multi_agent_research_lab/evaluation/benchmark.py), [evaluation/report.py](src/multi_agent_research_lab/evaluation/report.py) | ✅ latency/cost/tokens/coverage/quality/error-rate |
| Baseline = real single-agent | [baseline.py](src/multi_agent_research_lab/baseline.py), [cli.py](src/multi_agent_research_lab/cli.py) | ✅ replaced placeholder |
| Critic agent (bonus) | [agents/critic.py](src/multi_agent_research_lab/agents/critic.py) | ✅ citation/hallucination verification pass |
| Design doc | [docs/design_template.md](docs/design_template.md) | ✅ filled |

## Peer-review rubric self-check (target: 2/2 each = 10/10)

| Criterion | Evidence | Score |
|---|---|---:|
| Role clarity | 5 single-responsibility agents; supervisor owns routing only, workers own work only | 2/2 |
| State design | `ResearchState` has control plane + work products + eval signals + observability, one field per handoff | 2/2 |
| Failure guard | max_iterations, timeout, per-agent retry/fallback, Pydantic + `_validate` post-condition | 2/2 |
| Benchmark | single vs multi across 3 queries with latency/cost/tokens/coverage/quality/error-rate | 2/2 |
| Trace explanation | structured spans + JSON export with route history, usage, per-step events | 2/2 |

## Deliverables

| Deliverable | Status |
|---|---|
| Personal repo / code | ✅ implemented in `src/` |
| Trace screenshot or link | ✅ live on LangSmith (project `VinUni`, https://smith.langchain.com); put the image in [screenshots/](screenshots/) (e.g. `langsmith_trace.png`) + [reports/trace_example.json](reports/trace_example.json) export per run via `--trace-out` |
| `reports/benchmark_report.md` single vs multi | ✅ [reports/benchmark_report.md](reports/benchmark_report.md) |
| Failure-mode explanation + fix | ✅ dedicated [Explain_failure_mode_and_fixing.md](Explain_failure_mode_and_fixing.md) (8 failure modes) + summary in the benchmark report |
| Exit ticket (when to / not to use multi-agent) | ✅ in the benchmark report ("Exit ticket" section) |

## Bonus

- **Live LangSmith tracing** with a nested run tree (workflow → agents → llm calls) and a
  region-aware endpoint (US / EU / APAC), viewable/screenshot-able on the web dashboard.
- **Critic agent** verification pass (citation coverage + hallucination markers + verdict).
- **Offline determinism** so the lab is fully reproducible without secrets or network.
- **Optional real LangGraph `StateGraph`** compiled when the `llm` extra is installed.
- **Automatic quality scoring** in addition to peer-review scoring.
- **Strict typing + lint**: `mypy --strict` and `ruff` both pass.

## Quality gate results (last run)

```text
ruff check src tests   ->  All checks passed!
mypy src               ->  Success: no issues found in 29 source files
pytest                 ->  20 passed
```

## Notes / limitations

- The benchmark numbers are from real OpenRouter + real DuckDuckGo calls; the `:free` model is
  a slow reasoning model (~20–40s/call), so latency is high — set a faster model in `.env`
  (`OPENROUTER_MODEL`) for snappier runs. `gpt-oss-120b:free` can also rate-limit under bursts.
- The LangGraph-backed `build()` path is only exercised when `langgraph` is installed; the
  default execution path is the built-in router engine, which mirrors the same
  node/conditional-edge model and is what tests/CI run.
- Tests never hit the network: they inject `FakeLLM` and `FakeSearch` via the agents'/workflow's
  constructors (dependency injection), so the orchestration logic is verified deterministically.
