# Design

## Problem

Build a research assistant that takes a long natural-language query (e.g. *"Research GraphRAG
state-of-the-art and write a 500-word summary"*), gathers sources, analyzes them, and produces
a clear, **cited** answer for a stated audience — with a trace and benchmark, not just a pretty
demo output.

## Why multi-agent?

A single agent can answer short questions well, but for decompose-able research tasks it tends
to blur three different skills into one pass: finding evidence, reasoning over it, and writing.
That makes failures hard to localize and claims hard to verify. Splitting the work lets each
agent own one responsibility, keeps the shared state auditable, and lets us add a verification
(critic) pass. The cost is ~5× more tokens and higher latency (see
[benchmark report](../reports/benchmark_report.md)), so single-agent stays the right default
for short, single-shot questions.

## Agent roles

| Agent | Responsibility | Input | Output | Failure mode |
|---|---|---|---|---|
| Supervisor | Decide next route / stop; enforce caps | shared state | `next_agent` route | bad routing → loop (capped by `max_iterations`) |
| Researcher | Search, filter, dedupe sources; write notes | query, `max_sources` | `sources`, `research_notes` | empty/irrelevant results → fallback corpus, failure counted |
| Analyst | Extract claims, compare views, flag weak evidence | `research_notes`, `sources` | `analysis_notes`, `total_claims` | over-claims beyond sources → flagged as weak evidence |
| Writer | Synthesize cited final answer | research + analysis | `final_answer`, `cited_claims` | uncited claims → low citation coverage |
| Critic (bonus) | Verify citations / hallucination markers | `final_answer` | `critic_review` verdict | misses a subtle hallucination → still raises coverage signal |

## Shared state

`ResearchState` ([state.py](../src/multi_agent_research_lab/core/state.py)) fields and why:

- `request` — the immutable query + audience + `max_sources`.
- `iteration`, `route_history`, `next_agent`, `failures` — the **control plane**: enables the
  supervisor to route, the workflow to enforce caps, and humans to read the path taken.
- `sources`, `research_notes`, `analysis_notes`, `final_answer`, `critic_review` — the **work
  products**, one field per handoff so no context is lost between agents.
- `total_claims`, `cited_claims`, `total_input/output_tokens`, `estimated_cost_usd` — the
  **evaluation signals** the benchmark consumes (citation coverage, cost).
- `agent_results`, `trace`, `errors` — **observability**: who produced what, ordered span
  events, and recorded failures.

## Routing policy

```text
        ┌─────────────┐
        │ Supervisor  │◀──────────────┐ (after every worker)
        └──────┬──────┘               │
   no notes?   │ has notes, no analysis? has answer, no review?
   ▼           ▼                       ▼
Researcher → Analyst → Writer → Critic ──▶ done
```

Decision order (see `SupervisorAgent.decide`): missing research → researcher; research but no
analysis → analyst; no answer → writer; answer but no review (and critic enabled) → critic;
otherwise → done. `iteration >= max_iterations` or an agent past its failure budget short-circuits
to `done` / routes around the failing agent.

## Guardrails

- **Max iterations:** `Settings.max_iterations` (default 6), enforced in `SupervisorAgent`.
- **Timeout:** `Settings.timeout_seconds` (default 60), wall-clock deadline in `MultiAgentWorkflow.run`.
- **Retry:** `tenacity` retry in `LLMClient._complete_live`; per-agent retry counting in the workflow.
- **Fallback:** offline LLM/search clients; supervisor routes around agents that exceed the
  failure budget; `_validate` synthesizes a fallback answer if none was produced.
- **Validation:** Pydantic schemas on all I/O; `_validate` post-condition guarantees a final answer.

## Benchmark plan

| Query | Metrics | Expected outcome |
|---|---|---|
| Research GraphRAG state-of-the-art | latency, cost, coverage, quality | multi ≥ single on quality, ~5× cost |
| Compare single vs multi for support | same | multi better structured, higher token use |
| Summarize LLM production guardrails | same | both high coverage; multi adds critic verdict |

Run via `python -m multi_agent_research_lab.cli benchmark`; results in
[reports/benchmark_report.md](../reports/benchmark_report.md).
