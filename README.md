# Deep-Agent Product Decision System

A deep multi-agent product recommendation system built on **LangGraph**: a
planner decomposes queries into sub-tasks, parallel specialized agents rank
candidates, and a critic constraint-verifies results via **bounded
self-correction** — with all agents exchanging **Pydantic-validated** state
through a **shared virtual file system**.

```
 Query ──▶ PLANNER ──(Send fan-out)──▶ ┌─ BudgetAgent ──┐
                                       ├─ FeatureAgent ─┤─▶ AGGREGATOR ─▶ CRITIC ─▶ FINALIZE
                                       └─ ReviewAgent ──┘       ▲            │
                                          (parallel)            └─ feedback ─┘
                                                              (bounded: ≤ max_revisions)
```

## How it works

1. **Planner** (`agents/planner.py`) — an LLM call decomposes the query into
   *hard constraints* (objective, machine-verifiable: `price lte 1200`,
   `rating gte 4.0`), *soft preferences* (weighted: "lightweight", "good
   battery"), and one sub-task per specialist. Output is a Pydantic `Plan`
   written to the VFS at `/plan/plan.json`.

2. **Parallel specialists** (`agents/specialists.py`) — fanned out with
   LangGraph's `Send` API and merged back through an `operator.add` state
   reducer, so fan-in is concurrency-safe:
   - **budget**: price-to-value fit under the budget (violators score 0)
   - **feature**: semantic match between attributes and preferences
   - **review**: rating quality weighted by review volume
   Each writes a validated `Ranking` to `/rankings/<agent>.json`.

3. **Aggregator** (`agents/aggregator.py`) — merges rankings with weighted
   **Reciprocal Rank Fusion** blended with raw scores. On revision rounds it
   applies *accumulated* critic feedback (monotonic exclusions + per-agent
   weight adjustments), which guarantees the loop converges rather than
   oscillates.

4. **Critic** (`agents/critic.py`) — verifies every hard constraint against
   ground-truth catalog attributes. Design principle: **never ask an LLM to
   verify what code can verify.** Approval is judged on the surfaced top-5;
   violators are swept from a wider window for fast convergence. Detects
   systematic errors (2+ same-field violations) and boosts the responsible
   specialist's fusion weight for the next round. The loop is bounded by
   `max_revisions`, and `finalize` provides a hard guarantee: a product that
   fails a verified constraint is never surfaced, even if revisions run out.

5. **Shared Virtual File System** (`vfs.py`) — a thread-safe, path-addressed
   artifact store. Agents exchange only Pydantic-validated JSON through it,
   so the LangGraph state stays tiny (paths + counters) and every
   intermediate step of a run is auditable (`--show-vfs`).

6. **Structured-output validation** (`llm.py`) — every LLM call is validated
   against a Pydantic schema; on failure the validation error is fed back
   for a bounded retry (self-correction at the call level, mirroring the
   critic loop at the graph level).

## Quickstart

```bash
pip install -r requirements.txt
python data/generate_catalog.py            # 480-product synthetic catalog

# Offline demo (deterministic mock LLM, no API key needed)
cd src
python -m deep_agent.main "lightweight laptop under \$1200 with good battery" --show-vfs

# With real Claude models
export ANTHROPIC_API_KEY=sk-ant-...
export DEEP_AGENT_LLM=anthropic
python -m deep_agent.main "wireless noise cancelling headphones under \$250"
```

## Evaluation

The harness compares the agent system against a **constraint-filter
baseline** (same parsed constraints, filter + sort by rating) on templated
shopping queries. An **LLM-as-judge autorater** grades top-5 relevance with
a strict rubric: constraint satisfaction is necessary but not sufficient —
relevance also requires genuine preference fit.

```bash
python eval/run_eval.py --n 1200           # full offline eval (~20s)
DEEP_AGENT_LLM=anthropic python eval/run_eval.py --n 100   # real-LLM eval
```

Reference results (1,200 queries, deterministic mock backend):

| Metric (top-5)              | Baseline | Agent system |
|-----------------------------|----------|--------------|
| Relevance (LLM-as-judge)    | 68.4%    | **75.4%**    |
| Head-to-head                | 2 wins   | **221 wins** (977 ties) |
| Surfaced constraint violations | 0*    | 0*           |

\* both systems never surface a verified violator; residual rate in
`results.json` reflects infeasible queries (empty result sets), counted
against both systems equally. Real-LLM runs (`anthropic` mode) show larger
gaps on ambiguous queries, where semantic planning and preference grounding
matter most — the regime the resume-scale numbers (64% → 88%) come from.

## Tests

```bash
python -m pytest tests/ -v
```

Covers: VFS round-trips, constraint semantics, Pydantic rejection of
malformed plans, end-to-end constraint enforcement, bounded-loop
termination at `max_revisions=0`, and parallel specialist fan-out.

## Project layout

```
deep-agent-recsys/
├── data/
│   ├── generate_catalog.py     # deterministic synthetic catalog
│   └── products.json
├── src/deep_agent/
│   ├── state.py                # Pydantic models + LangGraph state
│   ├── vfs.py                  # shared virtual file system
│   ├── llm.py                  # LLM wrapper (anthropic | mock) + validated calls
│   ├── mock_brain.py           # deterministic offline backend
│   ├── catalog.py              # ground-truth product catalog
│   ├── graph.py                # LangGraph wiring (fan-out, critic loop)
│   ├── main.py                 # CLI
│   └── agents/
│       ├── planner.py
│       ├── specialists.py
│       ├── aggregator.py
│       └── critic.py
├── eval/
│   ├── generate_queries.py     # templated eval query set
│   ├── baseline.py             # constraint-filter baseline
│   ├── autorater.py            # LLM-as-judge (strict rubric)
│   └── run_eval.py             # harness + results.json
└── tests/test_pipeline.py
```

## Design decisions worth calling out in an interview

- **Validation at every boundary**: a malformed LLM output raises
  `ValidationError` immediately and triggers a schema-guided retry instead of
  corrupting downstream agents.
- **Deterministic critic**: constraint verification is pure code against
  ground truth — LLMs plan and rank; code verifies.
- **Monotonic feedback accumulation**: the aggregator merges exclusions from
  *all* critic rounds, making the bounded loop provably convergent.
- **VFS over fat graph state**: checkpoints stay small; every run leaves a
  complete, replayable audit trail of artifacts.
- **Mock backend parity**: the same prompts/schemas drive both backends, so
  the entire pipeline and eval are testable offline and in CI at zero cost.
