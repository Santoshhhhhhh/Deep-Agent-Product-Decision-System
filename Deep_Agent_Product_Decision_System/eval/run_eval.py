"""Run the evaluation: Deep-Agent system vs constraint-filter baseline.

Both systems answer the same queries; the LLM-as-judge autorater grades
top-5 relevance for each. Reports mean relevance, win/tie/loss counts,
and constraint-violation rates.

Usage:
  python eval/run_eval.py --n 200          # quick offline run (mock LLM)
  python eval/run_eval.py --n 1200         # full eval
  DEEP_AGENT_LLM=anthropic python eval/run_eval.py --n 100   # real LLM
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from autorater import judge_top_k                     # noqa: E402
from baseline import baseline_top_k                   # noqa: E402
from generate_queries import make_queries             # noqa: E402

from deep_agent.catalog import Catalog                # noqa: E402
from deep_agent.graph import recommend                # noqa: E402
from deep_agent.llm import LLMClient                  # noqa: E402
from deep_agent.state import Plan                     # noqa: E402
from deep_agent.vfs import VirtualFileSystem          # noqa: E402


def violation_rate(plan: Plan, products, k: int = 5) -> float:
    items = products[:k]
    if not items:
        return 1.0
    bad = sum(1 for p in items
              if not all(c.check(p) for c in plan.hard_constraints))
    return bad / len(items)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200, help="number of eval queries")
    ap.add_argument("--llm", choices=["mock", "anthropic"], default=None)
    ap.add_argument("--out", default=str(Path(__file__).parent / "results.json"))
    args = ap.parse_args()

    llm = LLMClient(mode=args.llm) if args.llm else LLMClient()
    judge = llm  # same backend for the judge; swap for a stronger model if desired
    catalog = Catalog()
    queries = make_queries(args.n)

    agent_scores, base_scores = [], []
    agent_viol, base_viol = [], []
    wins = ties = losses = 0
    t0 = time.time()

    for i, q in enumerate(queries, 1):
        vfs = VirtualFileSystem()
        answer, vfs = recommend(q, llm=llm, catalog=catalog, vfs=vfs)
        plan = vfs.read_model("/plan/plan.json", Plan)

        agent_products = [catalog.get(r.product_id)
                          for r in answer.recommendations]
        agent_products = [p for p in agent_products if p]
        base_products = baseline_top_k(plan, catalog)

        a = judge_top_k(judge, plan, agent_products)
        b = judge_top_k(judge, plan, base_products)
        agent_scores.append(a)
        base_scores.append(b)
        agent_viol.append(violation_rate(plan, agent_products))
        base_viol.append(violation_rate(plan, base_products))
        wins += a > b
        ties += a == b
        losses += a < b

        if i % 50 == 0 or i == len(queries):
            print(f"[{i}/{len(queries)}] agent={statistics.mean(agent_scores):.3f} "
                  f"baseline={statistics.mean(base_scores):.3f} "
                  f"({time.time() - t0:.0f}s)")

    results = {
        "n_queries": len(queries),
        "llm_mode": llm.mode,
        "top5_relevance": {
            "agent_system": round(statistics.mean(agent_scores), 4),
            "baseline": round(statistics.mean(base_scores), 4),
        },
        "constraint_violation_rate_top5": {
            "agent_system": round(statistics.mean(agent_viol), 4),
            "baseline": round(statistics.mean(base_viol), 4),
        },
        "head_to_head": {"wins": wins, "ties": ties, "losses": losses},
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\n" + json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
