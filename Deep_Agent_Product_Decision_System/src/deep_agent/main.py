"""CLI: python -m deep_agent.main "gaming laptop under $1500 with 16GB RAM" """
from __future__ import annotations

import argparse
import sys

from .graph import recommend
from .llm import LLMClient


def main() -> int:
    ap = argparse.ArgumentParser(description="Deep-Agent Product Decision System")
    ap.add_argument("query", nargs="+", help="natural-language product query")
    ap.add_argument("--llm", choices=["mock", "anthropic"], default=None,
                    help="LLM backend (default: $DEEP_AGENT_LLM or mock)")
    ap.add_argument("--max-revisions", type=int, default=2)
    ap.add_argument("--show-vfs", action="store_true",
                    help="print the VFS artifact tree after the run")
    args = ap.parse_args()

    query = " ".join(args.query)
    llm = LLMClient(mode=args.llm) if args.llm else LLMClient()

    answer, vfs = recommend(query, llm=llm, max_revisions=args.max_revisions)

    print(f"\nQuery: {answer.query}")
    print(f"Constraint-verified: {answer.constraint_verified} "
          f"(revisions used: {answer.revisions_used})\n")
    for i, r in enumerate(answer.recommendations, 1):
        print(f"{i}. {r.name}  —  ${r.price:.0f}, {r.rating}★  "
              f"(score {r.fused_score:.3f})")
        print(f"   {r.why}")
    if args.show_vfs:
        print("\nVFS artifacts:")
        print(vfs.tree())
    return 0


if __name__ == "__main__":
    sys.exit(main())
