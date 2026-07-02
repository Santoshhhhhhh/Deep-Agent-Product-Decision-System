"""Constraint-filter baseline.

The comparison system from the resume bullet: parse the same hard
constraints (reusing the planner so both systems see identical
constraints), filter the catalog, and sort by rating. No preference-aware
ranking, no fusion, no critic loop.
"""
from __future__ import annotations

from deep_agent.catalog import Catalog
from deep_agent.state import Plan, Product


def baseline_top_k(plan: Plan, catalog: Catalog, k: int = 5) -> list[Product]:
    pool = catalog.candidates(category=plan.category)
    passing = [p for p in pool
               if all(c.check(p) for c in plan.hard_constraints)]
    passing.sort(key=lambda p: (p.rating, p.review_count), reverse=True)
    return passing[:k]
