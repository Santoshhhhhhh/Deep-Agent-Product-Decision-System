"""LLM-as-judge autorater.

Grades each recommended product for relevance to the query: a product is
*relevant* iff it satisfies all hard constraints AND is a reasonable fit
for the stated preferences. The metric reported is top-5 relevance:
(# relevant items in top-5) / 5, averaged over queries.

In ``anthropic`` mode this is a real LLM judge; in ``mock`` mode a
deterministic rubric judge applies the same criteria, keeping the harness
reproducible and free to run.
"""
from __future__ import annotations

import json

from pydantic import BaseModel, Field

from deep_agent.catalog import Catalog
from deep_agent.llm import LLMClient
from deep_agent.state import Plan, Product

JUDGE_SYSTEM = """ROLE: judge
You are an impartial evaluator of product recommendations.
For each recommended product, decide if it is RELEVANT to the query:
- It must satisfy every hard constraint (use the provided attributes).
- It should plausibly serve the stated preferences.
Score each item in [0,1]; relevant = score >= 0.6. Judge strictly."""


class ItemScore(BaseModel):
    product_id: str
    relevant: bool
    score: float = Field(..., ge=0.0, le=1.0)


class JudgeReport(BaseModel):
    item_scores: list[ItemScore]


def judge_top_k(
    llm: LLMClient,
    plan: Plan,
    products: list[Product],
    k: int = 5,
) -> float:
    """Return top-k relevance in [0,1] for a recommendation list."""
    items = products[:k]
    if not items:
        return 0.0
    ctx = json.dumps({
        "query": plan.query,
        "hard_constraints": [c.model_dump() for c in plan.hard_constraints],
        "preferences": [p.model_dump() for p in plan.preferences],
        "recommendations": [
            {"product_id": p.id, "name": p.name, "price": p.price,
             "rating": p.rating, "attributes": p.attributes,
             "description": p.description}
            for p in items
        ],
    })
    report = llm.structured_call(
        system=JUDGE_SYSTEM,
        user=f"Judge these recommendations.\n<context>{ctx}</context>",
        response_model=JudgeReport,
    )
    by_id = {s.product_id: s for s in report.item_scores}
    relevant = sum(1 for p in items if by_id.get(p.id) and by_id[p.id].relevant)
    return relevant / k
