"""Specialist ranking agents (run in parallel via LangGraph Send).

Each specialist receives its SubTask plus the candidate pool, scores every
candidate on its own dimension (budget fit / feature match / review
quality), and writes a Pydantic-validated Ranking to the VFS at
``/rankings/<agent>.json``.
"""
from __future__ import annotations

import json

from ..catalog import Catalog
from ..llm import LLMClient
from ..state import Plan, Ranking, SubTask
from ..vfs import VirtualFileSystem

SYSTEM = """ROLE: specialist
You are the '{agent}' specialist of a product-recommendation system.
Score every candidate product between 0 and 1 on YOUR dimension only:
- budget: price-to-value fit under the budget constraint (0 if over budget)
- feature: how well attributes/description match the stated preferences
- review: review quality (rating) and credibility (review volume)
Any product violating one of YOUR assigned hard constraints scores 0.
Provide a one-line rationale per product. Score ALL products given."""


def run_specialist(
    llm: LLMClient,
    vfs: VirtualFileSystem,
    catalog: Catalog,
    plan_path: str,
    subtask: SubTask,
) -> str:
    plan = vfs.read_model(plan_path, Plan)
    candidates = catalog.candidates(category=plan.category)
    ctx = json.dumps({
        "agent": subtask.agent,
        "instruction": subtask.instruction,
        "constraints": [c.model_dump() for c in subtask.constraints],
        "preferences": [p.model_dump() for p in subtask.preferences],
        "products": [p.model_dump() for p in candidates],
    })
    ranking = llm.structured_call(
        system=SYSTEM.format(agent=subtask.agent),
        user=f"Rank these candidates.\n<context>{ctx}</context>",
        response_model=Ranking,
    )
    # Guardrail: specialists may only score products that actually exist.
    valid_ids = {p.id for p in candidates}
    ranking.candidates = [c for c in ranking.candidates if c.product_id in valid_ids]
    return vfs.write_model(f"/rankings/{subtask.agent}.json", ranking)
