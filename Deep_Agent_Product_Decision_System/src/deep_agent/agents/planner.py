"""Planner agent.

Decomposes the user query into (a) hard, machine-verifiable constraints,
(b) soft preferences, and (c) one sub-task per specialist agent. The
resulting Plan is Pydantic-validated and persisted to the VFS at
``/plan/plan.json``; downstream nodes receive only the path.
"""
from __future__ import annotations

import json

from ..llm import LLMClient
from ..state import Plan
from ..vfs import VirtualFileSystem

SYSTEM = """ROLE: planner
You are the planning agent of a product-recommendation system.
Decompose the user's shopping query into:
1. hard_constraints — objective pass/fail conditions on product fields
   (price, rating, ram_gb, etc.) using ops lte/gte/eq/contains.
2. preferences — soft desires with weights (battery life, portability...).
3. subtasks — exactly one per specialist: budget, feature, review.
Only put a constraint in hard_constraints if it is explicitly stated."""


def run_planner(llm: LLMClient, vfs: VirtualFileSystem, query: str) -> str:
    ctx = json.dumps({"query": query})
    plan = llm.structured_call(
        system=SYSTEM,
        user=f"Decompose this query.\n<context>{ctx}</context>",
        response_model=Plan,
    )
    return vfs.write_model("/plan/plan.json", plan)
