"""Critic agent.

Design principle: never ask an LLM to verify what code can verify.
The critic checks every hard constraint from the Plan against ground-truth
catalog attributes for the fused top-k. Violating products are excluded
and, when a systematic pattern appears (e.g. multiple over-budget picks),
the offending specialist's fusion weight is boosted so the next round
self-corrects. The loop is bounded by ``max_revisions`` in the graph.

Output: ``/critic/report_<n>.json`` (Pydantic ``Critique``).
"""
from __future__ import annotations

from collections import Counter

from ..catalog import Catalog
from ..state import Critique, Plan, Ranking, Violation
from ..vfs import VirtualFileSystem

TOP_K = 5


def run_critic(
    vfs: VirtualFileSystem,
    catalog: Catalog,
    plan_path: str,
    fused_path: str,
    revision: int,
) -> str:
    plan = vfs.read_model(plan_path, Plan)
    fused = vfs.read_model(fused_path, Ranking)
    top = fused.candidates[:TOP_K]
    # Approval is judged on the surfaced top-K, but violators are swept
    # from a wider window so the bounded loop converges in few rounds.
    sweep = fused.candidates[: TOP_K * 4]

    violations: list[Violation] = []
    for cand in sweep:
        product = catalog.get(cand.product_id)
        if product is None:
            violations.append(Violation(
                product_id=cand.product_id,
                constraint=plan.hard_constraints[0] if plan.hard_constraints
                else _null_constraint(),
                detail="hallucinated product id — not in catalog",
            ))
            continue
        for c in plan.hard_constraints:
            if not c.check(product):
                actual = product.attributes.get(c.field, getattr(product, c.field, None))
                violations.append(Violation(
                    product_id=cand.product_id,
                    constraint=c,
                    detail=f"{c.field}={actual} fails {c.op.value} {c.value}",
                ))

    excluded = sorted({v.product_id for v in violations})
    top_ids = {c.product_id for c in top}
    top_violations = [v for v in violations if v.product_id in top_ids]

    # Systematic-error feedback: if 2+ top-k picks violate the same field,
    # boost the specialist responsible for that dimension next round.
    weight_adjustments: dict[str, float] = {}
    field_counts = Counter(v.constraint.field for v in violations)
    for field, n in field_counts.items():
        if n >= 2:
            agent = "budget" if field == "price" else "feature"
            weight_adjustments[agent] = 2.0

    approved = len(top_violations) == 0
    notes = ("all hard constraints satisfied" if approved else
             f"{len(top_violations)} violation(s) in top-{TOP_K}; "
             f"excluding {len(excluded)} product(s) from the sweep window")

    critique = Critique(
        approved=approved,
        violations=violations,
        excluded_product_ids=excluded,
        weight_adjustments=weight_adjustments,
        notes=notes,
    )
    return vfs.write_model(f"/critic/report_{revision}.json", critique)


def _null_constraint():
    from ..state import Constraint, ConstraintOp
    return Constraint(field="id", op=ConstraintOp.EQ, value="__exists__")
