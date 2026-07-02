"""Aggregator agent.

Merges the specialists' rankings with weighted Reciprocal Rank Fusion
(RRF) blended with raw scores. On revision rounds it applies the critic's
feedback: excluded product IDs are dropped and per-agent weights are
adjusted. Output: ``/fused/round_<n>.json``.
"""
from __future__ import annotations

from collections import defaultdict

from ..state import Critique, Ranking, ScoredCandidate
from ..vfs import VirtualFileSystem

RRF_K = 20
DEFAULT_WEIGHTS = {"budget": 1.0, "feature": 1.2, "review": 0.8}


def run_aggregator(
    vfs: VirtualFileSystem,
    ranking_paths: list[str],
    revision: int,
    critique_path: str | None = None,
) -> str:
    # Accumulate feedback from ALL critic rounds so far, so exclusions are
    # monotonic and the bounded loop converges instead of oscillating.
    weights = dict(DEFAULT_WEIGHTS)
    excluded: set[str] = set()
    for report_path in vfs.ls("/critic"):
        critique = vfs.read_model(report_path, Critique)
        excluded.update(critique.excluded_product_ids)
        for agent, w in critique.weight_adjustments.items():
            weights[agent] = max(weights.get(agent, 1.0), w)

    fused: dict[str, float] = defaultdict(float)
    rationales: dict[str, list[str]] = defaultdict(list)

    for path in sorted(set(ranking_paths)):
        ranking = vfs.read_model(path, Ranking)
        w = weights.get(ranking.agent, 1.0)
        for rank, cand in enumerate(ranking.candidates, start=1):
            if cand.product_id in excluded:
                continue
            # Hybrid: RRF for rank stability + raw score for magnitude.
            fused[cand.product_id] += w * (1.0 / (RRF_K + rank) + 0.02 * cand.score)
            if cand.rationale:
                rationales[cand.product_id].append(f"{ranking.agent}: {cand.rationale}")

    if fused:
        top = max(fused.values())
        merged = Ranking(
            agent="aggregator",
            candidates=[
                ScoredCandidate(
                    product_id=pid,
                    score=round(score / top, 4),
                    rationale=" | ".join(rationales[pid][:3]),
                )
                for pid, score in fused.items()
            ],
        )
    else:
        merged = Ranking(agent="aggregator", candidates=[])
    return vfs.write_model(f"/fused/round_{revision}.json", merged)
