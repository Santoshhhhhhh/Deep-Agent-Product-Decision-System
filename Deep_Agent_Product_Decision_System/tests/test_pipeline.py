"""Tests for the Deep-Agent pipeline (run with: pytest tests/ -v).

All tests use the deterministic mock LLM, so they run offline.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
from pydantic import ValidationError

from deep_agent.catalog import Catalog
from deep_agent.graph import recommend
from deep_agent.llm import LLMClient
from deep_agent.state import Constraint, ConstraintOp, FinalAnswer, Plan, Ranking
from deep_agent.vfs import VirtualFileSystem


@pytest.fixture(scope="module")
def catalog():
    return Catalog()


@pytest.fixture()
def llm():
    return LLMClient(mode="mock")


def test_vfs_roundtrip():
    vfs = VirtualFileSystem()
    vfs.write("/a/b.json", '{"x": 1}')
    assert vfs.read("a/b.json") == '{"x": 1}'
    assert vfs.ls("/a") == ["/a/b.json"]
    with pytest.raises(FileNotFoundError):
        vfs.read("/missing.json")


def test_constraint_checks(catalog):
    p = catalog.candidates()[0]
    assert Constraint(field="price", op=ConstraintOp.LTE,
                      value=p.price + 1).check(p)
    assert not Constraint(field="price", op=ConstraintOp.LTE,
                          value=p.price - 1).check(p)


def test_plan_rejects_unknown_agent():
    with pytest.raises(ValidationError):
        Plan.model_validate({
            "query": "q",
            "subtasks": [{"agent": "hacker", "instruction": "x"}],
        })


def test_ranking_auto_sorted():
    r = Ranking(agent="feature", candidates=[
        {"product_id": "P1", "score": 0.2},
        {"product_id": "P2", "score": 0.9},
    ])
    assert r.candidates[0].product_id == "P2"


def test_end_to_end_constraint_verified(llm, catalog):
    answer, vfs = recommend(
        "lightweight laptop under $1200 with good battery",
        llm=llm, catalog=catalog,
    )
    assert isinstance(answer, FinalAnswer)
    assert 1 <= len(answer.recommendations) <= 5
    assert answer.constraint_verified
    for rec in answer.recommendations:
        product = catalog.get(rec.product_id)
        assert product is not None, "no hallucinated ids"
        assert product.price <= 1200, "hard budget constraint enforced"
    # all pipeline artifacts landed in the VFS
    assert vfs.exists("/plan/plan.json")
    assert vfs.exists("/rankings/budget.json")
    assert vfs.exists("/final/answer.json")


def test_bounded_self_correction(llm, catalog):
    """Even with max_revisions=0 the graph must terminate with an answer."""
    answer, _ = recommend("headphones under $100", llm=llm,
                          catalog=catalog, max_revisions=0)
    assert isinstance(answer, FinalAnswer)


def test_parallel_specialists_all_ran(llm, catalog):
    _, vfs = recommend("4k monitor under $600", llm=llm, catalog=catalog)
    for agent in ("budget", "feature", "review"):
        assert vfs.exists(f"/rankings/{agent}.json")
