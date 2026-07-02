"""Pydantic-validated state models exchanged between agents.

Every artifact written to the shared Virtual File System (VFS) is one of
these models, serialized as JSON. Validation happens at every node
boundary, so a malformed LLM output fails fast instead of silently
corrupting downstream agents.
"""
from __future__ import annotations

import operator
from enum import Enum
from typing import Annotated, Any, Optional

from pydantic import BaseModel, Field, field_validator
from typing_extensions import TypedDict


# ---------------------------------------------------------------------------
# Domain models
# ---------------------------------------------------------------------------

class ConstraintOp(str, Enum):
    LTE = "lte"
    GTE = "gte"
    EQ = "eq"
    CONTAINS = "contains"


class Constraint(BaseModel):
    """A hard, verifiable constraint extracted from the user query."""
    field: str = Field(..., description="Product attribute the constraint applies to, e.g. 'price'")
    op: ConstraintOp
    value: Any

    def check(self, product: "Product") -> bool:
        actual = product.attributes.get(self.field, getattr(product, self.field, None))
        if actual is None:
            return False
        try:
            if self.op == ConstraintOp.LTE:
                return float(actual) <= float(self.value)
            if self.op == ConstraintOp.GTE:
                return float(actual) >= float(self.value)
            if self.op == ConstraintOp.EQ:
                return str(actual).lower() == str(self.value).lower()
            if self.op == ConstraintOp.CONTAINS:
                if isinstance(actual, (list, tuple)):
                    return any(str(self.value).lower() in str(a).lower() for a in actual)
                return str(self.value).lower() in str(actual).lower()
        except (TypeError, ValueError):
            return False
        return False


class Preference(BaseModel):
    """A soft preference — influences ranking but is not pass/fail."""
    description: str
    weight: float = Field(1.0, ge=0.0, le=5.0)


class SubTask(BaseModel):
    """A unit of work the planner assigns to a specialist agent."""
    agent: str = Field(..., description="Specialist agent name: budget | feature | review")
    instruction: str
    constraints: list[Constraint] = Field(default_factory=list)
    preferences: list[Preference] = Field(default_factory=list)


class Plan(BaseModel):
    """Planner output: the query decomposed into parallel sub-tasks."""
    query: str
    category: Optional[str] = None
    hard_constraints: list[Constraint] = Field(default_factory=list)
    preferences: list[Preference] = Field(default_factory=list)
    subtasks: list[SubTask] = Field(min_length=1)

    @field_validator("subtasks")
    @classmethod
    def _known_agents(cls, v: list[SubTask]) -> list[SubTask]:
        known = {"budget", "feature", "review"}
        for st in v:
            if st.agent not in known:
                raise ValueError(f"Unknown specialist agent '{st.agent}' (must be one of {known})")
        return v


class Product(BaseModel):
    id: str
    name: str
    category: str
    price: float = Field(..., ge=0)
    rating: float = Field(..., ge=0, le=5)
    review_count: int = Field(..., ge=0)
    attributes: dict[str, Any] = Field(default_factory=dict)
    description: str = ""


class ScoredCandidate(BaseModel):
    product_id: str
    score: float = Field(..., ge=0.0, le=1.0)
    rationale: str = ""


class Ranking(BaseModel):
    """A specialist agent's ranked view over the candidate pool."""
    agent: str
    candidates: list[ScoredCandidate]

    @field_validator("candidates")
    @classmethod
    def _sorted_desc(cls, v: list[ScoredCandidate]) -> list[ScoredCandidate]:
        return sorted(v, key=lambda c: c.score, reverse=True)


class Violation(BaseModel):
    product_id: str
    constraint: Constraint
    detail: str


class Critique(BaseModel):
    """Critic output: constraint-verification report on the merged top-k."""
    approved: bool
    violations: list[Violation] = Field(default_factory=list)
    excluded_product_ids: list[str] = Field(default_factory=list)
    weight_adjustments: dict[str, float] = Field(default_factory=dict)
    notes: str = ""


class Recommendation(BaseModel):
    product_id: str
    name: str
    price: float
    rating: float
    fused_score: float
    why: str


class FinalAnswer(BaseModel):
    query: str
    recommendations: list[Recommendation]
    revisions_used: int
    constraint_verified: bool


# ---------------------------------------------------------------------------
# LangGraph shared state
# ---------------------------------------------------------------------------

class GraphState(TypedDict, total=False):
    """The (small) state that flows through the LangGraph graph.

    Heavy artifacts live in the VFS; the graph state carries only the
    query, VFS paths, control-flow counters, and reducer-merged rankings.
    """
    query: str
    plan_path: str
    # Parallel specialists append their VFS ranking paths here; the
    # operator.add reducer makes the fan-in safe under concurrency.
    ranking_paths: Annotated[list[str], operator.add]
    fused_path: str
    critique_path: str
    revision: int
    max_revisions: int
    approved: bool
    final_path: str
