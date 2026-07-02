"""LangGraph wiring for the Deep-Agent Product Decision System.

Topology:

    planner ──(Send fan-out)──▶ specialist ×3 (parallel)
                                     │  (ranking_paths reducer fan-in)
                                     ▼
                               aggregator ──▶ critic ──▶ finalize ▶ END
                                     ▲            │
                                     └── revise ──┘   (bounded: max_revisions)

Heavy artifacts live in the VFS; the graph state (see state.GraphState)
carries only paths and control-flow counters, so checkpoints stay tiny.
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from .agents.aggregator import run_aggregator
from .agents.critic import TOP_K, run_critic
from .agents.planner import run_planner
from .agents.specialists import run_specialist
from .catalog import Catalog
from .llm import LLMClient
from .state import (Critique, FinalAnswer, GraphState, Plan, Ranking,
                    Recommendation, SubTask)
from .vfs import VirtualFileSystem


def build_graph(llm: LLMClient, vfs: VirtualFileSystem, catalog: Catalog):
    # ---------------------------------------------------------------- nodes
    def planner_node(state: GraphState) -> dict:
        plan_path = run_planner(llm, vfs, state["query"])
        return {"plan_path": plan_path, "revision": 0,
                "max_revisions": state.get("max_revisions", 2),
                "ranking_paths": []}

    def fan_out(state: GraphState) -> list[Send]:
        plan = vfs.read_model(state["plan_path"], Plan)
        return [
            Send("specialist", {"plan_path": state["plan_path"],
                                "subtask": st.model_dump()})
            for st in plan.subtasks
        ]

    def specialist_node(payload: dict) -> dict:
        subtask = SubTask.model_validate(payload["subtask"])
        path = run_specialist(llm, vfs, catalog, payload["plan_path"], subtask)
        return {"ranking_paths": [path]}  # reducer-merged fan-in

    def aggregator_node(state: GraphState) -> dict:
        critique_path = state.get("critique_path")
        fused_path = run_aggregator(
            vfs, state["ranking_paths"], state.get("revision", 0), critique_path
        )
        return {"fused_path": fused_path}

    def critic_node(state: GraphState) -> dict:
        critique_path = run_critic(
            vfs, catalog, state["plan_path"], state["fused_path"],
            state.get("revision", 0),
        )
        critique = vfs.read_model(critique_path, Critique)
        return {"critique_path": critique_path, "approved": critique.approved,
                "revision": state.get("revision", 0) + 1}

    def route_after_critic(state: GraphState) -> str:
        if state.get("approved") or state["revision"] > state.get("max_revisions", 2):
            return "finalize"
        return "aggregator"  # bounded self-correction round

    def finalize_node(state: GraphState) -> dict:
        plan = vfs.read_model(state["plan_path"], Plan)
        fused = vfs.read_model(state["fused_path"], Ranking)
        recs = []
        # Hard guarantee: even if the bounded loop exhausted its revisions,
        # never surface a product that fails a verified hard constraint.
        for cand in fused.candidates:
            if len(recs) >= TOP_K:
                break
            p = catalog.get(cand.product_id)
            if p is None:
                continue
            if not all(c.check(p) for c in plan.hard_constraints):
                continue
            recs.append(Recommendation(
                product_id=p.id, name=p.name, price=p.price, rating=p.rating,
                fused_score=cand.score, why=cand.rationale or "high fused score",
            ))
        answer = FinalAnswer(
            query=plan.query, recommendations=recs,
            revisions_used=state["revision"] - 1 if state.get("approved")
            else state["revision"],
            constraint_verified=bool(state.get("approved")),
        )
        return {"final_path": vfs.write_model("/final/answer.json", answer)}

    # ---------------------------------------------------------------- graph
    g = StateGraph(GraphState)
    g.add_node("planner", planner_node)
    g.add_node("specialist", specialist_node)
    g.add_node("aggregator", aggregator_node)
    g.add_node("critic", critic_node)
    g.add_node("finalize", finalize_node)

    g.add_edge(START, "planner")
    g.add_conditional_edges("planner", fan_out, ["specialist"])
    g.add_edge("specialist", "aggregator")
    g.add_edge("aggregator", "critic")
    g.add_conditional_edges("critic", route_after_critic,
                            {"aggregator": "aggregator", "finalize": "finalize"})
    g.add_edge("finalize", END)
    return g.compile()


def recommend(query: str, llm: LLMClient | None = None,
              catalog: Catalog | None = None,
              vfs: VirtualFileSystem | None = None,
              max_revisions: int = 2) -> tuple[FinalAnswer, VirtualFileSystem]:
    """Convenience entry point: run the full graph for one query."""
    llm = llm or LLMClient()
    catalog = catalog or Catalog()
    vfs = vfs or VirtualFileSystem()
    app = build_graph(llm, vfs, catalog)
    out = app.invoke({"query": query, "max_revisions": max_revisions})
    answer = vfs.read_model(out["final_path"], FinalAnswer)
    return answer, vfs
