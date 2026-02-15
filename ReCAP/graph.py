from typing import Optional
from .state import State, Phase
from langgraph.graph import StateGraph, END
from .env import Env
from .nodes import (
    init_node,
    plan_pi,
    classify,
    act,
    finalize,
    push,
    refine,
)

def _route_by_phase(state: State) -> str:
    """
    Router reads state["phase"] and returns the next node name (graph node id).
    """
    ph = state["phase"]

    if ph == Phase.PLAN:
        return "plan_pi"
    if ph == Phase.CLASSIFY:
        return "classify"
    if ph == Phase.PUSH:
        return "push"
    if ph == Phase.ACT:
        return "act"
    if ph == Phase.REFINE:
        return "refine"
    if ph == Phase.FINAL:
        return "finalize"

    if ph == Phase.INIT:
        return "init_node"

    # 未知 phase 直接终止或抛错
    return "finalize"







def build_agent(env: Optional[Env] = None):
    """
    Build a LangGraph agent from your current node functions.

    Design:
    - Every node function mutates state and sets state["phase"].
    - The graph uses one router `_route_by_phase` to jump to the next node.
    - env is injected via closures for nodes that need it (init_node, classify, act, refine_rho if needed).
    """
    g = StateGraph(State)

    def _init(s: State) -> State:
        return init_node(s, env=env)

    def _classify(s: State) -> State:
        return classify(s, env=env)

    def _act(s: State) -> State:
        return act(s, env=env)

    def _refine(s: State) -> State:
        return refine(s, env=env)

    g.add_node("init_node", _init)
    g.add_node("plan_pi", plan_pi)
    g.add_node("classify", _classify)
    g.add_node("push", push)
    g.add_node("act", _act)
    g.add_node("refine", _refine)
    g.add_node("finalize", finalize)

    g.set_entry_point("init_node")


    for n in ["init_node", "plan_pi", "classify", "push", "act", "refine"]:
        g.add_conditional_edges(
            n,
            _route_by_phase,
            {
                "init_node": "init_node",
                "plan_pi": "plan_pi",
                "classify": "classify",
                "push": "push",
                "act": "act",
                "refine": "refine",
                "finalize": "finalize",
            },
        )


    g.add_edge("finalize", END)

    return g.compile()