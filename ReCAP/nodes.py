from typing import Optional

from . import prompt
from .env import Env
from .llm import chat
from .state import Node, Phase, State
from .utils import check_primitive


def _step_to_text(step) -> str:
    if isinstance(step, str):
        return step.strip()
    if isinstance(step, dict):
        for k in ("goal", "action", "cmd", "command", "task", "subtask"):
            v = step.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
    return str(step).strip()


def _remaining_subtasks_text(tasks) -> str:
    return str(tasks)


def init_node(state: State, env: Optional[Env] = None):
    """
    初始化 node，下一阶段进入 CLASSIFY。
    """
    system_prompt = prompt.system_prompt
    env_runner = env if env is not None else Env
    init_obs = env_runner.reset(state["root_task"])
    root_task_decomposition_prompt = prompt.ROOT_TASK_DECOMPOSITION_PROMPT.format(
        system_prompt=system_prompt,
        rule=env.rule if env else "",
        init_obs=init_obs,
        goal=state["root_task"],
    )

    response = chat(root_task_decomposition_prompt)
    node = Node()
    node["goal"] = state["root_task"]
    node["thought"] = response["thought"]
    node["sub_tasks"] = response["sub_tasks"]
    node["cursor"] = 0

    state["stack"] = [node]
    state["context_shared"] = list(state["stack"])
    state["observation"] = init_obs
    state["phase"] = Phase.CLASSIFY
    state["final_answer"] = ""
    return state


def plan_pi(state: State):
    """
    phase=PLAN，生成当前节点的子任务后进入 CLASSIFY。
    """
    node = state["stack"][-1]
    plan_prompt = prompt.DOWNWARD_PROMPT.format(goal=node["goal"])
    response = chat(plan_prompt)
    node["thought"] = response["thought"]
    node["sub_tasks"] = response["sub_tasks"]
    node["cursor"] = 0
    state["context_shared"] = list(state["stack"])
    state["phase"] = Phase.CLASSIFY
    return state


def classify(state: State, env: Optional[Env] = None):
    """
    phase=CLASSIFY: 选择 ACT(primitive) 或 PUSH(subgoal)。
    """
    node = state["stack"][-1]
    state["context_shared"] = list(state["stack"])
    if node["cursor"] >= len(node["sub_tasks"]):
        state["phase"] = Phase.FINAL if len(state["stack"]) == 1 else Phase.REFINE
        return state

    state["phase"] = Phase.ACT if check_primitive(node) else Phase.PUSH
    return state


def act(state: State, env: Optional[Env] = None):
    """
    phase=ACT: 执行 primitive，并推进 cursor，随后进入 REFINE。
    """
    node = state["stack"][-1]
    head_step = node["sub_tasks"][node["cursor"]]
    env_runner = env if env is not None else Env
    observation = env_runner.step(_step_to_text(head_step))
    state["observation"] = observation
    node["cursor"] += 1
    state["context_shared"] = list(state["stack"])
    # 环境已完成时，直接终止图执行，避免继续循环规划
    if bool(getattr(env_runner, "_last_done", False)):
        state["halt_reason"] = "env_done"
        state["phase"] = Phase.FINAL
        return state

    state["phase"] = Phase.REFINE
    return state


def refine(state: State, env: Optional[Env] = None):
    """
    回溯并更新父节点计划；root 节点则回到 CLASSIFY 进行收敛检查。
    """
    node = state["stack"][-1]
    obs = state.get("observation", "")

    # primitive 后的 ρ：在当前节点上显式应用 S[1:]，不进行 pop
    if obs and node["cursor"] > 0 and node["cursor"] <= len(node["sub_tasks"]):
        done_task_name = _step_to_text(node["sub_tasks"][node["cursor"] - 1])
        remaining_tasks = node["sub_tasks"][node["cursor"] :]
        node["sub_tasks"] = remaining_tasks
        node["cursor"] = 0
        state["context_shared"] = list(state["stack"])

        up_prompt = prompt.LEAF_BACKTRACK_PROMPT.format(
            done_task_name=done_task_name,
            obs=obs,
            previous_stage_task_name=node["goal"],
            previous_stage_think=node["thought"],
            remaining_subtask_str=_remaining_subtasks_text(remaining_tasks),
        )
        response = chat(up_prompt)
        node["thought"] = response["thought"]
        node["sub_tasks"] = response["sub_tasks"]
        node["cursor"] = 0
        state["context_shared"] = list(state["stack"])
        state["phase"] = Phase.CLASSIFY
        return state

    # subgoal 完成后回溯：pop child -> 更新 parent（显式 S[1:]）-> ρ(parent)
    if len(state["stack"]) > 1:
        child = state["stack"].pop()
        done_task_name = child["goal"]
        parent = state["stack"][-1]
        parent["cursor"] += 1
        remaining_tasks = parent["sub_tasks"][parent["cursor"] :]
        parent["sub_tasks"] = remaining_tasks
        parent["cursor"] = 0
        state["context_shared"] = list(state["stack"])

        up_prompt = prompt.NON_LEAF_BACKTRACK_PROMPT.format(
            done_task_name=done_task_name,
            previous_stage_task_name=parent["goal"],
            previous_stage_think=parent["thought"],
            remaining_subtask_str=_remaining_subtasks_text(remaining_tasks),
        )
        response = chat(up_prompt)
        parent["thought"] = response["thought"]
        parent["sub_tasks"] = response["sub_tasks"]
        parent["cursor"] = 0
        state["context_shared"] = list(state["stack"])
        state["phase"] = Phase.CLASSIFY
        return state

    state["context_shared"] = list(state["stack"])
    state["phase"] = Phase.CLASSIFY
    return state


def push(state: State):
    """
    phase=PUSH: 将 head subgoal 入栈，下一阶段进入 PLAN。
    """
    parent = state["stack"][-1]
    head = parent["sub_tasks"][parent["cursor"]]

    child = Node()
    child["goal"] = _step_to_text(head)
    child["cursor"] = 0
    child["thought"] = ""
    child["sub_tasks"] = []

    state["stack"].append(child)
    state["phase"] = Phase.PLAN
    state["context_shared"] = list(state["stack"])
    return state


def finalize(state: State):
    assert len(state["stack"]) == 1, "finalize时stack长度不为1, 无法finalize"
    node = state["stack"][0]
    state["final_answer"] = node["thought"]
    state["halt_reason"] = "success"
    state["phase"] = Phase.FINAL
    state["context_shared"] = list(state["stack"])
    return state
