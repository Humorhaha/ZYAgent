from typing import Any

from .env import Env
from .state import Node

def _head_step(node: Node) -> Any:
    return node["sub_tasks"][node["cursor"]]


def _step_to_text(step: Any) -> str:
    if isinstance(step, str):
        return step.strip()

    if isinstance(step, dict):
        for k in ("action", "cmd", "command", "goal", "task", "subtask"):
            v = step.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()

    return str(step).strip()


def check_primitive(node: Node) -> bool:
    """
    primitive ⇔ head_step 在当前环境 admissible actions 中。
    没有可执行动作集合时，默认 False。
    """
    head = _step_to_text(_head_step(node))
    valid_actions = Env._admissible_commands()
    if not valid_actions:
        return False
    return head in valid_actions
