import os
from pathlib import Path

from dotenv import load_dotenv

# 先加载 .env，确保 LangSmith tracing 在导入 langgraph 前已生效
_ENV_PATH = Path(__file__).parent / ".env"
if _ENV_PATH.exists():
    load_dotenv(_ENV_PATH)

try:
    from .graph import build_agent
    from .env import Env
    from .state import Phase
except ImportError:
    # 兼容 `python ReCAP/run.py` 直接运行
    import sys

    _HERE = os.path.dirname(os.path.abspath(__file__))
    _ROOT = os.path.dirname(_HERE)
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)

    from ReCAP.graph import build_agent
    from ReCAP.env import Env
    from ReCAP.state import Phase

import time
import textwrap


def _line(title: str = "", width: int = 88, char: str = "=") -> str:
    if not title:
        return char * width
    text = f" {title} "
    pad = max(0, width - len(text))
    left = pad // 2
    right = pad - left
    return f"{char * left}{text}{char * right}"


def _short(text, n: int = 180) -> str:
    s = str(text or "").replace("\n", " ").strip()
    return s if len(s) <= n else s[: n - 3] + "..."


def _fmt_phase(phase) -> str:
    return getattr(phase, "value", str(phase))


def _print_report(root_task: str, final_state: dict, duration_s: float) -> None:
    context = final_state.get("context_shared", []) or []
    stack = final_state.get("stack", []) or []

    print(_line("ReCAP Run Report"))
    print(f"Task        : {root_task}")
    print(f"Halt Reason : {final_state.get('halt_reason', 'N/A')}")
    print(f"Final Phase : {_fmt_phase(final_state.get('phase', 'N/A'))}")
    print(f"Elapsed     : {duration_s:.2f}s")
    print(f"Stack Size  : {len(stack)}")
    print(f"Context Size: {len(context)}")
    print(_line())

    print(_line("Final Answer", char="-"))
    print(textwrap.fill(final_state.get("final_answer", ""), width=88))
    print(_line(char="-"))

    if context:
        print(_line("Context Trace", char="-"))
        for i, node in enumerate(context, 1):
            goal = _short(node.get("goal", ""))
            thought = _short(node.get("thought", ""))
            cursor = node.get("cursor", 0)
            subtasks = node.get("sub_tasks", []) or []
            print(f"[{i:02d}] goal={goal}")
            print(f"     cursor={cursor}  subtasks={len(subtasks)}")
            print(f"     thought={thought}")
        print(_line(char="-"))

    obs = final_state.get("observation", "")
    if obs:
        print(_line("Last Observation", char="-"))
        print(textwrap.fill(_short(obs, 600), width=88))
        print(_line(char="-"))


env = Env()
agent = build_agent(env)
root_task = Env.peek_query() or "Solve the current ALFWorld task."

state = {
    "root_task": root_task,
    "stack": [],
    "context_shared": [],    
    "observation": "",
    "phase": Phase.INIT,
    "final_answer": "",
}

start = time.time()
final_state = agent.invoke(state)
duration = time.time() - start
pic = agent.get_graph().draw_mermaid_png(output_file_path="recap_agent.png")
_print_report(root_task, final_state, duration)
