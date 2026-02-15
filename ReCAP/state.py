from typing_extensions import TypedDict
from typing import List
from enum import Enum


class Node(TypedDict):
    goal: str   #当前子任务目标
    thought: str #当前子任务分解思考过程
    sub_tasks: List[str]    #当前子任务分解后的子任务列表
    cursor: int #子任务下标


class Phase(Enum):
    """
    - INIT       : initialize root node and state containers
    - PLAN       : π plan for current node (thought + ordered sub_tasks)
    - CLASSIFY   : classify head step; set phase to ACT (primitive) or PUSH (subgoal)
    - ACT        : execute primitive head step via env/tool
    - OBSERVE    : apply observation; advance cursor; append minimal message if desired
    - REFINE     : ρ refine remaining plan/thought based on observation; choose next phase
    - PUSH       : push a new child node for subgoal head step
    - POP        : pop completed child node and resume parent
    - REINJECT   : reinject parent's thought + remaining subtasks into shared messages
    - FINAL      : produce final_answer; terminal phase
    """
    INIT = "init"
    PLAN = "plan"
    CLASSIFY = "classify"
    ACT = "act"
    OBSERVE = "observe"
    REFINE = "refine"
    PUSH = "push"
    POP = "pop"
    REINJECT = "reinject"
    FINAL = "final"



class State(TypedDict):
    """
    Global state passed across LangGraph nodes.

    Fields:
    - root_task: The root user task.
    - stack: Recursion stack. stack[-1] is the current active frame.
    - context_shared: Shared LLM context window (Node list).
    - observation: The latest environment/tool observation.
    - phase: Current phase for conditional edge routing.
    - halt_reason: If done, the reason for halting (e.g., "success", "timeout", ...).
    - final_answer: Final output for the user.
    """
    root_task: str
    stack: List[Node]   #递归调用stack
    context_shared: List[Node] #全局上下文
    observation: str #当前观察结果
    phase: Phase #当前阶段, 用于conditional edge routing
    halt_reason: str #完成任务的原因
    final_answer: str 
