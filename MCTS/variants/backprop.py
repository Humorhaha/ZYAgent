"""
Backpropagation Variants - 反向传播阶段变体

实现不同的价值回传策略，支持文本反馈。

变体:
    - ReflexionMCTS: 带文本反馈的反向传播 (LATS / ExpeL)
"""

from typing import Optional, List, TypeVar, Dict, Any
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcts import MCTS, Node, Environment

State = TypeVar("State")
Action = TypeVar("Action")


class ReflexionMCTS(MCTS[State, Action]):
    """带文本反馈的反向传播 (Reflexion Backpropagation)
    
    公式 (LATS / ExpeL):
        M(s) ← M(s) ∪ {feedback from failed leaf}
    
    机制:
        - 当叶子节点失败时，提取错误摘要
        - 将摘要回传给父节点存储在 feedback_buffer
        - 下次扩展时，LLM 会看到兄弟节点的失败原因
    
    特点:
        - 实现了 "经验学习"，避免重蹈覆辙
        - 适合代码生成、逻辑推理等需要纠错的场景
    """
    
    def __init__(
        self,
        env: Environment[State, Action],
        c_param: float = 1.414,
        failure_threshold: float = 0.3,  # 低于此分数视为失败
        max_feedback_per_node: int = 3,  # 每个节点最多存储的反馈数
        max_depth: Optional[int] = None,
    ):
        super().__init__(env, c_param=c_param, max_depth=max_depth)
        self.failure_threshold = failure_threshold
        self.max_feedback_per_node = max_feedback_per_node
    
    def backpropagate(self, node: Node[State, Action], reward: float) -> None:
        """带反馈的反向传播"""
        # 判断是否失败
        is_failure = reward < self.failure_threshold
        
        # 如果失败，提取反馈
        feedback = None
        if is_failure:
            feedback = self._extract_feedback(node)
        
        # 回传更新
        current = node
        depth = 0
        while current is not None:
            # 更新统计量
            current.visits += 1
            current.value += reward
            
            # 如果是失败路径，将反馈传给父节点
            if feedback and current.parent is not None:
                self._add_feedback_to_parent(current.parent, feedback, depth)
            
            current = current.parent
            depth += 1
    
    def _extract_feedback(self, node: Node) -> str:
        """从失败节点提取反馈信息
        
        默认实现: 返回状态的字符串表示。
        子类应重写以提取更有意义的错误摘要。
        """
        # 检查是否有 LLM 生成的错误信息
        if "error_message" in node.metadata:
            return node.metadata["error_message"]
        
        if "llm_feedback" in node.metadata:
            return node.metadata["llm_feedback"]
        
        # 默认返回动作信息
        return f"Action '{node.action}' led to low reward ({node.q_value:.2f})"
    
    def _add_feedback_to_parent(
        self, 
        parent: Node, 
        feedback: str, 
        depth: int
    ) -> None:
        """将反馈添加到父节点"""
        # 初始化 feedback_buffer
        if "feedback_buffer" not in parent.metadata:
            parent.metadata["feedback_buffer"] = []
        
        buffer: List[Dict[str, Any]] = parent.metadata["feedback_buffer"]
        
        # 添加新反馈
        feedback_entry = {
            "feedback": feedback,
            "depth": depth,  # 反馈来源的深度
        }
        buffer.append(feedback_entry)
        
        # 限制缓冲区大小
        if len(buffer) > self.max_feedback_per_node:
            # 保留最近的反馈
            parent.metadata["feedback_buffer"] = buffer[-self.max_feedback_per_node:]
    
    def get_feedback_for_expansion(self, node: Node) -> List[str]:
        """获取用于扩展时注入 Prompt 的反馈列表
        
        调用方可以在生成新子节点时，将这些反馈注入到 LLM Prompt 中：
        "Previous attempts failed because: {feedback}. Please try a different approach."
        """
        buffer = node.metadata.get("feedback_buffer", [])
        return [entry["feedback"] for entry in buffer]
    
    def expand(self, node: Node[State, Action]) -> Node[State, Action]:
        """扩展时注入反馈信息"""
        # 获取之前的失败反馈
        feedbacks = self.get_feedback_for_expansion(node)
        
        # 调用基类扩展
        child = super().expand(node)
        
        # 将反馈信息附加到新子节点，供 LLM 生成时参考
        if feedbacks:
            child.metadata["parent_feedbacks"] = feedbacks
        
        return child


class FullTraceReflexionMCTS(ReflexionMCTS):
    """完整轨迹反思变体 (Full Trace Reflexion)
    
    与 ReflexionMCTS 的区别:
        - 回传完整的失败轨迹，而不仅仅是摘要
        - 信息量更大，但 Context 消耗巨大
        - 适合短逻辑链的场景
    """
    
    def _extract_feedback(self, node: Node) -> str:
        """提取完整的失败轨迹"""
        trace_parts = []
        current = node
        
        while current is not None:
            if current.action is not None:
                trace_parts.append(f"Step: {current.action}")
            current = current.parent
        
        trace_parts.reverse()
        
        # 构建完整轨迹
        trace = " -> ".join(trace_parts)
        
        # 添加失败原因
        reason = node.metadata.get("error_message", "Unknown failure")
        
        return f"Failed trace: [{trace}]. Reason: {reason}"
