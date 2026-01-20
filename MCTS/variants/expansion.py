"""
Expansion Variants - 扩展阶段变体

实现不同的节点扩展策略，控制搜索树的宽度。

变体:
    - ProgressiveWideningMCTS: 渐进式宽度扩展
    - ConfidenceAwareMCTS: 置信度感知扩展
"""

import math
from typing import Optional, List, TypeVar
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcts import MCTS, Node, Environment

State = TypeVar("State")
Action = TypeVar("Action")


class ProgressiveWideningMCTS(MCTS[State, Action]):
    """渐进式宽度扩展 (Progressive Widening)
    
    公式 (AlphaGo / Stochastic Beam):
        |children(s)| <= C * N(s)^α
    
    特点:
        - 初始只生成 1 个子节点
        - 随着访问次数增加，逐渐解锁更多子节点
        - 大幅节省 Token 成本
    """
    
    def __init__(
        self,
        env: Environment[State, Action],
        c_param: float = 1.414,
        widening_constant: float = 1.0,
        widening_alpha: float = 0.5,
        max_depth: Optional[int] = None,
    ):
        super().__init__(env, c_param=c_param, max_depth=max_depth)
        self.widening_constant = widening_constant  # C
        self.widening_alpha = widening_alpha  # α
    
    def expand(self, node: Node[State, Action]) -> Node[State, Action]:
        """渐进式扩展: 根据访问次数决定子节点数"""
        # 计算允许的最大子节点数
        max_children = self._compute_max_children(node)
        
        # 如果已达上限，不再扩展
        if len(node.children) >= max_children:
            return node if not node.children else node.children[0]
        
        # 获取可用动作
        actions = self.env.get_actions(node.state)
        existing_actions = {child.action for child in node.children}
        available_actions = [a for a in actions if a not in existing_actions]
        
        if not available_actions:
            return node
        
        # 选择先验最高的动作展开
        action = self._select_best_prior_action(node, available_actions)
        new_state = self.env.step(node.state, action)
        prior = self.env.get_prior(node.state, action)
        
        child = Node(
            state=new_state,
            action=action,
            prior=prior,
            is_terminal=self.env.is_terminal(new_state),
        )
        return node.add_child(child)
    
    def _compute_max_children(self, node: Node) -> int:
        """计算当前允许的最大子节点数"""
        if node.visits == 0:
            return 1  # 首次访问，只允许 1 个子节点
        
        # |children| <= C * N^α
        max_children = int(self.widening_constant * (node.visits ** self.widening_alpha))
        return max(1, max_children)
    
    def _select_best_prior_action(
        self, 
        node: Node, 
        available_actions: List[Action]
    ) -> Action:
        """选择先验最高的动作"""
        best_action = available_actions[0]
        best_prior = self.env.get_prior(node.state, best_action)
        
        for action in available_actions[1:]:
            prior = self.env.get_prior(node.state, action)
            if prior > best_prior:
                best_prior = prior
                best_action = action
        
        return best_action


class ConfidenceAwareMCTS(MCTS[State, Action]):
    """置信度感知扩展 (Confidence-Aware Expansion)
    
    机制:
        - 仅当 Top-1 子节点得分低于阈值时，才触发扩展
        - 类似于 "不满意才重做"
    
    特点:
        - Token 消耗最省
        - 适合预算有限的场景
    """
    
    def __init__(
        self,
        env: Environment[State, Action],
        c_param: float = 1.414,
        satisfaction_threshold: float = 0.8,
        max_children: int = 5,
        max_depth: Optional[int] = None,
    ):
        super().__init__(env, c_param=c_param, max_depth=max_depth)
        self.satisfaction_threshold = satisfaction_threshold
        self.max_children = max_children
    
    def expand(self, node: Node[State, Action]) -> Node[State, Action]:
        """置信度感知扩展: 只有不满意时才扩展更多"""
        # 首次扩展
        if not node.children:
            return self._expand_one(node)
        
        # 检查最佳子节点是否满意
        best_child = max(node.children, key=lambda c: c.q_value if c.visits > 0 else 0)
        
        # 如果已达上限或满意，不再扩展
        if len(node.children) >= self.max_children:
            return best_child
        
        if best_child.visits > 0 and best_child.q_value >= self.satisfaction_threshold:
            return best_child
        
        # 不满意，尝试扩展新的子节点
        return self._expand_one(node)
    
    def _expand_one(self, node: Node) -> Node:
        """扩展一个新的子节点"""
        actions = self.env.get_actions(node.state)
        existing_actions = {child.action for child in node.children}
        available_actions = [a for a in actions if a not in existing_actions]
        
        if not available_actions:
            return node.children[0] if node.children else node
        
        # 随机选择一个未探索的动作
        import random
        action = random.choice(available_actions)
        new_state = self.env.step(node.state, action)
        prior = self.env.get_prior(node.state, action)
        
        child = Node(
            state=new_state,
            action=action,
            prior=prior,
            is_terminal=self.env.is_terminal(new_state),
        )
        return node.add_child(child)
