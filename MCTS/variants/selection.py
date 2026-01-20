"""
Selection Variants - 选择阶段变体

实现不同的 UCT 公式变体，用于平衡 Exploitation 和 Exploration。

变体:
    - PUCTMCTS: AlphaZero 风格的 PUCT 公式
    - EntropyUCTMCTS: 基于熵的动态探索
"""

import math
from typing import Optional, List, TypeVar
import sys
import os

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcts import MCTS, Node, Environment

State = TypeVar("State")
Action = TypeVar("Action")


class PUCTMCTS(MCTS[State, Action]):
    """PUCT (Polynomial Upper Confidence Trees) 变体
    
    公式 (AlphaZero / LATS):
        a_t = argmax_a [ Q(s,a) + c_puct * P(s,a) * sqrt(sum_b N(s,b)) / (1 + N(s,a)) ]
    
    特点:
        - 使用 Prior P(s,a) 引导探索
        - 适合有先验信息的场景 (如 LLM Logprobs)
    """
    
    def __init__(
        self,
        env: Environment[State, Action],
        c_puct: float = 2.0,
        max_depth: Optional[int] = None,
    ):
        super().__init__(env, c_param=c_puct, max_depth=max_depth)
        self.c_puct = c_puct
    
    def select_action(self, node: Node[State, Action]) -> Node[State, Action]:
        """使用 PUCT 公式选择子节点"""
        if not node.children:
            return node
        
        total_visits = sum(child.visits for child in node.children)
        
        def puct_score(child: Node) -> float:
            if child.visits == 0:
                return float('inf')
            
            # Q(s, a): 平均价值
            q_value = child.q_value
            
            # P(s, a): 先验概率
            prior = child.prior
            
            # PUCT 探索项
            exploration = self.c_puct * prior * math.sqrt(total_visits) / (1 + child.visits)
            
            return q_value + exploration
        
        return max(node.children, key=puct_score)


class EntropyUCTMCTS(MCTS[State, Action]):
    """基于熵的动态探索 UCT 变体
    
    公式 (TS-LLM / Engineering Optimization):
        a_t = argmax_a [ Q(s,a) + λ * H(P(·|s)) + c * sqrt(N(s)) / (1 + N(s,a)) ]
    
    特点:
        - 当策略熵 H 较高 (模型困惑) 时，自动增加探索力度
        - 适合 LLM 推理场景，能自适应不确定性
    """
    
    def __init__(
        self,
        env: Environment[State, Action],
        c_param: float = 1.414,
        lambda_entropy: float = 0.5,
        max_depth: Optional[int] = None,
    ):
        super().__init__(env, c_param=c_param, max_depth=max_depth)
        self.lambda_entropy = lambda_entropy
    
    def select_action(self, node: Node[State, Action]) -> Node[State, Action]:
        """使用带熵项的 UCT 公式选择子节点"""
        if not node.children:
            return node
        
        # 计算策略熵 H = -sum(p * log(p))
        entropy = self._compute_entropy(node)
        
        def entropy_uct_score(child: Node) -> float:
            if child.visits == 0:
                return float('inf')
            
            q_value = child.q_value
            
            # 熵奖励: 越困惑，越鼓励探索
            entropy_bonus = self.lambda_entropy * entropy
            
            # 标准 UCT 探索项
            exploration = self.c_param * math.sqrt(math.log(node.visits) / child.visits)
            
            return q_value + entropy_bonus + exploration
        
        return max(node.children, key=entropy_uct_score)
    
    def _compute_entropy(self, node: Node) -> float:
        """计算子节点先验分布的熵"""
        if not node.children:
            return 0.0
        
        priors = [child.prior for child in node.children]
        total = sum(priors)
        if total == 0:
            return 0.0
        
        # 归一化
        probs = [p / total for p in priors]
        
        # 计算熵
        entropy = 0.0
        for p in probs:
            if p > 0:
                entropy -= p * math.log(p + 1e-10)
        
        return entropy
