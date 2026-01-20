"""
Monte Carlo Tree Search (MCTS) - 基类实现

设计理念:
    本模块提供一个朴素的、可扩展的 MCTS 框架。
    通过继承和重写关键方法，可以实现多种变体 (UCT, PUCT, Semantic-UCT 等)。
    
核心结构:
    - Node: 搜索树的节点，存储状态、访问次数、价值等信息。
    - MCTS: 搜索控制器，负责执行 Selection -> Expansion -> Evaluation -> Backpropagation 循环。
    
扩展点 (Extension Hooks):
    - select_action(): 实现不同的 UCT/PUCT 公式变体。
    - expand(): 实现不同的扩展策略 (Fixed-k, Progressive Widening)。
    - evaluate(): 实现不同的评估策略 (Rollout, LLM-Scoring, PRM)。
    - backpropagate(): 实现不同的反馈传递 (Scalar-only, Reflexion)。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Generic, TypeVar, Tuple
import math
import random


# 泛型: 状态和动作类型
State = TypeVar("State")
Action = TypeVar("Action")


@dataclass
class Node(Generic[State, Action]):
    """MCTS 树节点
    
    Attributes:
        state: 当前状态
        action: 导致此状态的动作 (根节点为 None)
        parent: 父节点
        children: 子节点列表
        visits: 访问次数 N(s)
        value: 累计价值 (用于计算 Q(s) = value / visits)
        prior: 先验概率 P(s, a) (PUCT 使用)
        is_terminal: 是否为终止节点
        metadata: 额外信息 (如 LLM 反馈、错误日志等)
    """
    state: State
    action: Optional[Action] = None
    parent: Optional["Node"] = None
    children: List["Node"] = field(default_factory=list)
    visits: int = 0
    value: float = 0.0
    prior: float = 1.0  # 默认均匀先验
    is_terminal: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def q_value(self) -> float:
        """计算平均价值 Q(s) = value / visits"""
        if self.visits == 0:
            return 0.0
        return self.value / self.visits
    
    @property
    def is_fully_expanded(self) -> bool:
        """检查是否已完全展开 (需要由子类根据动作空间判断)"""
        # 默认实现: 如果有子节点就认为已展开
        # 实际应用中需要根据可用动作数判断
        return len(self.children) > 0
    
    @property
    def is_leaf(self) -> bool:
        """是否为叶子节点"""
        return len(self.children) == 0
    
    def add_child(self, child: "Node") -> "Node":
        """添加子节点"""
        child.parent = self
        self.children.append(child)
        return child
    
    def best_child(self, c_param: float = 1.414) -> Optional["Node"]:
        """选择最佳子节点 (UCT 公式)
        
        UCT = Q(s, a) + c * sqrt(ln(N(parent)) / N(s, a))
        
        Args:
            c_param: 探索常数 (UCT 的 c)
            
        Returns:
            UCT 值最高的子节点
        """
        if not self.children:
            return None
        
        def uct_score(child: Node) -> float:
            if child.visits == 0:
                return float('inf')  # 未访问的节点优先
            exploitation = child.q_value
            exploration = c_param * math.sqrt(math.log(self.visits) / child.visits)
            return exploitation + exploration
        
        return max(self.children, key=uct_score)
    
    def __repr__(self) -> str:
        return f"Node(visits={self.visits}, value={self.value:.2f}, q={self.q_value:.3f}, children={len(self.children)})"


class Environment(ABC, Generic[State, Action]):
    """环境接口 - 定义状态转移和奖励
    
    MCTS 需要一个环境来:
    1. 获取当前状态的合法动作
    2. 执行动作并返回新状态
    3. 判断是否终止
    4. 获取终止状态的奖励
    """
    
    @abstractmethod
    def get_actions(self, state: State) -> List[Action]:
        """获取当前状态的合法动作列表"""
        ...
    
    @abstractmethod
    def step(self, state: State, action: Action) -> State:
        """执行动作，返回新状态"""
        ...
    
    @abstractmethod
    def is_terminal(self, state: State) -> bool:
        """检查是否为终止状态"""
        ...
    
    @abstractmethod
    def get_reward(self, state: State) -> float:
        """获取终止状态的奖励 (仅在终止状态调用)"""
        ...
    
    def get_prior(self, state: State, action: Action) -> float:
        """获取动作的先验概率 (PUCT 使用)
        
        默认返回均匀分布。子类可重写以使用 LLM Logprobs。
        """
        actions = self.get_actions(state)
        return 1.0 / len(actions) if actions else 1.0


class MCTS(ABC, Generic[State, Action]):
    """Monte Carlo Tree Search 基类
    
    核心流程:
        1. Selection: 从根节点选择到叶子节点
        2. Expansion: 展开叶子节点
        3. Evaluation: 评估新节点 (Rollout 或 LLM 打分)
        4. Backpropagation: 回传价值更新
        
    使用示例:
        ```python
        env = MyEnvironment()
        mcts = VanillaMCTS(env, c_param=1.414)
        
        # 搜索
        best_action = mcts.search(initial_state, num_simulations=100)
        ```
    """
    
    def __init__(
        self,
        env: Environment[State, Action],
        c_param: float = 1.414,
        max_depth: Optional[int] = None,
    ):
        """
        Args:
            env: 环境接口
            c_param: 探索常数 (UCT 公式中的 c)
            max_depth: 最大搜索深度 (None 表示无限制)
        """
        self.env = env
        self.c_param = c_param
        self.max_depth = max_depth
        self.root: Optional[Node[State, Action]] = None
    
    def search(
        self,
        initial_state: State,
        num_simulations: int = 100,
    ) -> Action:
        """执行 MCTS 搜索
        
        Args:
            initial_state: 初始状态
            num_simulations: 模拟次数
            
        Returns:
            最佳动作
        """
        # 初始化根节点
        self.root = Node(state=initial_state)
        
        for _ in range(num_simulations):
            # 1. Selection: 选择到叶子节点
            node = self._select(self.root)
            
            # 2. Expansion: 展开节点
            if not node.is_terminal and not self._is_max_depth(node):
                node = self._expand(node)
            
            # 3. Evaluation: 评估节点
            reward = self._evaluate(node)
            
            # 4. Backpropagation: 回传更新
            self._backpropagate(node, reward)
        
        # 返回访问次数最多的动作 (Robust Child)
        return self._best_action(self.root)
    
    def _select(self, node: Node[State, Action]) -> Node[State, Action]:
        """Selection 阶段: 沿着树选择直到叶子节点
        
        使用 UCT 公式选择子节点。
        """
        while not node.is_leaf and not node.is_terminal:
            node = self.select_action(node)
        return node
    
    def _expand(self, node: Node[State, Action]) -> Node[State, Action]:
        """Expansion 阶段: 展开叶子节点
        
        默认实现: 随机选择一个未探索的动作展开。
        子类可重写以实现 Progressive Widening 等策略。
        """
        return self.expand(node)
    
    def _evaluate(self, node: Node[State, Action]) -> float:
        """Evaluation 阶段: 评估节点价值
        
        默认实现: 随机 Rollout 到终止状态。
        子类可重写以实现 LLM 打分、PRM 等策略。
        """
        return self.evaluate(node)
    
    def _backpropagate(self, node: Node[State, Action], reward: float) -> None:
        """Backpropagation 阶段: 回传更新
        
        默认实现: 更新访问次数和累计价值。
        子类可重写以实现 Reflexion Feedback 等策略。
        """
        self.backpropagate(node, reward)
    
    def _is_max_depth(self, node: Node[State, Action]) -> bool:
        """检查是否达到最大深度"""
        if self.max_depth is None:
            return False
        depth = 0
        current = node
        while current.parent is not None:
            depth += 1
            current = current.parent
        return depth >= self.max_depth
    
    def _best_action(self, root: Node[State, Action]) -> Action:
        """选择最终动作 (访问次数最多的子节点)"""
        if not root.children:
            raise ValueError("Root has no children after search")
        best_child = max(root.children, key=lambda c: c.visits)
        return best_child.action
    
    # =====================================================================
    # 扩展点 (Extension Hooks) - 子类可重写以实现不同变体
    # =====================================================================
    
    def select_action(self, node: Node[State, Action]) -> Node[State, Action]:
        """选择子节点的策略 (可重写)
        
        默认: 标准 UCT 公式。
        变体: PUCT, Entropy-UCT, Semantic-UCT 等。
        """
        return node.best_child(self.c_param)
    
    def expand(self, node: Node[State, Action]) -> Node[State, Action]:
        """展开节点的策略 (可重写)
        
        默认: 展开所有可用动作。
        变体: Top-k, Progressive Widening, Confidence-aware 等。
        """
        actions = self.env.get_actions(node.state)
        
        # 过滤已有子节点对应的动作
        existing_actions = {child.action for child in node.children}
        available_actions = [a for a in actions if a not in existing_actions]
        
        if not available_actions:
            # 所有动作都已展开，返回当前节点
            return node
        
        # 选择一个动作展开
        action = self._select_expansion_action(node, available_actions)
        new_state = self.env.step(node.state, action)
        prior = self.env.get_prior(node.state, action)
        
        child = Node(
            state=new_state,
            action=action,
            prior=prior,
            is_terminal=self.env.is_terminal(new_state),
        )
        return node.add_child(child)
    
    def _select_expansion_action(
        self,
        node: Node[State, Action],
        available_actions: List[Action],
    ) -> Action:
        """选择要展开的动作 (默认随机)
        
        子类可重写以实现基于先验的选择。
        """
        return random.choice(available_actions)
    
    def evaluate(self, node: Node[State, Action]) -> float:
        """评估节点价值 (可重写)
        
        默认: 随机 Rollout。
        变体: LLM-Scoring, Process Reward Model 等。
        """
        if node.is_terminal:
            return self.env.get_reward(node.state)
        
        # Random Rollout
        state = node.state
        while not self.env.is_terminal(state):
            actions = self.env.get_actions(state)
            if not actions:
                break
            action = random.choice(actions)
            state = self.env.step(state, action)
        
        return self.env.get_reward(state)
    
    def backpropagate(self, node: Node[State, Action], reward: float) -> None:
        """反向传播更新 (可重写)
        
        默认: 累加 visits 和 value。
        变体: Reflexion Feedback (回传文本) 等。
        """
        current = node
        while current is not None:
            current.visits += 1
            current.value += reward
            current = current.parent
    
    # =====================================================================
    # 调试与可视化
    # =====================================================================
    
    def get_stats(self) -> Dict[str, Any]:
        """获取搜索统计信息"""
        if self.root is None:
            return {"status": "not_initialized"}
        
        def count_nodes(node: Node) -> int:
            return 1 + sum(count_nodes(c) for c in node.children)
        
        def max_depth(node: Node, depth: int = 0) -> int:
            if not node.children:
                return depth
            return max(max_depth(c, depth + 1) for c in node.children)
        
        return {
            "total_nodes": count_nodes(self.root),
            "max_depth": max_depth(self.root),
            "root_visits": self.root.visits,
            "root_children": len(self.root.children),
            "best_child_visits": max((c.visits for c in self.root.children), default=0),
        }
    
    def print_tree(self, max_depth: int = 3) -> None:
        """打印搜索树 (调试用)"""
        if self.root is None:
            print("Tree not initialized")
            return
        
        def _print(node: Node, depth: int, prefix: str = "") -> None:
            if depth > max_depth:
                return
            action_str = f"[{node.action}]" if node.action else "[ROOT]"
            print(f"{prefix}{action_str} visits={node.visits}, q={node.q_value:.3f}")
            for i, child in enumerate(node.children):
                is_last = i == len(node.children) - 1
                _print(child, depth + 1, prefix + ("    " if is_last else "│   "))
        
        _print(self.root, 0)


class VanillaMCTS(MCTS[State, Action]):
    """朴素 MCTS 实现 (标准 UCT)
    
    这是最基础的 MCTS 变体，使用:
    - Selection: UCT 公式
    - Expansion: 随机选择未探索动作
    - Evaluation: Random Rollout
    - Backpropagation: 标准均值更新
    """
    pass  # 继承基类的所有默认实现
