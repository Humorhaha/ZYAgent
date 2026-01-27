"""
Monte Carlo Tree Search (MCTS) - 面向对象实现

基于用户提供的参考代码重构，采用面向对象风格。
核心流程: Selection -> Expansion -> Simulation -> Backpropagation
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from abc import ABC, abstractmethod
import math
import random
import time
import numpy as np


@dataclass
class Node:
    """MCTS 树节点
    
    Attributes:
        new_action: 当前节点新增的动作（步骤文本）
        action: 等同于 new_action（兼容性）
        history_action: 从根到当前节点的完整动作历史（等价于参考代码中的 y）
        reflection: 节点的反思/反馈文本
        summary: 节点摘要
        parent: 父节点引用
        children: 子节点字典 {action: Node}
        visits: 访问次数 (等价于 numVisits)
        value: 节点价值 (等价于 V)
        prior: 先验概率
        is_terminal: 是否为终止节点
        on_final_route: 是否在最终路径上
        is_fully_expanded: 是否已完全展开
        final_ans_flag: 标记是否为最终答案 (0/1)
        visit_sequence: 访问顺序编号
    """
    new_action: str = ''
    action: str = ''
    history_action: str = ''
    reflection: str = ''
    summary: str = ''
    parent: Optional["Node"] = None
    children: Dict[str, "Node"] = field(default_factory=dict)
    visits: int = 0
    value: float = 0.0
    prior: float = 1.0
    is_terminal: bool = False
    on_final_route: bool = False
    is_fully_expanded: bool = False
    final_ans_flag: int = 0
    visit_sequence: int = 0
    
    @property
    def depth(self) -> int:
        """计算节点深度（从根到当前节点的距离）"""
        d = 0
        node = self.parent
        while node is not None:
            d += 1
            node = node.parent
        return d
    
    @property
    def q_value(self) -> float:
        """计算平均价值 Q(s) = value / visits"""
        if self.visits == 0:
            return 0.0
        return self.value / self.visits
    
    @property
    def is_leaf(self) -> bool:
        """是否为叶子节点"""
        return len(self.children) == 0
    
    def add_child(self, new_action: str) -> "Node":
        """添加子节点
        
        Args:
            new_action: 新动作（步骤文本）
            
        Returns:
            新创建的子节点
        """
        child = Node(new_action=new_action)
        child.parent = self
        child.action = new_action
        child.history_action = self.history_action + new_action
        self.children[new_action] = child
        return child
    
    def update_value(self, value: float) -> None:
        """更新节点价值"""
        self.value = value
    
    def update_reflection(self, reflection: str) -> None:
        """更新节点反思"""
        self.reflection = reflection
    
    def trace_route(self) -> None:
        """标记从当前节点到根的路径"""
        cur_node = self
        while cur_node is not None:
            cur_node.on_final_route = True
            cur_node = cur_node.parent
    
    def get_best_value(self) -> Tuple["Node", float]:
        """获取子树中价值最高的节点
        
        Returns:
            (最佳节点, 最佳价值)
        """
        if not self.is_fully_expanded:
            return self, self.value
        max_V = self.value
        max_node = self
        for child in self.children.values():
            sub_node, sub_value = child.get_best_value()
            if sub_value >= max_V:
                max_V = sub_value
                max_node = sub_node
        return max_node, max_V
    
    def __repr__(self) -> str:
        return f"Node(depth={self.depth}, visits={self.visits}, value={self.value:.2f}, children={len(self.children)})"


class MCTSTask(ABC):
    """MCTS 任务抽象基类
    
    定义 MCTS 运行所需的配置参数和必须由子类实现的方法。
    子类需要实现具体的 LLM 调用逻辑。
    """
    
    def __init__(
        self,
        # 常量
        low: float = 0.0,
        end_gate: float = 0.95,
        INF: float = 1e9,
        # 搜索模式
        limit_type: str = 'iteration',
        iteration_limit: int = 100,
        time_limit: int = 60000,
        # 扩展与模拟
        branch: int = 3,
        roll_branch: int = 3,
        roll_forward_steps: int = 5,
        # 策略配置
        use_reflection: str = 'common',
        sample_value: str = 'partial',
        reward_model_type: str = 'vm',
        roll_policy: str = 'greedy',
        # 超参数
        alpha: float = 0.5,
        exploration_constant: float = 1.414,
    ):
        # 常量
        self.low = low
        self.end_gate = end_gate
        self.INF = INF
        
        # 搜索模式配置
        self.limit_type = limit_type
        self.iteration_limit = iteration_limit
        self.time_limit = time_limit
        
        # 扩展与模拟配置
        self.branch = branch
        self.roll_branch = roll_branch
        self.roll_forward_steps = roll_forward_steps
        
        # 策略配置
        self.use_reflection = use_reflection
        self.sample_value = sample_value
        self.reward_model_type = reward_model_type
        self.roll_policy = roll_policy
        
        # 超参数
        self.alpha = alpha
        self.exploration_constant = exploration_constant
        
        # 内部状态
        self.node_count = 0
    
    def update_count(self) -> None:
        """更新节点计数"""
        self.node_count += 1
    
    # =========================================================================
    # 抽象方法 - 子类必须实现
    # =========================================================================
    
    @abstractmethod
    def get_next_step(self, history: str, step_n: int) -> str:
        """生成下一步动作
        
        Args:
            history: 到当前节点的动作历史
            step_n: 当前步骤编号
            
        Returns:
            下一步动作的文本
        """
        ...
    
    @abstractmethod
    def get_next_step_use_reflection(self, history: str, step_n: int, reflection: str) -> str:
        """使用反思生成下一步动作
        
        Args:
            history: 到当前节点的动作历史
            step_n: 当前步骤编号
            reflection: 反思文本
            
        Returns:
            下一步动作的文本
        """
        ...
    
    @abstractmethod
    def get_reflection(self, history: str, step_n: int) -> str:
        """获取完整反思
        
        Args:
            history: 到当前节点的动作历史
            step_n: 当前步骤编号
            
        Returns:
            反思文本，如果已解决返回 '<end>'
        """
        ...
    
    @abstractmethod
    def get_simple_reflection(self, history: str, step_n: int) -> str:
        """获取简化反思
        
        Args:
            history: 到当前节点的动作历史
            step_n: 当前步骤编号
            
        Returns:
            简化反思文本，如果已解决返回 '<end>'
        """
        ...
    
    @abstractmethod
    def get_step_value(self, history: str) -> float:
        """评估步骤价值
        
        Args:
            history: 到当前节点的动作历史
            
        Returns:
            价值分数
        """
        ...


class MCTS:
    """Monte Carlo Tree Search 面向对象实现
    
    核心流程:
        1. Selection: 从根节点选择到叶子节点
        2. Expansion: 展开叶子节点
        3. Simulation: 模拟评估新节点
        4. Backpropagation: 回传价值更新
        
    使用示例:
        ```python
        task = MyMCTSTask(...)  # 继承 MCTSTask 的具体实现
        mcts = MCTS(task)
        root, best_node, finish_info = mcts.search()
        ```
    """
    
    def __init__(self, task: MCTSTask):
        """
        Args:
            task: MCTS 任务配置对象
        """
        self.task = task
        self.root: Optional[Node] = None
    
    # =========================================================================
    # 主搜索接口
    # =========================================================================
    
    def search(self) -> Tuple[Node, Optional[Node], Any]:
        """执行 MCTS 搜索
        
        Returns:
            (root, best_node, finish_info)
            - root: 搜索树根节点
            - best_node: 最佳解答节点（如果找到）
            - finish_info: 完成信息（时间或轮次）
        """
        # 初始化根节点
        self.root = Node(new_action='')
        
        if self.task.limit_type == 'time':
            return self._search_by_time()
        else:
            return self._search_by_iteration()
    
    def _search_by_time(self) -> Tuple[Node, Optional[Node], Any]:
        """按时间限制搜索"""
        time_limit_sec = self.task.time_limit / 1000
        time_start = time.time()
        time_end = time_start + time_limit_sec
        
        while time.time() < time_end:
            elapsed = time.time() - time_start
            print(f'<开始新搜索轮次，目前总时间: {elapsed:.2f}s>\n')
            
            found, node = self.execute_round()
            if found:
                print('已找到解决方案！\n')
                return self.root, node, time.time() - time_start
        
        return self._finalize_search()
    
    def _search_by_iteration(self) -> Tuple[Node, Optional[Node], Any]:
        """按迭代次数搜索"""
        for i in range(self.task.iteration_limit):
            print(f'<开始新搜索轮次，目前已完成轮次数: {i}>\n')
            
            found, node = self.execute_round()
            if found:
                print('已找到解决方案！\n')
                return self.root, node, i + 1
        
        return self._finalize_search()
    
    def _finalize_search(self) -> Tuple[Node, Optional[Node], Any]:
        """搜索结束后的结果处理"""
        if self.task.sample_value == 'full':
            print('采样完成。\n')
            return self.root, None, -1
        
        if self.task.reward_model_type == 'vm':
            best_node, best_V = self.root.get_best_value()
            print(f'在规定时间/轮次内未找到满足要求价值的解答，采用最高价值解答代替。\n'
                  f'Solution: {best_node.history_action}\n')
            return self.root, best_node, -1
        
        print('尚未支持解答选择，采样结束。\n')
        return self.root, None, -1
    
    # =========================================================================
    # 核心 MCTS 流程
    # =========================================================================
    
    def execute_round(self) -> Tuple[bool, Node]:
        """执行一轮 Selection-Expansion-Simulation-Backpropagation
        
        Returns:
            (found, node)
            - found: 是否找到满足条件的解
            - node: 当前处理的节点
        """
        # Selection 阶段
        print('-' * 40)
        print('选择节点阶段\n')
        found, node = self.select_node(self.root)
        if found:
            if self.task.sample_value != 'full':
                return True, node
            else:
                node.reflection = '<end>'
        
        # Expansion 阶段
        print('-' * 40)
        print('扩充阶段\n')
        if node.reflection == '<end>':
            print('跳过此阶段。\n')
        else:
            node = self.expand(node)
        
        # Simulation 阶段 (仅当使用 value model 时)
        if self.task.reward_model_type == 'vm':
            print('-' * 40)
            print('模拟搜索阶段\n')
            if node.reflection == '<end>':
                print('跳过此阶段。\n')
            else:
                roll_node = self.get_best_child(node)
                if self.task.roll_policy == 'greedy':
                    best_V = self.greedy_policy(roll_node)
                else:
                    best_V = self.random_policy(roll_node)
                roll_node.value = roll_node.value * (1 - self.task.alpha) + best_V * self.task.alpha
                roll_node.visits += 1
        
        # Backpropagation 阶段
        print('-' * 40)
        print('反向传播阶段\n')
        self.backpropagate(node)
        
        return False, node
    
    def select_node(self, node: Node) -> Tuple[bool, Node]:
        """Selection 阶段: 沿着树选择直到未完全展开的节点
        
        Args:
            node: 起始节点（通常是根节点）
            
        Returns:
            (is_terminal, selected_node)
        """
        while node.is_fully_expanded:
            node = self.get_best_child(node)
        
        if self.is_terminal(node):
            node.final_ans_flag = 1
            return True, node
        
        return False, node
    
    def expand(self, node: Node) -> Node:
        """Expansion 阶段: 展开节点
        
        Args:
            node: 要展开的节点
            
        Returns:
            展开后的节点
        """
        # 如果没有反思，先获取反思
        if not node.reflection:
            if self.task.use_reflection == 'common':
                reflection = self.task.get_reflection(node.history_action, node.depth + 1)
            else:
                reflection = self.task.get_simple_reflection(node.history_action, node.depth + 1)
            node.update_reflection(reflection)
        
        if node.reflection == '<end>':
            return node
        
        # 获取下一步动作
        actions = self._get_next_steps_expand(node)
        if not actions:
            node.update_reflection('<end>')
            return node
        
        # 添加所有新动作作为子节点
        for action in actions:
            if action not in node.children:
                child = node.add_child(action)
                value = self.task.get_step_value(child.history_action)
                child.update_value(value)
                
                if self.task.sample_value == 'full':
                    if self.task.use_reflection == 'common':
                        child.update_reflection(
                            self.task.get_reflection(child.history_action, child.depth + 1)
                        )
                    else:
                        child.update_reflection(
                            self.task.get_simple_reflection(child.history_action, child.depth + 1)
                        )
                
                child.visit_sequence = self.task.node_count
                self.task.update_count()
        
        node.is_fully_expanded = True
        return node
    
    def _get_next_steps_expand(self, node: Node) -> List[str]:
        """获取扩展阶段的下一步动作列表
        
        Args:
            node: 当前节点
            
        Returns:
            动作列表
        """
        next_steps = []
        reflection = node.reflection
        
        for _ in range(self.task.branch):
            proposal = ''
            retry_count = 3
            
            while not proposal and retry_count > 0:
                if self.task.use_reflection == 'common':
                    proposal = self.task.get_next_step_use_reflection(
                        node.history_action, node.depth + 1, reflection
                    )
                else:
                    proposal = self.task.get_next_step(node.history_action, node.depth + 1)
                retry_count -= 1
            
            if proposal:
                next_steps.append(proposal)
        
        return next_steps
    
    def _get_next_steps_roll(self, history: str, step_n: int) -> List[str]:
        """获取 rollout 阶段的下一步动作列表
        
        Args:
            history: 当前动作历史
            step_n: 当前步骤编号
            
        Returns:
            动作列表
        """
        next_steps = []
        
        for _ in range(self.task.roll_branch):
            proposal = ''
            retry_count = 3
            
            while not proposal and retry_count > 0:
                proposal = self.task.get_next_step(history, step_n)
                retry_count -= 1
            
            if proposal:
                next_steps.append(proposal)
        
        return next_steps
    
    def get_best_child(self, node: Node) -> Node:
        """选择最佳子节点 (UCT 公式)
        
        UCT = V + c * sqrt(2 * ln(N_parent) / N_child)
        
        Args:
            node: 父节点
            
        Returns:
            UCT 值最高的子节点
        """
        best_value = self.task.low
        best_nodes = []
        
        for child in node.children.values():
            if child.visits > 0:
                node_value = child.value + self.task.exploration_constant * math.sqrt(
                    2 * math.log(node.visits) / child.visits
                )
            else:
                node_value = child.value + self.task.INF
            
            if node_value > best_value:
                best_value = node_value
                best_nodes = [child]
            elif node_value == best_value:
                best_nodes.append(child)
        
        return random.choice(best_nodes) if best_nodes else list(node.children.values())[0]
    
    def is_terminal(self, node: Node) -> bool:
        """判断节点是否为终止节点
        
        Args:
            node: 要判断的节点
            
        Returns:
            是否终止
        """
        if self.task.reward_model_type == 'vm':
            return node.value >= self.task.end_gate
        return False
    
    # =========================================================================
    # 策略方法 (Simulation)
    # =========================================================================
    
    def random_policy(self, node: Node) -> float:
        """随机策略: 随机选择动作进行 rollout
        
        Args:
            node: 起始节点
            
        Returns:
            rollout 过程中的最大价值
        """
        max_V = self.task.low
        history = node.history_action
        cur_step = node.depth + 1
        
        # 获取反思
        if self.task.use_reflection == 'common':
            reflection = self.task.get_reflection(history, cur_step)
        else:
            reflection = self.task.get_simple_reflection(history, cur_step)
        node.update_reflection(reflection)
        
        if reflection == '<end>':
            print('This step has been resolved and does not require simulation.\n')
            return node.value
        
        # 前向模拟
        for _ in range(self.task.roll_forward_steps):
            next_steps = self._get_next_steps_roll(history, cur_step)
            if not next_steps:
                break
            
            action = random.choice(next_steps)
            history = history + action
            cur_step += 1
            
            value = self.task.get_step_value(history)
            if value > max_V:
                max_V = value
            
            # 检查是否结束
            if self.task.use_reflection == 'common':
                cur_ref = self.task.get_reflection(history, cur_step)
            else:
                cur_ref = self.task.get_simple_reflection(history, cur_step)
            
            if cur_ref == '<end>':
                break
        
        return max_V
    
    def greedy_policy(self, node: Node) -> float:
        """贪婪策略: 每步选择价值最高的动作
        
        Args:
            node: 起始节点
            
        Returns:
            rollout 过程中的最大价值
        """
        max_V = self.task.low
        history = node.history_action
        cur_step = node.depth + 1
        
        # 获取反思
        if self.task.use_reflection == 'common':
            reflection = self.task.get_reflection(history, cur_step)
        else:
            reflection = self.task.get_simple_reflection(history, cur_step)
        node.update_reflection(reflection)
        
        if reflection == '<end>':
            print('This step has been resolved and does not require simulation.\n')
            return node.value
        
        # 前向模拟
        for _ in range(self.task.roll_forward_steps):
            actions = self._get_next_steps_roll(history, cur_step)
            if not actions:
                break
            
            # 评估所有候选动作
            new_histories = [history + action for action in actions]
            cur_step += 1
            values = [self.task.get_step_value(h) for h in new_histories]
            
            # 选择最佳
            idx = int(np.argmax(values))
            history = new_histories[idx]
            value = values[idx]
            
            if value > max_V:
                max_V = value
            
            # 检查是否结束
            if self.task.use_reflection == 'common':
                cur_ref = self.task.get_reflection(history, cur_step)
            else:
                cur_ref = self.task.get_simple_reflection(history, cur_step)
            
            if cur_ref == '<end>':
                break
        
        return max_V
    
    # =========================================================================
    # 反向传播
    # =========================================================================
    
    def backpropagate(self, node: Node) -> None:
        """反向传播: 更新从当前节点到根的所有节点
        
        Args:
            node: 起始节点
        """
        while node is not None:
            node.visits += 1
            
            if node.is_fully_expanded:
                # 计算子节点的加权平均价值
                child_weighted_values = [
                    child.value * child.visits 
                    for child in node.children.values()
                ]
                total_visits = sum(child.visits for child in node.children.values())
                
                if total_visits > 0:
                    node.value = sum(child_weighted_values) / total_visits
            
            node = node.parent
    
    # =========================================================================
    # 调试与可视化
    # =========================================================================
    
    def get_stats(self) -> Dict[str, Any]:
        """获取搜索统计信息"""
        if self.root is None:
            return {"status": "not_initialized"}
        
        def count_nodes(node: Node) -> int:
            return 1 + sum(count_nodes(c) for c in node.children.values())
        
        def max_tree_depth(node: Node, depth: int = 0) -> int:
            if not node.children:
                return depth
            return max(max_tree_depth(c, depth + 1) for c in node.children.values())
        
        return {
            "total_nodes": count_nodes(self.root),
            "max_depth": max_tree_depth(self.root),
            "root_visits": self.root.visits,
            "root_children": len(self.root.children),
            "best_child_visits": max(
                (c.visits for c in self.root.children.values()), default=0
            ),
        }
    
    def print_tree(self, max_depth: int = 3) -> None:
        """打印搜索树 (调试用)"""
        if self.root is None:
            print("Tree not initialized")
            return
        
        def _print(node: Node, depth: int, prefix: str = "") -> None:
            if depth > max_depth:
                return
            action_str = f"[{node.action[:20]}...]" if len(node.action) > 20 else f"[{node.action}]"
            if not node.action:
                action_str = "[ROOT]"
            print(f"{prefix}{action_str} visits={node.visits}, V={node.value:.3f}")
            
            children_list = list(node.children.values())
            for i, child in enumerate(children_list):
                is_last = i == len(children_list) - 1
                _print(child, depth + 1, prefix + ("    " if is_last else "│   "))
        
        _print(self.root, 0)
