"""
SPIRAL MCTS - 基于 SPIRAL 框架的树搜索

继承自 mcts.MCTS 基类，实现 SPIRAL 架构：
    - Planner (π_planner): 生成候选动作
    - Simulator (W_sim): 预测观察 + 基础奖励
    - Critic (C_critic): 评估动作的策略价值 ρ_ref (使用 LLM)

奖励计算:
    R_t = α * R_base(a_t) + (1 - α) * ρ_ref
"""

from abc import ABC, abstractmethod
from typing import Any, Generic, List, Optional, Protocol, Tuple, TypeVar

from .mcts import MCTS, Node, Environment

State = TypeVar("State")
Action = TypeVar("Action")


# =============================================================================
# Critic 协议 (SPIRAL 特有)
# =============================================================================

class Critic(ABC, Protocol[State, Action]):
    """Critic: 评估动作的策略价值 ρ_ref
    
    evaluate() 返回 (score, reasoning) 元组，reasoning 会被存入节点 metadata。
    """
    
    @abstractmethod
    def evaluate(
        self, 
        state: State, 
        action: Action, 
        next_state: State
    ) -> Tuple[float, str]:
        """评估动作，返回 (分数, 思考过程)
        
        Args:
            state: 当前状态
            action: 执行的动作
            next_state: 动作后的新状态
            
        Returns:
            (score, reasoning) 元组
            - score: [0, 1] 的策略分数
            - reasoning: 评估的思考过程
        """
        ...


# =============================================================================
# LLM Critic 实现
# =============================================================================

class LLMCritic(Generic[State, Action]):
    """使用 LLM 进行评估的 Critic
    
    调用 LLM.evaluate_action() 获取分数和思考过程。
    
    Example:
        >>> from LLM import LLM
        >>> llm = LLM()
        >>> critic = LLMCritic(llm)
        >>> score, reasoning = critic.evaluate(state, action, next_state)
    """
    
    def __init__(self, llm: Any):
        """
        Args:
            llm: LLM 实例 (来自 LLM 模块)
        """
        self._llm = llm
    
    def evaluate(
        self, 
        state: State, 
        action: Action, 
        next_state: State
    ) -> Tuple[float, str]:
        """调用 LLM 评估动作
        
        将状态和动作转换为字符串后调用 LLM。
        """
        # 将状态/动作转换为字符串描述
        state_str = self._to_string(state)
        action_str = self._to_string(action)
        next_state_str = self._to_string(next_state)
        
        return self._llm.evaluate_action(
            current_state=state_str,
            action=action_str,
            next_state=next_state_str,
        )
    
    def _to_string(self, obj: Any) -> str:
        """将对象转换为字符串描述"""
        if hasattr(obj, 'to_prompt') and callable(obj.to_prompt):
            return obj.to_prompt()
        if hasattr(obj, '__str__'):
            return str(obj)
        return repr(obj)



# =============================================================================
# SPIRAL 环境适配器
# =============================================================================

class SPIRALEnvironment(ABC, Environment[State, Action]):
    """SPIRAL 环境：整合 Planner + Simulator 接口
    
    这是一个适配器，将 SPIRAL 的 Planner/Simulator 接口转换为 MCTS 基类需要的 Environment 接口。
    """
    
    @abstractmethod
    def propose_actions(self, state: State) -> List[Action]:
        """Planner: 生成候选动作列表"""
        ...
    
    @abstractmethod
    def simulate(self, state: State, action: Action) -> State:
        """Simulator: 预测下一状态"""
        ...
    
    @abstractmethod
    def is_terminal(self, state: State) -> bool:
        """Simulator: 检查终止状态"""
        ...
    
    @abstractmethod
    def get_base_reward(self, state: State) -> float:
        """Simulator: 获取基础奖励 R_base"""
        ...
    
    # Environment 接口适配
    def get_actions(self, state: State) -> List[Action]:
        return self.propose_actions(state)
    
    def step(self, state: State, action: Action) -> State:
        return self.simulate(state, action)
    
    def get_reward(self, state: State) -> float:
        return self.get_base_reward(state)


# =============================================================================
# SPIRAL MCTS 实现
# =============================================================================

class SPIRALMCTS(MCTS[State, Action]):
    """SPIRAL MCTS: 使用 UCT 选择和组合奖励回传
    
    特点：
        - Selection: 标准 UCT
        - Expansion: 使用 Planner 提议动作
        - Evaluation: 组合奖励 R = α*R_base + (1-α)*ρ_ref
        - Backpropagation: 标准回传 + 思考过程存入节点
        
    Algorithm 1 (SPIRAL):
        1. Selection: 从根节点选择 UCT 最高的子节点直到叶子
        2. Expansion: 调用 Planner 生成动作，Simulator 预测下一状态
        3. Evaluation: Critic 评估新节点，返回 (score, reasoning)
        4. Backpropagation: 计算 R = α*R_base + (1-α)*ρ_ref 并回传
    """
    
    def __init__(
        self,
        env: SPIRALEnvironment[State, Action],
        critic: Critic[State, Action],
        c_param: float = 1.414,
        reward_alpha: float = 0.5,
        max_depth: Optional[int] = None,
    ):
        """
        Args:
            env: SPIRAL 环境 (整合 Planner + Simulator)
            critic: 动作评估器 (推荐使用 LLMCritic)
            c_param: UCT 探索常数
            reward_alpha: 奖励混合系数 (R = α*R_base + (1-α)*ρ_ref)
            max_depth: 最大搜索深度
        """
        super().__init__(env, c_param=c_param, max_depth=max_depth)
        self.critic = critic
        self.reward_alpha = reward_alpha
    
    def evaluate(self, node: Node[State, Action]) -> float:
        """计算组合奖励: R = α * R_base + (1 - α) * ρ_ref
        
        同时将 Critic 的思考过程存入 node.metadata['critic_reasoning']
        """
        # 获取基础奖励
        r_base = self.env.get_reward(node.state)
        
        if node.is_terminal:
            node.metadata["critic_reasoning"] = "终止节点，使用基础奖励"
            return r_base
        
        # Critic 提供策略评估 (返回 score 和 reasoning)
        if node.action is not None and node.parent is not None:
            rho_ref, reasoning = self.critic.evaluate(
                node.parent.state,
                node.action,
                node.state
            )
            # 存储思考过程到节点 metadata
            node.metadata["critic_reasoning"] = reasoning
            node.metadata["critic_score"] = rho_ref
        else:
            rho_ref = r_base
            node.metadata["critic_reasoning"] = "根节点，使用基础奖励"
        
        # 组合奖励: R = α * R_base + (1 - α) * ρ_ref
        combined_reward = self.reward_alpha * r_base + (1 - self.reward_alpha) * rho_rxef
        node.metadata["combined_reward"] = combined_reward
        
        return combined_reward
