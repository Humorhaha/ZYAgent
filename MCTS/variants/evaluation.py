"""
Evaluation Variants - 评估阶段变体

实现不同的节点价值评估策略，解决奖励稀疏问题。

变体:
    - LLMScoringMCTS: LLM 自我打分 (Dense Reward)
    - ProcessRewardMCTS: 过程奖励模型 (PRM)
"""

from abc import abstractmethod
from typing import Optional, Protocol, TypeVar, List
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcts import MCTS, Node, Environment

State = TypeVar("State")
Action = TypeVar("Action")


class LLMEvaluator(Protocol):
    """LLM 评估器协议"""
    
    def score(self, state: State, rubric: str = "") -> float:
        """对状态进行打分，返回 [0, 1] 之间的分数"""
        ...


class ProcessRewardModel(Protocol):
    """过程奖励模型 (PRM) 协议"""
    
    def score_step(self, state: State, step_index: int) -> float:
        """对单个步骤进行打分，返回该步骤正确的概率"""
        ...


class LLMScoringMCTS(MCTS[State, Action]):
    """LLM 打分评估 (LATS 风格)
    
    公式:
        r = LLM_eval(s, rubric) ∈ [0, 1]
        Q(s, a) = mean(r_i) + w * HasFeedback(s)
    
    特点:
        - Dense Reward: 每个节点都有即时反馈
        - 可引入 Self-Refine 质量奖励
    """
    
    def __init__(
        self,
        env: Environment[State, Action],
        evaluator: Optional[LLMEvaluator] = None,
        c_param: float = 1.414,
        scoring_rubric: str = "Is this reasoning step logically sound and helpful for solving the problem?",
        max_depth: Optional[int] = None,
    ):
        super().__init__(env, c_param=c_param, max_depth=max_depth)
        self.evaluator = evaluator
        self.scoring_rubric = scoring_rubric
    
    def evaluate(self, node: Node[State, Action]) -> float:
        """使用 LLM 进行评估"""
        if node.is_terminal:
            return self.env.get_reward(node.state)
        
        if self.evaluator is None:
            # 降级到基类的随机 Rollout
            return super().evaluate(node)
        
        # 调用 LLM 打分
        score = self.evaluator.score(node.state, self.scoring_rubric)
        
        # 存储打分供后续参考
        node.metadata["llm_score"] = score
        
        return score


class ProcessRewardMCTS(MCTS[State, Action]):
    """过程奖励模型 (Process Reward Model, PRM)
    
    公式 (Math-Shepherd / Let's Verify Step by Step):
        V(s) = Π_{i=1}^{t} P_θ(step_i is correct | s_{0...i-1})
    
    特点:
        - 使用专门的 Verifier 模型
        - 对每个步骤独立打分
        - 比结果奖励 (ORM) 更有效
    """
    
    def __init__(
        self,
        env: Environment[State, Action],
        prm: Optional[ProcessRewardModel] = None,
        c_param: float = 1.414,
        use_log_sum: bool = True,  # 使用对数和而非乘积，避免下溢
        max_depth: Optional[int] = None,
    ):
        super().__init__(env, c_param=c_param, max_depth=max_depth)
        self.prm = prm
        self.use_log_sum = use_log_sum
    
    def evaluate(self, node: Node[State, Action]) -> float:
        """使用 PRM 评估路径价值"""
        if self.prm is None:
            # 降级到基类的随机 Rollout
            return super().evaluate(node)
        
        # 回溯收集路径上所有步骤
        path_states = []
        current = node
        while current is not None:
            path_states.append(current.state)
            current = current.parent
        path_states.reverse()
        
        # 计算路径上每步的正确概率
        step_scores = []
        for i, state in enumerate(path_states):
            score = self.prm.score_step(state, i)
            step_scores.append(score)
        
        # 存储中间结果
        node.metadata["step_scores"] = step_scores
        
        # 计算总价值
        if self.use_log_sum:
            # V = exp(sum(log(p_i))) = Π p_i
            import math
            log_sum = sum(math.log(s + 1e-10) for s in step_scores)
            return math.exp(log_sum)
        else:
            # 直接乘积
            result = 1.0
            for s in step_scores:
                result *= s
            return result


class MockLLMEvaluator:
    """Mock LLM 评估器 (用于测试)"""
    
    def score(self, state, rubric: str = "") -> float:
        import random
        return random.uniform(0.3, 0.9)


class MockPRM:
    """Mock 过程奖励模型 (用于测试)"""
    
    def score_step(self, state, step_index: int) -> float:
        import random
        # 越后面的步骤越不确定
        base = 0.9 - step_index * 0.05
        return max(0.5, base + random.uniform(-0.1, 0.1))
