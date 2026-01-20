"""
MCTS Configuration - 可调参数配置

本文件集中管理 MCTS 所有变体的可调参数，方便统一调试和实验。
每个参数都附有推荐值范围和调参建议。
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MCTSConfig:
    """MCTS 基础配置"""
    
    # =========================================================================
    # 核心参数 (Core)
    # =========================================================================
    
    num_simulations: int = 100
    """模拟次数
    - 推荐值: 50 ~ 500
    - 说明: 越大搜索越完备，但耗时越长
    - 调参: 简单任务 50-100，复杂任务 200-500
    """
    
    max_depth: Optional[int] = 20
    """最大搜索深度
    - 推荐值: 10 ~ 50 (None = 无限制)
    - 说明: 防止搜索树过深导致资源浪费
    - 调参: 根据任务的最长解答步数设置
    """
    
    # =========================================================================
    # Selection 参数 (选择阶段)
    # =========================================================================
    
    c_param: float = 1.414
    """UCT 探索常数 (标准 UCT)
    - 推荐值: 1.0 ~ 2.0
    - 说明: sqrt(2) ≈ 1.414 是理论最优值
    - 调参: 太大过度探索，太小容易陷入局部最优
    """
    
    c_puct: float = 2.0
    """PUCT 探索常数 (AlphaZero 风格)
    - 推荐值: 1.0 ~ 4.0
    - 说明: 控制先验 P(s,a) 对探索的影响权重
    - 调参: 先验质量高时可调大，先验噪声大时调小
    """
    
    lambda_entropy: float = 0.5
    """熵奖励系数 (Entropy UCT)
    - 推荐值: 0.1 ~ 1.0
    - 说明: 当模型困惑 (高熵) 时，增加探索力度
    - 调参: LLM 推理场景建议 0.3-0.7
    """
    
    # =========================================================================
    # Expansion 参数 (扩展阶段)
    # =========================================================================
    
    widening_constant: float = 1.0
    """渐进式宽度常数 C (Progressive Widening)
    - 推荐值: 0.5 ~ 2.0
    - 说明: 控制子节点数量的基础值
    - 公式: |children| <= C * N(s)^α
    """
    
    widening_alpha: float = 0.5
    """渐进式宽度指数 α (Progressive Widening)
    - 推荐值: 0.3 ~ 0.7
    - 说明: 控制子节点增长速度
    - 调参: 越大子节点增长越快，Token 消耗越多
    """
    
    max_children: int = 5
    """最大子节点数 (Constrained / Confidence-aware)
    - 推荐值: 3 ~ 10
    - 说明: 硬性限制每个节点的最大子节点数
    - 调参: Token 预算紧张时设小，充裕时可放大
    """
    
    satisfaction_threshold: float = 0.8
    """满意度阈值 (Confidence-aware Expansion)
    - 推荐值: 0.6 ~ 0.9
    - 说明: 当最佳子节点分数高于此值时，不再扩展
    - 调参: 越高要求越严格，扩展越多
    """
    
    # =========================================================================
    # Evaluation 参数 (评估阶段)
    # =========================================================================
    
    scoring_rubric: str = "Is this step logically sound and helpful?"
    """LLM 打分提示词 (LLM Scoring)
    - 说明: 用于 Prompt LLM 对步骤进行评分
    - 调参: 根据任务类型定制提示词
    """
    
    use_log_sum: bool = True
    """使用对数和 (Process Reward Model)
    - 推荐值: True
    - 说明: V = exp(sum(log(p_i))) 避免乘积下溢
    - 调参: 长路径必须用 True，短路径可用 False
    """
    
    # =========================================================================
    # Backpropagation 参数 (反向传播阶段)
    # =========================================================================
    
    failure_threshold: float = 0.3
    """失败阈值 (Reflexion Backprop)
    - 推荐值: 0.2 ~ 0.5
    - 说明: 低于此分数视为失败，触发反馈回传
    - 调参: 越高越敏感，反馈越多
    """
    
    max_feedback_per_node: int = 3
    """每个节点最大反馈数 (Reflexion Backprop)
    - 推荐值: 2 ~ 5
    - 说明: 限制反馈缓冲区大小，防止 Context 爆炸
    - 调参: Context Window 小时设 2，大时可设 5
    """


# =============================================================================
# 预设配置 (Presets)
# =============================================================================

@dataclass
class EconomicalConfig(MCTSConfig):
    """经济型配置 (Economical Scout)
    
    目标: 极致性价比，适合在线服务
    特点: Token 消耗低，响应快
    """
    num_simulations: int = 50
    max_depth: int = 15
    c_puct: float = 1.5
    lambda_entropy: float = 0.3
    widening_alpha: float = 0.3  # 慢速扩展
    max_children: int = 3
    satisfaction_threshold: float = 0.7
    max_feedback_per_node: int = 2


@dataclass
class DeepThinkerConfig(MCTSConfig):
    """深度思考配置 (Deep Thinker)
    
    目标: 追求 SOTA 解决率，适合离线任务
    特点: 搜索完备，不计成本
    """
    num_simulations: int = 300
    max_depth: int = 50
    c_puct: float = 3.0
    lambda_entropy: float = 0.7
    widening_constant: float = 1.5
    widening_alpha: float = 0.6  # 快速扩展
    max_children: int = 8
    satisfaction_threshold: float = 0.9
    failure_threshold: float = 0.4
    max_feedback_per_node: int = 5


@dataclass
class BalancedConfig(MCTSConfig):
    """平衡配置 (Balanced)
    
    目标: 性能与成本的平衡
    特点: 适中的搜索深度和宽度
    """
    num_simulations: int = 100
    max_depth: int = 25
    c_puct: float = 2.0
    lambda_entropy: float = 0.5
    widening_constant: float = 1.0
    widening_alpha: float = 0.5
    max_children: int = 5
    satisfaction_threshold: float = 0.8
    failure_threshold: float = 0.3
    max_feedback_per_node: int = 3


# =============================================================================
# 使用示例
# =============================================================================

if __name__ == "__main__":
    # 打印所有预设配置
    print("=" * 60)
    print("MCTS Configuration Presets")
    print("=" * 60)
    
    configs = {
        "Economical": EconomicalConfig(),
        "Balanced": BalancedConfig(),
        "DeepThinker": DeepThinkerConfig(),
    }
    
    for name, config in configs.items():
        print(f"\n{name}:")
        print("-" * 40)
        for field_name in config.__dataclass_fields__:
            value = getattr(config, field_name)
            print(f"  {field_name}: {value}")
