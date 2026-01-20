"""
MCTS Variants - 变体模块

本模块提供 MCTS 的多种工程化变体实现，可独立使用或组合使用。

Selection Variants:
    - PUCT: AlphaZero 风格的先验引导 UCT
    - EntropyUCT: 基于熵的动态探索

Expansion Variants:
    - ProgressiveWidening: 渐进式宽度扩展
    - ConfidenceAwareExpansion: 置信度感知扩展

Evaluation Variants:
    - LLMScoring: LLM 自我打分
    - ProcessRewardModel: 过程奖励模型 (PRM)

Backpropagation Variants:
    - ReflexionBackprop: 带文本反馈的反向传播
"""

from .selection import PUCTMCTS, EntropyUCTMCTS
from .expansion import ProgressiveWideningMCTS, ConfidenceAwareMCTS
from .evaluation import LLMScoringMCTS, ProcessRewardMCTS
from .backprop import ReflexionMCTS

__all__ = [
    # Selection Variants
    "PUCTMCTS",
    "EntropyUCTMCTS",
    # Expansion Variants
    "ProgressiveWideningMCTS",
    "ConfidenceAwareMCTS",
    # Evaluation Variants
    "LLMScoringMCTS",
    "ProcessRewardMCTS",
    # Backpropagation Variants
    "ReflexionMCTS",
]
