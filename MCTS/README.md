# MCTS Module: Monte Carlo Tree Search for LLM Agents

本模块提供一个可扩展的 **MCTS (Monte Carlo Tree Search)** 框架，专为 LLM Agent 场景优化。

## 核心架构

```
MCTS/
├── mcts.py             # [核心] 基类 + VanillaMCTS
├── variants/           # [变体] 不同阶段的优化实现
│   ├── selection.py    # PUCT, EntropyUCT
│   ├── expansion.py    # ProgressiveWidening, ConfidenceAware
│   ├── evaluation.py   # LLMScoring, ProcessReward
│   └── backprop.py     # Reflexion, FullTraceReflexion
├── MCTS_Review.md      # 公式与论文参考
└── README.md           # 本文档
```

---

## 快速开始

### 1. 基础使用 (Vanilla MCTS)

```python
from MCTS.mcts import VanillaMCTS, Environment

# 实现你的环境
class MyEnv(Environment):
    def get_actions(self, state): return [...]
    def step(self, state, action): return new_state
    def is_terminal(self, state): return True/False
    def get_reward(self, state): return 1.0

env = MyEnv()
mcts = VanillaMCTS(env, c_param=1.414)
best_action = mcts.search(initial_state, num_simulations=100)
```

### 2. 使用变体 (Variants)

```python
from MCTS.variants import PUCTMCTS, ProgressiveWideningMCTS, ReflexionMCTS

# PUCT (AlphaZero 风格)
mcts = PUCTMCTS(env, c_puct=2.0)

# 渐进式扩展 (节省 Token)
mcts = ProgressiveWideningMCTS(
    env, 
    widening_constant=1.0, 
    widening_alpha=0.5
)

# 带反思的反向传播
mcts = ReflexionMCTS(
    env, 
    failure_threshold=0.3,
    max_feedback_per_node=3
)
```

---

## 变体一览

| 阶段 | 变体 | 描述 | 适用场景 |
|------|------|------|----------|
| **Selection** | `PUCTMCTS` | AlphaZero PUCT 公式 | 有先验信息 (LLM Logprobs) |
| | `EntropyUCTMCTS` | 动态探索 (熵奖励) | LLM 推理，自适应不确定性 |
| **Expansion** | `ProgressiveWideningMCTS` | 渐进式宽度 | 算力充足，完备搜索 |
| | `ConfidenceAwareMCTS` | 置信度感知 | 预算有限，省 Token |
| **Evaluation** | `LLMScoringMCTS` | LLM 自我打分 | Dense Reward |
| | `ProcessRewardMCTS` | 过程奖励 (PRM) | 数学/代码推理 |
| **Backprop** | `ReflexionMCTS` | 错误摘要回传 | 避免重蹈覆辙 |
| | `FullTraceReflexionMCTS` | 完整轨迹反思 | 短逻辑链 |

---

## 预设组合

### "Economical Scout" (经济型)
```python
from MCTS.variants import EntropyUCTMCTS, ConfidenceAwareMCTS

class EconomicalMCTS(EntropyUCTMCTS, ConfidenceAwareMCTS):
    pass  # 多重继承组合
```

### "Deep Thinker" (深度型)
```python
from MCTS.variants import PUCTMCTS, ProgressiveWideningMCTS, ReflexionMCTS

class DeepThinkerMCTS(PUCTMCTS, ProgressiveWideningMCTS, ReflexionMCTS):
    pass
```

---

## 扩展指南

### 自定义 Selection

```python
from MCTS.mcts import MCTS, Node

class MySelectionMCTS(MCTS):
    def select_action(self, node: Node) -> Node:
        # 实现你的选择逻辑
        return max(node.children, key=lambda c: your_score(c))
```

### 集成 LLM 评估

```python
from MCTS.variants.evaluation import LLMScoringMCTS

class MyLLMEvaluator:
    def score(self, state, rubric: str) -> float:
        # 调用你的 LLM API
        return call_gpt4(f"Score this: {state}")

mcts = LLMScoringMCTS(env, evaluator=MyLLMEvaluator())
```

---

## 参考

详细的公式推导和论文引用见 [MCTS_Review.md](./MCTS_Review.md)。
