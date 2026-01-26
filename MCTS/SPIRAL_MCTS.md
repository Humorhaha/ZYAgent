# SPIRAL MCTS

基于 SPIRAL 论文的树搜索实现。

## 核心架构

```
                    ┌─────────────────────────────────────────┐
                    │              SPIRAL Agent               │
                    │  (Planner + Simulator + Critic 三角色)   │
                    └─────────────────────────────────────────┘
                                        │
        ┌───────────────────────────────┼───────────────────────────────┐
        ▼                               ▼                               ▼
  ┌──────────┐                   ┌──────────┐                   ┌──────────┐
  │ Planner  │                   │Simulator │                   │  Critic  │
  │ π_planner│                   │  W_sim   │                   │ C_critic │
  └──────────┘                   └──────────┘                   └──────────┘
       │                               │                               │
       │ a_t ~ π(s_t)                  │ o_{t+1} = W(s_t, a_t)         │ ρ_ref = C(s_t, a_t)
       │ 提议动作                       │ 预测观察                       │ 策略评分
       ▼                               ▼                               ▼
  ┌────────────────────────────────────────────────────────────────────────┐
  │                         组合奖励 (Backpropagation)                      │
  │               R_t = α · R_base(a_t) + (1 - α) · ρ_ref                  │
  └────────────────────────────────────────────────────────────────────────┘
```

## 搜索流程

```
                             ┌───┐
                             │ s₀│ Initial Request
                             └─┬─┘
                               │
            ┌──────────────────┼──────────────────┐
            ▼                  ▼                  ▼
          ┌───┐              ┌───┐              ┌───┐
          │ s₁│              │ s₂│◄─────────────│ s₃│
          └───┘              └─┬─┘              └───┘
                               │
                    ┌──────────┴──────────┐
                    ▼                     ▼
                 ┌─────┐               ┌─────┐
                 │s₃,₁ │               │s₃,₂ │◄── Critic: ρ_ref
                 └─────┘               └──┬──┘
                                          │
                                          ▼
                                       ┌─────┐
                                       │ s₄  │ Terminal
                                       └─────┘
```

**四阶段循环:**

| 阶段 | 操作 | 公式 |
|------|------|------|
| **Selection** | UCT 选择到叶子 | `a = argmax[Q + c√(ln N_p / N)]` |
| **Expansion** | Planner 提议动作 | `a_t ~ π_planner(s_t)` |
| **Simulation** | Simulator 预测 + Critic 评分 | `o_t = W_sim(s, a)`, `ρ = C_critic(s, a)` |
| **Backprop** | 组合奖励回传 | `R = α·R_base + (1-α)·ρ_ref` |

## 接口设计

### SPIRALEnvironment

整合 Planner + Simulator 的抽象基类：

```python
class SPIRALEnvironment(ABC):
    # Planner 接口
    def propose_actions(self, state) -> List[Action]: ...
    
    # Simulator 接口
    def simulate(self, state, action) -> State: ...
    def is_terminal(self, state) -> bool: ...
    def get_base_reward(self, state) -> float: ...
```

### Critic

评估动作的策略价值：

```python
class Critic(Protocol):
    def evaluate(self, state, action, next_state) -> float:
        """返回 ρ_ref ∈ [0, 1]"""
```

### SPIRALMCTS

主搜索类：

```python
mcts = SPIRALMCTS(
    env,                    # SPIRALEnvironment 实现
    critic,                 # Critic 实现
    c_param=1.414,          # UCT 探索常数
    reward_alpha=0.5,       # R = α·R_base + (1-α)·ρ_ref
    max_depth=None,         # 最大深度
)
action = mcts.search(state, num_simulations=100)
```

## 奖励计算

```
R_t = α · R_base(a_t) + (1 - α) · ρ_ref
      ━━━━━━━━━━━━━━━   ━━━━━━━━━━━━━━━
      Simulator 奖励      Critic 评分
      (结果启发式)        (策略评估)
```

- `α = 1.0`: 纯结果奖励 (忽略 Critic)
- `α = 0.0`: 纯策略评分 (忽略 R_base)
- `α = 0.5`: 平衡组合 (推荐)

## 扩展点

| 组件 | 扩展方式 | 示例 |
|------|----------|------|
| Planner | 实现 `propose_actions` | LLM 动作采样 |
| Simulator | 实现 `simulate` | 世界模型预测 |
| Critic | 实现 `evaluate` | LLM 自我评估 / PRM |
