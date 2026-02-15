# SPIRAL MCTS Implementation Map

## 理论与实现的映射

我们当前使用的 **System 2 MCTS** 实际上是对 **SPIRAL Framework** 的一种具体化实现。下表展示了理论角色与代码模块的对应关系。

### 角色映射

| SPIRAL 角色 | 功能描述 | 代码实现 | 对应 Prompt |
| :--- | :--- | :--- | :--- |
| **Planner** (规划器) | 给定当前状态，提议下一步可能的动作 | `MCTSAdapter.get_next_step()` | `MCTS_NEXT_STEP`<br>`MCTS_NEXT_STEP_WITH_REFLECTION` |
| **Simulator** (模拟器) | 预测执行动作后的新状态 (世界模型) | `MCTS.random_policy()` (Rollout)<br>*注: 我们直接用 LLM 预测"下一步会发生什么"，而非物理仿真* | `MCTS_NEXT_STEP` (在 Rollout 中预测后继步骤) |
| **Critic** (评估器) | 评估当前状态/动作对达成目标的价值 | `MCTSAdapter.get_step_value()` | `MCTS_VALUE_EVALUATION` |

### 增强机制 (Beyond SPIRAL)

我们在标准 SPIRAL 的基础上增加了 **Reflexion (反思)** 机制：

*   **Prompt**: `MCTS_REFLECTION`
*   **机制**: 在 Planner 提议动作之前，先让 Critic (以反思模式) 对当前历史进行点评。如果发现方向错误，会**强制注入**批评意见给 Planner。
*   **代码**: `MCTS.expand()` 中的 `if not node.reflection: ...` 逻辑。

---

## 完整工作流 (Workflow)

1.  **Request**: 用户输入 IoT 诊断任务。
2.  **Planner (Expansion)**:
    *   先检查 `Critic` 意见 (`MCTS_REFLECTION`)。
    *   LLM 生成 $K$ 个候选步骤 (`MCTS_NEXT_STEP`).
3.  **Critic (Evaluation)**:
    *   LLM 对每个步骤打分 (`MCTS_VALUE_EVALUATION`).
    *   分数 $V \in [0, 1]$ 作为节点的初始价值。
4.  **Simulator (Rollout)**:
    *   对于最有潜力的节点，LLM 继续推演后续 5 步 (`MCTS_NEXT_STEP` in loop).
    *   记录推演中出现的最高价值作为该路径的最终评分。
5.  **Output**: 搜索结束，输出价值最高的轨迹。
