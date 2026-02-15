# System 2 MCTS Architecture (Current Implementation)

Current Status: **Activated (System 2 MCTSStrategy)**
Implementation: `system2/mcts.py` (Adapter) + `MCTS/base.py` (Core)

本系统采用一种 **基于 LLM 反思 (Reflection) 和 价值评估 (Value Model)** 的增强型 Monte Carlo Tree Search。它专为解决工业 IoT 诊断等复杂多步推理任务而设计。

---

## 核心架构 (Architecture)

```mermaid
graph TD
    User[用户任务] --> Strategy[MCTSStrategy]
    Strategy --> Adapter[MCTSAdapter]
    Adapter --> Core[MCTS Core]
    
    subgraph Core [MCTS 循环]
        Select[Selection: UCT 选择]
        Expand[Expansion: LLM 生成 & 反思]
        Sim[Simulation: 快速推演]
        Back[Backprop: 价值回传]
        
        Select --> Expand --> Sim --> Back
    end
    
    subgraph LLM [LLM 服务 (Qwen/OpenAI)]
        P_Next[Prompt: MCTS_NEXT_STEP]
        P_Ref[Prompt: MCTS_REFLECTION]
        P_Val[Prompt: MCTS_VALUE_EVALUATION]
    end
    
    Expand --> P_Ref
    Expand --> P_Next
    Expand --> P_Val
    Sim --> P_Next
```

---

## 详细流程 (Process Flow)

### 1. Selection (选择阶段)
使用经典的 **UCT (Upper Confidence Bound for Trees)** 公式选择最有潜力的节点进行扩展。
- **公式**: $UCT = V + C \cdot \sqrt{\frac{2 \ln N_{parent}}{N_{child}}}$
- **目标**: 平衡 "利用" (Exploitation, 选择高分节点) 和 "探索" (Exploration, 选择少访问节点)。

### 2. Expansion (扩展阶段) - *关键创新点*
这是与传统 MCTS 最大的不同点，引入了 **反思 (Reflection)** 机制。

1.  **Reflection Check (纠偏)**:
    - 调用 `MCTS_REFLECTION` Prompt。
    - 询问: "当前路径是否有效？是否已解决？" (Output: `<end>` 或 批评意见)。
    - *作用*: 防止盲目扩展错误的路径。

2.  **Generative Expansion (生成)**:
    - 调用 `MCTS_NEXT_STEP` (或 `_WITH_REFLECTION`) Prompt。
    - 基于历史和反思意见，生成 $K$ 个候选动作 (Branch Factor)。
    - *作用*: 利用 LLM 的知识库生成具体的排查步骤。

3.  **Initial Evaluation (初筛)**:
    - 对每个新生成的动作，立即调用 `MCTS_VALUE_EVALUATION`。
    - 评分: 0.0 - 1.0 (0.4=猜测, 0.8=有据, 1.0=确诊)。
    - *作用*: 直接赋予新节点先验价值 (Prior Value)。

### 3. Simulation (模拟阶段)
为了验证新动作的长期效果，进行快速推演 (Rollout)。
- **策略**: Greedy (贪婪) 或 Random (随机)。
- **过程**: 连续调用 LLM 生成后续步骤，直到达到深度限制或解决问题。
- **打分**: 记录推演路径上出现的**最高价值状态**。

### 4. Backpropagation (反向传播)
将模拟阶段得到的最高价值回传给当前节点及其所有父节点。
- **更新方式**: 平均值更新 (Average Update)。
- *结果*: 好的尝试会提高整条路径的权重，引导下一轮 UCT 选择该路径。

---

## Prompt 映射

| MCTS 阶段 | 对应的 Prompt | 作用 |
| :--- | :--- | :--- |
| **Expansion** | `MCTS_REFLECTION` | 检查状态/生成批评 |
| **Expansion** | `MCTS_NEXT_STEP` | 生成候选动作 |
| **Evaluation** | `MCTS_VALUE_EVALUATION` | 节点打分 (Prior) |
| **Rollout** | `MCTS_NEXT_STEP` | 快速推演步骤 |
| **Rollout** | `MCTS_REFLECTION` | 检查推演终止条件 |

---

## 代码对应关系

- **主循环**: `MCTS/base.py` -> `MCTS.search()`
- **Prompt 接口**: `system2/mcts.py` -> `MCTSAdapter`
- **Prompt 定义**: `LLM/prompts.py`