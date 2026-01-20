# SPIRAL 框架实现分析

基于对 [scripts/taskbench_spiral_method_final.py](file:///Users/richw/SPIRAL/scripts/taskbench_spiral_method_final.py) 代码的分析，以下是 SPIRAL 框架在代码层面的具体实现方式。

## 核心组件

SPIRAL 智能体被实现为一个 **蒙特卡洛树搜索 (Monte Carlo Tree Search, MCTS)** 算法，用于探索可能的工具调用序列空间。

### 1. 状态表示 ([Node](file:///Users/richw/SPIRAL/scripts/taskbench_spiral_method_final.py#122-148))
- **代码对应**: `class Node`
- **核心概念**: 每一个节点代表规划过程中的一个状态 $s_t$。
- **实现方式**: 状态由其 `chain` 属性定义，这是一个字符串列表，代表了当前的对话历史（用户请求 $\rightarrow$ 工具调用 $\rightarrow$ 观察结果 $\rightarrow$ ...）。

### 2. 规划器 ($\pi_{planner}$)
- **核心概念**: 根据当前状态提出下一个动作 $a_t$（即工具调用）。
- **实现方式**:
    - 代码通过提示（Prompting）一个 LLM 来生成下一步操作。
    - **提示词**: 包含当前规划（`current_node.chain`）、工具定义，以及严格的规则“只回复下一行代码”。
    - **扩展 (Expansion)**: 在 MCTS 的扩展阶段，调用“规划器”来创建一个新的子节点，该节点包含提议的 [api_call](file:///Users/richw/SPIRAL/scripts/taskbench_smriv_mcts_revised_final.py#66-92) 或 `finish` 动作。

### 3. 模拟器 ($\mathcal{W}_{sim}$)
- **核心概念**: 预测在状态 $s_t$ 下执行动作 $a_t$ 后的结果（观察）$o_{t+1}$。
- **代码对应**: `class SimulatedToolExecutor`
- **实现方式**:
    - SPIRAL **并不在搜索过程中真正执行工具**（这可能危险或昂贵），而是使用另一个 LLM 来**模拟**工具的执行。
    - **提示词**: “你是一个模拟的 API 工具... 请提供一个逼真的、单行的观察结果...”。
    - 生成的观察结果会紧跟在工具调用之后被添加到节点的 `chain` 中。

### 4. 评价器 ($\mathcal{C}_{critic}$)
- **核心概念**: 评估当前状态/规划的质量。
- **实现方式**:
    - 在此代码实现中，“评价器”被简化为一个**基于规则的启发式方法**，而非习得的模型。
    - **奖励机制 ($R_t$)**:
        - **有效的工具调用**: +0.1（给予微小的正向奖励以鼓励推进）。
        - **无效的工具调用**（语法/格式错误）: -1.0（惩罚）。
        - **完成任务 (`finish`)**: +1.0（对完成规划给予高额奖励）。
    - **反向传播**: 这些奖励会向上传播，用于更新父节点的 $Q$ 值（`value_sum / visits`）。

## 搜索算法 (MCTS)

该框架将上述组件通过标准的 MCTS 循环结合在一起：

1.  **选择 (Selection)**: 使用 **UCT (Upper Confidence Bound applied to Trees)** 公式从根节点开始遍历树，以平衡“利用”（Exploitation）和“探索”（Exploration）。
    ```python
    def uct_score(self):
        exploitation = self.value_sum / self.visits
        # 探索项，鼓励访问较少探索的节点
        exploration = exploration_constant * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration
    ```
2.  **扩展 (Expansion)**: **规划器** (LLM) 为选中的节点生成一个新的工具调用。
3.  **模拟 (Simulation)**:
    - 代码首先验证工具调用的语法。
    - 如果有效，**模拟器** (LLM) 生成一个观察结果。
    - 创建一个包含 `[...历史, 工具调用, 观察结果]` 的新节点。
4.  **反向传播 (Backpropagation)**: **评价器**的评分 (0.1, -1.0 或 1.0) 被累加到路径上所有节点的价值中。

## 总结映射表

| SPIRAL 组件 | 代码实现 |在这个系统中的角色 |
| :--- | :--- | :--- |
| **Agent (智能体)** | [process_problem_with_spiral](file:///Users/richw/SPIRAL/scripts/taskbench_spiral_method_final.py#152-260) | 编排整个 MCTS 循环。 |
| **Planner (规划器)** | `client.send(...)` (带提示词) | 生成候选的 [api_call](file:///Users/richw/SPIRAL/scripts/taskbench_smriv_mcts_revised_final.py#66-92)。 |
| **Simulator (模拟器)** | [SimulatedToolExecutor](file:///Users/richw/SPIRAL/scripts/taskbench_spiral_method_final.py#98-118) | 为规划过程“幻觉”出工具的返回结果。 |
| **Critic (评价器)** | 硬编码的奖励规则 (0.1 / 1.0 / -1.0) | 评估步骤的有效性及任务是否完成。 |
