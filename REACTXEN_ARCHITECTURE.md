# ReactXen 架构详解

ReactXen 是一个增强型的 Agent 设计框架，其核心在于引入了 **双层循环架构 (Dual-Loop Architecture)**，旨在通过反思（Reflection）和评估（Review）机制，显著提升 Agent 在复杂任务中的解决能力和准确性。

## 核心架构设计

架构主要由两层循环构成：
1.  **外层循环 (Outer Loop)**：负责策略优化、结果评估与反思。
2.  **内层循环 (Inner Loop)**：负责具体的推理、工具调用与行动执行。

```mermaid
graph TD
    Start[User Question] --> OuterLoop
    subgraph OuterLoop [ReactReflectAgent (Manager)]
        Reflect[Reflexion / Strategy Adjustment] --> ReactExec
        subgraph InnerLoop [ReactAgent (Worker)]
            ReactExec[ReAct Execution Loop]
            Think[Thought] --> Act[Action]
            Act --> Observe[Observation]
            Observe --> Think
        end
        ReactExec --> Review[ReviewerAgent Evaluation]
        Review -->|Failed / Needs Improvement| Reflect
    end
    Review -->|Success| End[Final Answer]
```

---

## 主要组件解析

### 1. ReactReflectAgent (总指挥)
*   **角色**：系统的主入口与编排者。
*   **职责**：
    *   **管理迭代**：控制最大重试次数 (`num_reflect_iteration`)。
    *   **上下文注入**：将上一轮的失败经验（Reflexion）注入到新的上下文中，防止重蹈覆辙。
    *   **决策制定**：根据 [ReviewerAgent](file:///Users/richw/ReActXen/src/reactxen/agents/reviewer_agent/agent.py#9-205) 的反馈决定是输出最终结果还是进入下一轮修正。

### 2. ReactAgent (执行者)
*   **角色**：具体的任务执行单元。
*   **职责**：
    *   **CoT 推理**：遵循 `Thought` -> `Action` -> `Observation` 的标准 ReAct 范式。
    *   **工具使用**：解析并调用具体的 Tool（如搜索、计算器等）。
    *   **状态维护**：维护当前的 `scratchpad`（思维链记录）。

### 3. ReviewerAgent (阅卷人)
*   **角色**：独立的评估专家。
*   **职责**：
    *   **客观评价**：在 [ReactAgent](file:///Users/richw/ReActXen/src/reactxen/agents/react/agents.py#350-1752) 认为任务完成后，对其产出进行独立审计。
    *   **结构化反馈**：提供包含 `Status` (成功/失败)、`Reasoning` (原因) 和 `Suggestions` (改进建议) 的结构化评价。

---

## 工作流程 (Workflow)

1.  **初始化**：用户提出问题，系统实例化 [ReactReflectAgent](file:///Users/richw/ReActXen/src/reactxen/agents/react/agents.py#1754-2144)。
2.  **尝试执行 (Attempt)**：
    *   [ReactReflectAgent](file:///Users/richw/ReActXen/src/reactxen/agents/react/agents.py#1754-2144) 启动 [ReactAgent](file:///Users/richw/ReActXen/src/reactxen/agents/react/agents.py#350-1752)。
    *   [ReactAgent](file:///Users/richw/ReActXen/src/reactxen/agents/react/agents.py#350-1752) 进行多轮思考和行动，直到得出潜在答案。
3.  **评估 (Evaluation)**：
    *   [ReviewerAgent](file:///Users/richw/ReActXen/src/reactxen/agents/reviewer_agent/agent.py#9-205) 介入，检查 [ReactAgent](file:///Users/richw/ReActXen/src/reactxen/agents/react/agents.py#350-1752) 的过程和答案。
    *   如果评估通过 -> **输出结果**。
    *   如果评估不通过 -> **生成反思**。
4.  **反思 (Reflection)**：
    *   系统基于 Review 的意见生成自我反思 (Reflexion)。
    *   **关键点**：这些反思会被添加到下一次 [ReactAgent](file:///Users/richw/ReActXen/src/reactxen/agents/react/agents.py#350-1752) 启动时的 Prompt 中 ("Before, I made a mistake... Next time I should...")。
5.  **循环**：带着反思重新进入步骤 2，直到成功或达到最大迭代次数。

## 优势

*   **自我纠错**：避免了传统 ReAct Agent 一条路走到黑的问题。
*   **质量保证**：通过引入 Reviewer 角色，保证了输出结果的可靠性。
*   **鲁棒性**：面对复杂长程任务，通过多轮迭代逐步逼近正确答案。

---

## 核心代码实现 (Core Implementation)

以下展示了 ReactXen 架构中关键组件的核心算法逻辑。为了便于阅读，无关代码已省略，重点逻辑附带详细注释。

### 1. ReactReflectAgent：外层反思循环

此类负责“执行-评估-反思”的大循环。

```python
# src/reactxen/agents/react/agents.py

class ReactReflectAgent(ReactAgent):
    def run(self, reset=True, reflect_strategy=ReflexionStrategy.REFLEXION):
        # 1. 初始化循环计数器
        skip_review = False
        
        # 2. 进入外层最大迭代循环 (num_reflect_iteration)
        # 这个循环保证了如果初次尝试失败，有机会进行自我修正
        for i in range(self.num_reflect_iteration):
            
            # --- 反思 (Reflection) 阶段 ---
            # 如果不是第一次迭代（即之前的尝试失败了），则需要进行反思
            if i > 0:
                if not self.is_finished():
                    # Case A: 任务未完成（例如步数耗尽），生成基于过程的反思
                    self.reflect(reflect_strategy)
                else:
                    # Case B: 任务虽然标记完成，但 Review 质量不合格
                    # ReviewerAgent 会返回 status, reasoning, suggestions
                    review = self.reviewagt.evaluate_response(...)
                    
                    if "Not" in review["status"] or "Partially" in review["status"]:
                        # 记录 Review 意见
                        review_str = f"Task Status: {review['status']}..."
                        self.reviews.append(review_str)
                        
                        # 基于 Review 意见生成新的反思 (Reflexion)
                        # 这些反思会更新到 self.reflections_str，并在下一次 ReAct 执行时作为 Prompt 上下文
                        self.reflect(reflect_strategy)
                    else:
                        # 评估通过，任务成功，跳出循环
                        skip_review = True
                        break

            # --- 执行 (Execution) 阶段 ---
            # 调用父类 ReactAgent 的 run 方法执行具体的推理行动
            # 注意：父类 run 方法会读取 self.reflections_str 注入到 System Prompt 中
            ReactAgent.run(self, reset)
```

### 2. ReactAgent：内层 ReAct 循环

此类负责标准的 Thought-Action-Observation 循环。

```python
# src/reactxen/agents/react/agents.py

class ReactAgent:
    def step(self) -> None:
        # 1. 思考 (Think)
        # 调用 LLM 生成下一步的 Thought
        # 提示词中包含了之前生成的 Reflections (如果有)
        self.scratchpad += f"\nThought {self.step_n}:"
        let_me_think_dict = self.prompt_agent(prefix=f"Thought {self.step_n}:" ...)
        self.scratchpad += " " + let_me_think_dict["thought"]

        # 2. 行动 (Act)
        # 要求 LLM 基于当前 Thought 生成具体的 Action (工具调用)
        self.scratchpad += f"\nAction {self.step_n}:"
        action_dict = self.prompt_agent(prefix=f"Action {self.step_n}:" ...)

        # 3. 观察 (Observe & Tool Execution)
        # 解析 Action 对应的工具名称和参数
        action_type = action_dict["action"]
        argument = action_dict["action_input"]
        
        # 遍历工具列表找到匹配的工具并执行
        for tool in self.cbm_tools:
            if tool.name.lower() == action_type.lower():
                # 执行工具
                tool_output = tool.run(tool_input=dictionary)
                
                # 将工具返回结果写入暂存区，作为下一步观察的输入
                self.scratchpad += tool_output 
                break
        
        # 循环继续，直到 LLM 输出 "Finish" 或 "Final Answer"
```

### 3. ReviewerAgent：独立评估

此类作为一个独立的 Agent，负责对结果进行第三方视角的审计。

```python
# src/reactxen/agents/reviewer_agent/agent.py

class ReviewerAgent:
    def evaluate_response(self, question, agent_think, agent_response):
        # 构建评估 Prompt
        # 输入包括：原始问题、Agent 的思考过程 (Scratchpad)、最终答案
        prompt = self.review_prompt.format(
            question=question,
            agent_think=agent_think,
            agent_response=agent_response,
        )

        # 调用 LLM 进行评估
        # 要求 LLM 必须返回 JSON 格式结果
        review_result = self.llm(prompt, ...)
        
        # 解析返回的 JSON
        # 包含字段：
        # - status: "Success", "Failed", "Partially Completed"
        # - reasoning: 详细的评分理由
        # - suggestions: 如果失败，给出具体的改进建议
        parsed_result = self.extract_and_parse_json(review_result)
        
        return parsed_result
```


### 4. Tiny Trajectory Stores (TTS) 实现机制

在当前的开源代码版本中，**Tiny Trajectory Stores (TTS)** 的概念并非作为一个独立的数据库模块存在，而是通过以下几个组件协同实现的：

1.  **轨迹导出 (Trajectory Export)**: [RAFAAgent](file:///Users/richw/ReActXen/src/reactxen/agents/rafa/agents.py#91-1207) 类中的 [export_trajectory](file:///Users/richw/ReActXen/src/reactxen/agents/react/agents.py#1843-1848) 方法负责将 Agent 的推理过程（System Prompt, Demonstration, Scratchpad, End State）打包成 JSON 格式。这就是“轨迹”的生成端。
2.  **静态存储 (Static Store)**: 目前的“Store”主要体现为预设的高质量轨迹库，集中管理在 [src/reactxen/agents/react/prompts/fewshots.py](file:///Users/richw/ReActXen/src/reactxen/agents/react/prompts/fewshots.py) 中。例如 `MPE_SIMPLE4` 和 `COT_SIMPLE_REFLECTION2` 等变量，本质上就是被“蒸馏”过的、用于 Few-Shot Learning 的优质轨迹片段。
3.  **蒸馏 Agent (Distiller Agent)**: [QueryDistillerAgent](file:///Users/richw/ReActXen/src/reactxen/agents/distiller_agent/agent.py#8-62) (位于 `src/reactxen/agents/distiller_agent`) 具备从原始问题中提取关键信息的能力，这是构建高效轨迹的前置步骤。

**代码示例：轨迹生成与导出**

```python
# src/reactxen/agents/rafa/agents.py

def export_trajectory(self):
    # 1. 初始化轨迹结构
    json_trajectory = {
        "type": "mpe-agent",
        "task": self.question,
        "scratchpad": self.scratchpad,  # 包含完整的 Thought-Action-Observation 记录
        "final_answer": self.answer
    }

    # 2. 解析 Scratchpad 提取结构化步骤
    # 将文本格式的思维链解析为结构化的 list
    thoughts = thought_pattern.findall(self.scratchpad)
    actions = action_pattern.findall(self.scratchpad)
    observations = observation_pattern.findall(self.scratchpad)

    trajectory = []
    for j in range(len(thoughts)):
        new_entry = {
            "thought": thoughts[j],
            "action": actions[j],
            "observation": observations[j] if j < len(observations) else ""
        }
        trajectory.append(new_entry)

    json_trajectory["trajectory"] = trajectory
    
    # 3. 返回可供存储或学习的 JSON 对象
    return json_trajectory
```

**代码示例：静态轨迹库 (The "Tiny Store")**

```python
# src/reactxen/agents/react/prompts/fewshots.py

# 这是一个典型的“Tiny Trajectory”，被存储为字符串常量，用于 Prompt 注入
MPE_SIMPLE4 = """
Question: Is there any failure mode for Gearbox 3...?
Thought 1: I need to get the failure modes...
Action 1: Get Failure Mode Information
Action Input 1: Gearbox 3
Observation 1: Failure modes: ['(1) Bearing Failure...']
...
Action 3: Finish
Action Input 3: Yes, there is a failure mode...
"""
```


**TTS 核心优势 (Advantages)**

1.  **高信噪比 (High Signal-to-Noise Ratio)**:
    *   相比于存储所有历史交互的大型数据库，TTS 仅存储经过筛选和“蒸馏”后的优质轨迹。
    *   这意味着 Agent 在检索参考时，不会被低质量或错误的尝试误导。

2.  **上下文效率 (Context Efficiency)**:
    *   “Tiny” 意味着轨迹是精简的。在有限的 LLM Context Window (上下文窗口) 内，可以由 System Prompt 注入更多的 Few-Shot 示例。
    *   例如，`MPE_SIMPLE4` 通过精炼的 Thought-Action 序列，用极少的 Token 展示了完整的解题逻辑。

3.  **针对性引导 (Targeted Guidance)**:
    *   通过静态库分类（如 `COT` vs `COT_REFLECT`），可以针对不同类型的问题（如纯推理 vs 需要自我纠错的场景）加载最匹配的思维模板。
    *   这实现了类似 "Retrieval-Augmented Generation (RAG)" 的效果，但是是针对“推理模式”的 RAG。

---
*Generated by ReactXen Analysis Bot*
