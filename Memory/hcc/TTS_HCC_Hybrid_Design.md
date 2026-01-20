# TTS + HCC Hybrid Architecture: The "Teacher & Notebook" Model

## 1. 核心理念 (Core Concept)

为了保持架构的 **Simple (KISS)** 与 **Efficient**，我们将两者分工如下：

*   **TTS (The Teacher)**: **只读 (Read-Only)**。
    *   **作用**: 提供 "Cold Start" 能力。在 System Prompt 中注入 1-2 个最匹配的黄金范例。
    *   **目的**: 规范 Agent 的输出格式和推理深度 (Format & Reasoning Style)。
*   **HCC (The Notebook)**: **读写 (Read-Write)**。
    *   **作用**: 管理 "Long Context"。作为动态的记忆容器，存储当前任务的所有历史。
    *   **目的**: 解决 Context Window 限制，积累任务特定的经验。

## 2. Prompt 结构设计 (The "Sandwich" Prompt)

这是本架构的核心。我们将 Context Window 划分为三个刚性区域：

```text
+---------------------------------------------------------------+
| [Section 1: System Instruction]                               |
| "你是一个专业的 AI 助手..."                                     |
+---------------------------------------------------------------+
| [Section 2: TTS Few-Shot Examples] (The Teacher)              |
| "以下是标准的高质量解题范例，请模仿其思考方式："                    |
| Example 1: Q: ... Thought: ... Action: ...                    |
| Example 2: Q: ... Thought: ... Action: ...                    |
+---------------------------------------------------------------+
| [Section 3: HCC Dynamic Context] (The Notebook)               |
| "以下是你当前的任务进度和积累的知识："                             |
|                                                               |
| >> Prior Wisdom (L3): "以前的任务告诉你：学习率用 1e-4..."        |
|                                                               |
| >> Phase Summaries (L2):                                      |
| "Phase 1 Summary: 数据加载成功，但发现存在缺失值..."              |
| "Phase 2 Summary: 尝试了 MLP 模型，效果不佳 (Acc: 40%)..."      |
|                                                               |
| >> Recent Trace (L1): (最新的 10-20 步)                        |
| [Step 21] Action: ...                                         |
| [Step 22] Observation: ...                                    |
+---------------------------------------------------------------+
| [Section 4: Current Turn]                                     |
| User: "下一步我们要怎么改进模型？"                                |
| Assistant: (Agent generate here...)                           |
+---------------------------------------------------------------+
```

## 3. 精简版实现逻辑 (Simple Implementation)

不需要复杂的调度器，只需要在大循环中嵌入简单的 Hook。

```python
class HybridAgent:
    def __init__(self, tts_store, hcc_memory):
        self.tts = tts_store          # 静态库
        self.hcc = hcc_memory         # 动态 HCC 实例
        self.llm = YourLLM()

    def run(self, task_description):
        # 1. 初始化 & 预取
        # 从 TTS 拿 2 个最好的例子，从 HCC L3 拿相关的过往智慧
        tts_examples = self.tts.retrieve(task_description, k=2)
        hcc_wisdom = self.hcc.prefetch(task_description)
        
        # 初始化 HCC 任务上下文
        self.hcc.initialize_context(task_description)

        step_count = 0
        
        while not self.is_finished():
            # 2. 构建 "三明治" Prompt
            # 这里的 build_context() 会自动把 HCC 的 L2(摘要) 和 L1(近期) 拼好
            dynamic_context = self.hcc.build_context() 
            
            full_prompt = f"""
            {SYSTEM_PROMPT}
            
            == EXAMPLES (Learn from these) ==
            {tts_examples}
            
            == CURRENT TASK CONTEXT (Your memory) ==
            {hcc_wisdom}
            {dynamic_context}
            
            == YOUR GOAL ==
            {task_description}
            """

            # 3. LLM 执行
            response = self.llm.generate(full_prompt)
            thought, action = parse(response)
            
            # 执行 Action...
            observation = env.step(action)
            
            # 4. 写入 HCC (L1)
            event = Event(step=step_count, content=f"Thought: {thought}\nAction: {action}\nObs: {observation}")
            self.hcc.add_event(event)

            # 5. [关键简化] 自动维护内存 (Keeping it Simple)
            # 不用复杂的 AI 判断，简单的规则：每 20 步强制总结一次 (Promotion)
            # 或者当 L1 的长度超过 Token 限制时触发
            if self.hcc.l1_needs_compression(): 
                print(">> Memory full, compressing recent history into Summary...")
                self.hcc.promote_phase()  # 调用 LLM 把 L1 压缩成 L2
            
            step_count += 1

        # 6. 任务结束，存入智慧 (L3)
        self.hcc.promote_task(task_description, final_result)
```

## 4. 为什么这样设计最简单高效？

1.  **解耦清晰**：
    *   想提高 Agent 的逻辑能力？去改 **TTS** 里的例子。
    *   想提高 Agent 的记忆力？不需要改 Prompt 结构，HCC 会自动在后台压缩 L1 -> L2。
2.  **刚性规则替代复杂逻辑**：
    *   使用 **Rule-based Promotion** (例如："每 20 步" 或 "Context > 80%") 替代复杂的 Stage 识别模型。这大大降低了工程复杂度，虽然牺牲了一点点语义上的精准切分，但在工程上极其稳健。
3.  **Token 预算可控**：
    *   TTS 占用固定 ~1000 Tokens。
    *   HCC L2 (Summaries) 随时间缓慢增长，但因为是摘要，且 L3 只取 Top-K，所以整体 Context 几乎永远不会溢出。

## 5. 目录结构建议

```
Memory/
├── hcc/              # HCC 动态内存模块 (现有的)
│   ├── hcc.py
│   └── ...
└── tts/              # TTS 静态库模块 (新建)
    ├── store.py      # 简单的字典或 JSON 加载器
    └── examples/     # 存放 .txt 或 .json 的范例文件
        ├── data_science_fewshot.txt
        └── general_reasoning_fewshot.txt
```
