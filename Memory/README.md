# Memory Module: The "Teacher & Notebook" Architecture

ReactXen 的 Memory 模块采用创新的 **"Teacher & Notebook" (导师与笔记本)** 混合架构，将静态的高质量教学范例与动态的长时程任务记忆完美融合。

## 核心设计理念

为了实现 "**Keep it Simple and Stupid but Efficient**"，我们将记忆系统分为两个正交的维度：

### 1. The Teacher (TTS - Tiny Trajectory Stores)
*   **角色**: 严厉的导师。
*   **性质**: **静态、只读 (Read-Only)**。
*   **功能**: 提供经过蒸馏的 "Golden Trajectories" (黄金范例)。
*   **目的**: 教 Agent "**怎么做**" (Standard Reasoning Process)。解决 Cold Start 问题，规范思维范式。

### 2. The Notebook (HCC - Hierarchical Cognitive Caching)
*   **角色**: 勤奋的记录员。
*   **性质**: **动态、读写 (Read-Write)**。
*   **功能**: 管理当前任务的状态，并随着时间推移自动压缩历史。
    *   **L1 (工作记忆)**: 记录最近的详细步骤。
    *   **L2 (短期总结)**: 自动压缩过往阶段的经验。
    *   **L3 (长期智慧)**: 沉淀跨任务的可复用智慧。
*   **目的**: 记 Agent "**做了什么**" (Task Progress)。解决 Context Window 限制，积累自我经验。

---

## 目录结构

```text
Memory/
├── memory.py           # [入口] 统一接口 AgentMemory
├── demo.py             # 演示脚本
├── hcc/                # HCC 子系统 (Notebook)
│   ├── hcc.py          # 核心缓存实现 (L1/L2/L3)
│   ├── context_migration.py # 记忆迁移逻辑
│   └── ...
└── tts/                # TTS 子系统 (Teacher)
    ├── tts.py          # 轨迹存储与检索
    └── examples/       # 轨迹文件 (JSON/TXT)
```

---

## 快速开始

使用统一接口 `AgentMemory` 可以轻松接入这个强力架构。

### 1. 基础集成

```python
from Memory.memory import AgentMemory, TrajectoryCategory

# 初始化
memory = AgentMemory()

# 1. Load Examples (Teacher is ready)
memory.load_examples("./examples/")

# 2. Start Task (Notebook is opened)
initial_context = memory.start_task(
    task_description="Train a ResNet on CIFAR-10",
    user_instruction="Target accuracy: 95%"
)

# 3. Build Prompt (Construct the "Sandwich")
prompt = memory.build_prompt(
    task="What is the first step?",
    system_prompt="You are an AI assistant.",
    example_category=TrajectoryCategory.DATA_SCIENCE  # 指定要看哪类参考书
)

print(prompt)
# Output structure:
# [System Prompt]
# [TTS Examples] (Teacher's guidance)
# [HCC Context] (Notebook's content)
# [Current Task]
```

### 2. 运行时记录

在 Agent 的思考-行动循环中，实时记录事件：

```python
# 记录 Agent 的思考和行动
memory.record_thought_action(
    thought="I need to check data balance.",
    action="check_balance(data)",
    observation="Class 0: 5000, Class 1: 500"
)

# 记录环境/系统的反馈
memory.record_event("System warning: Disk full", EventType.ENVIRONMENT)
```

### 3. 连接真实 LLM (生产环境)

为了让 HCC 能够自动总结 (Promotion)，你需要注入真实的 LLM：

```python
class MyLLM:
    def generate(self, prompt):
        return call_gpt4(prompt)

memory = AgentMemory(llm_provider=MyLLM())

# 现在，当步数达到阈值 (默认 20)，System 会自动调用 MyLLM 
# 将 L1 的详细历史压缩为 L2 的摘要。
```

---

## 高级特性

### Sandwich Prompt 结构

`memory.build_prompt()` 会自动构建如下的高效 Prompt 结构：

```text
+---------------------------------------------------------------+
| System Instruction                                            |
+---------------------------------------------------------------+
| ## Reference Examples (TTS)                                   |
| Example 1: Q: ... Thought: ... Action: ...                    |
+---------------------------------------------------------------+
| ## Task Context (HCC)                                         |
| >> Prior Wisdom (L3): "Learning rate 1e-4 works best..."      |
| >> Phase Summary (L2): "Data loaded, Model defined..."        |
| >> Recent Steps (L1): [Step 20] Action: Train...              |
+---------------------------------------------------------------+
| ## Current Task                                               |
| User: "What's next?"                                          |
+---------------------------------------------------------------+
```

### 自动记忆维护 (Auto-Maintenance)

为了保持 SIMPLE，采用了基于规则的维护策略：
*   **Auto-Promotion**: 每 `hcc_auto_promote_steps` (默认 20) 步，或者当 L1 长度过长时，自动触发 L1 -> L2 的压缩。
*   **Smart Retrieval**: TTS 检索只返回 Top-K (默认 2) 个最相关的例子，严格控制 Token 消耗。

---

## 配置说明

通过 `MemoryConfig` 调整行为：

```python
from Memory.memory import MemoryConfig

config = MemoryConfig(
    tts_max_examples=3,       # 每次看 3 个例子
    hcc_auto_promote_steps=15, # 每 15 步总结一次
    hcc_similarity_threshold=0.6 # L3 检索更严格
)

memory = AgentMemory(config=config)
```
