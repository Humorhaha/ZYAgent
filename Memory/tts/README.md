# Tiny Trajectory Stores (TTS)

基于 ReactXen 架构设计的静态轨迹存储系统，用于为 Agent 提供高质量的 Few-Shot 示例。

## 核心理念

TTS 是一个 **只读 (Read-Only)** 的精炼轨迹库，解决的是 "**如何教 Agent 正确的推理模式**" 的问题。

与 HCC 的关系：
- **TTS** 是 "教科书"：提供静态的标准解题范例，教 Agent 怎么做
- **HCC** 是 "笔记本"：管理动态的任务进度和经验，记 Agent 做了什么

## 文件结构

- `tts.py`: 核心实现，包含 `TinyTrajectoryStore` 主类及 `Trajectory`/`TrajectoryStep` 数据类
- `__init__.py`: 模块导出

## 集成与使用

### 1. 基础调用

```python
from Memory.tts import (
    TinyTrajectoryStore,
    Trajectory,
    TrajectoryStep,
    TrajectoryCategory,
)

# 1. 初始化
store = TinyTrajectoryStore()

# 2. 从目录批量加载轨迹
store.load_from_directory("./examples/fewshots/")

# 3. 或手动添加轨迹
trajectory = Trajectory(
    trajectory_id="example_1",
    category=TrajectoryCategory.DATA_SCIENCE,
    task="How to train a ResNet model?",
    steps=[
        TrajectoryStep(step_id=1, thought="I need to load the data first", action="LoadData", action_input="dataset.csv", observation="Data loaded: 1000 rows"),
        TrajectoryStep(step_id=2, thought="Now I should define the model", action="DefineModel", action_input="ResNet50", observation="Model created"),
        TrajectoryStep(step_id=3, thought="Time to train", action="Train", action_input="epochs=10", observation="Training complete, accuracy=95%"),
    ],
    final_answer="Use ResNet50 with 10 epochs training."
)
store.add(trajectory)

# 4. 检索相关轨迹
examples = store.retrieve(
    query="How to train an image classifier?",
    category=TrajectoryCategory.DATA_SCIENCE,
    k=2
)

# 5. 格式化为 Prompt 文本
prompt_text = store.format_for_prompt(examples, max_tokens=1000)
print(prompt_text)
```

### 2. 从文本文件加载

TTS 支持从标准 ReAct 格式的文本文件加载轨迹：

```text
Question: How to calculate the factorial of 5?
Thought 1: I need to multiply 5 * 4 * 3 * 2 * 1
Action 1: Calculate
Action Input 1: 5 * 4 * 3 * 2 * 1
Observation 1: 120
Final Answer: The factorial of 5 is 120.

---

Question: What is the capital of France?
Thought 1: I should search for this information
Action 1: Search
Action Input 1: capital of France
Observation 1: Paris is the capital of France
Final Answer: Paris
```

多个轨迹用 `---` 分隔。

### 3. 与 HCC 混合使用

```python
from Memory.tts import TinyTrajectoryStore, TrajectoryCategory
from Memory.hcc import HierarchicalCognitiveCache, ContextMigrator

# 初始化两个系统
tts = TinyTrajectoryStore()
tts.load_from_directory("./examples/")

hcc = HierarchicalCognitiveCache()
migrator = ContextMigrator(hcc)

# 构建 "三明治" Prompt
def build_hybrid_prompt(task_description: str) -> str:
    # 1. 从 TTS 获取 Few-Shot 示例
    tts_examples = tts.retrieve(task_description, k=2)
    examples_text = tts.format_for_prompt(tts_examples, max_tokens=1000)
    
    # 2. 从 HCC 获取动态上下文
    hcc_context = migrator.build_context()
    
    # 3. 组装 Prompt
    return f"""
{SYSTEM_PROMPT}

== EXAMPLES (Learn from these) ==
{examples_text}

== TASK CONTEXT (Your memory) ==
{hcc_context}

== YOUR GOAL ==
{task_description}
"""
```

## 轨迹类别

TTS 支持按类别管理轨迹：

| 类别 | 描述 | 使用场景 |
|------|------|----------|
| `GENERAL` | 通用推理 | 默认类别 |
| `DATA_SCIENCE` | 数据科学/ML | 机器学习任务 |
| `CODE_GENERATION` | 代码生成 | 编程任务 |
| `REASONING` | 逻辑推理 | 数学/逻辑题 |
| `QA` | 问答 | 知识问答 |
| `TOOL_USE` | 工具调用 | API/函数调用 |
| `REFLECTION` | 自我反思 | 纠错场景 |

## 语义检索 (可选)

如果需要基于语义相似度检索轨迹，可以注入 Embedding Provider：

```python
from Memory.hcc import SimpleEmbeddingProvider  # 复用 HCC 的

embedder = SimpleEmbeddingProvider()
store = TinyTrajectoryStore(embedding_provider=embedder)
```

## 扩展说明

- **Embedding Provider**: 默认使用关键词匹配，生产环境建议对接 OpenAI 或 HuggingFace Embeddings
- **Token 预算**: `format_for_prompt()` 支持 `max_tokens` 参数，自动截断超出的轨迹
