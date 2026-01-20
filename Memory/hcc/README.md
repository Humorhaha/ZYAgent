# Hierarchical Cognitive Caching (HCC)

基于 ML-Master 2.0 论文实现的层次化认知缓存系统。该模块旨在通过三层缓存架构解决长时程机器学习任务中的上下文管理问题，防止上下文溢出并实现跨任务的知识积累。

## 核心架构

HCC 模拟计算机内存层次结构，将上下文分为三层：

| 层级 | 名称 | 存储内容 | 特点 | 对应论文 |
|------|------|----------|------|----------|
| **L1** | Evolving Experience | 原始执行轨迹 (代码、日志、计划) | 高保真、短期工作记忆 | Sec 3.3.1 |
| **L2** | Refined Knowledge | 阶段性知识摘要 | 紧凑、中期策略记忆 | Sec 3.3.2 |
| **L3** | Prior Wisdom | 跨任务可迁移智慧 | 向量检索、长期持久记忆 | Sec 3.3.3 |

## Context Migration (上下文迁移)

系统通过三种机制管理信息在缓存层间的流动：

1.  **Context Prefetching (预取)**: 任务开始时，利用语义检索从 L3 获取相关的先验智慧。
2.  **Context Hit (命中)**: 构建上下文时，优先使用 L1 的原始轨迹；对于已完成的阶段，回退使用 L2 的精炼摘要。
3.  **Context Promotion (提升)**:
    *   **Phase-level (P1)**: 阶段结束时，LLM 将 L1 轨迹压缩为 L2 知识单元。
    *   **Task-level (P2)**: 任务结束时，LLM 将积累的知识蒸馏为 L3 智慧条目。

## 文件结构

- `hcc.py`: 核心实现，包含 `HierarchicalCognitiveCache` 主类及 `L1`/`L2`/`L3` 缓存类。
- `context_migration.py`: 负责管理信息流动 (Prefetch/Hit/Promote) 及 LLM 交互。
- `embeddings.py`: 向量嵌入接口及简单实现，用于 L3 检索。
- `__init__.py`: 模块导出及类型定义。

## 集成与使用 (Integration)

### 1. 基础调用

其他模块可以通过以下方式集成 HCC：

```python
from Memory.hcc import (
    HierarchicalCognitiveCache, 
    ContextMigrator,
    Event, 
    EventType
)

# 1. 初始化系统
# 默认使用 Mock LLM 和 Simple Embedding，仅用于测试
hcc = HierarchicalCognitiveCache()
migrator = ContextMigrator(hcc)

# 2. 任务初始化 (Context Prefetching)
# 系统会自动查询 L3 (Prior Wisdom) 并将其包含在初始上下文中
initial_context = migrator.initialize_context(
    task_description="Image classification on plant leaves",
    user_instruction="Maximize F1 score"
)
print(f"Initial Context:\n{initial_context}")

# 3. 记录事件 (L1 Cache)
# 在任务执行循环中，持续记录环境反馈和 Agent 动作
hcc.add_event(Event(
    step=0, 
    event_type=EventType.ENVIRONMENT, 
    content="Dataset loaded successfully. Shape: (1000, 3, 224, 224)"
))

# 4. 构建上下文 (Context Hit)
# 获取当前上下文，系统会自动组合 L1 的近期事件和 L2 的历史摘要
context = migrator.build_context()

# 5. 阶段结束提升 (P1 Promotion)
# 当一个探索阶段结束（如一次完整的代码修改验证循环），触发阶段提升
# 这将调用 LLM 将详细轨迹压缩为 L2 知识单元
migrator.promote_phase()

# 6. 任务结束提升 (P2 Promotion)
# 任务彻底完成时，触发任务提升
# 这将调用 LLM 蒸馏出可迁移的智慧存入 L3
migrator.promote_task(
    task_descriptor="Image classification on plant leaves",
    final_code="model = ResNet50(pretrained=True)...",
    final_result="F1: 0.98"
)
```

### 2. 生产环境集成

在生产环境中，你需要注入真实的 LLM 和 Embedding 后端：

```python
from Memory.hcc import ContextMigrator, HierarchicalCognitiveCache, EmbeddingProvider
import numpy as np

# 1. 实现 LLM protocol (用于 ContextMigrator)
class RealLLMProvider:
    def generate(self, prompt: str) -> str:
        # 调用你的 LLM API (如 OpenAI/Claude/DeepSeek)
        return call_my_llm_api(prompt)

# 2. 实现 EmbeddingProvider (用于 HierarchicalCognitiveCache)
class OpenAIEmbedding(EmbeddingProvider):
    def embed(self, text: str) -> np.ndarray:
        # 调用 Embedding API
        vector = get_openai_embedding(text)
        return np.array(vector)

# 3. 注入依赖
real_embedding = OpenAIEmbedding()
real_llm = RealLLMProvider()

hcc = HierarchicalCognitiveCache(embedding_provider=real_embedding)
migrator = ContextMigrator(hcc, llm_provider=real_llm)
```

## 扩展说明

- **LLM Backend**: 默认使用 `MockLLMProvider`，生产环境需在 `context_migration.py` 中实现对接真实 LLM。
- **Embedding**: 默认使用 `SimpleEmbeddingProvider` (基于 hash 的伪向量)，生产环境需在 `embeddings.py` 中对接 OpenAI 或 HuggingFace 模型。
