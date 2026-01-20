"""
Hierarchical Cognitive Caching (HCC) - ML-Master 2.0

基于论文设计的三层认知缓存系统，用于管理长时任务中的上下文信息。

三层缓存架构:
- L1 (Evolving Experience): 工作记忆，保存高保真执行轨迹
- L2 (Refined Knowledge): 策略记忆，存储阶段性精炼知识
- L3 (Prior Wisdom): 长期记忆，存储跨任务可迁移智慧

Context Migration机制:
- Prefetch: 从L3预取相关先验智慧
- Hit: 构建上下文时的缓存命中策略
- Promote: 将执行轨迹提升压缩为知识/智慧
"""

from .hcc import (
    HierarchicalCognitiveCache,
    L1EvolvingExperience,
    L2RefinedKnowledge,
    L3PriorWisdom,
    Event,
    EventType,
    KnowledgeUnit,
    WisdomEntry,
)

from .context_migration import (
    ContextMigrator,
)

from .embeddings import (
    EmbeddingProvider,
    SimpleEmbeddingProvider,
)

__all__ = [
    # Core Cache Classes
    "HierarchicalCognitiveCache",
    "L1EvolvingExperience",
    "L2RefinedKnowledge",
    "L3PriorWisdom",
    # Data Types
    "Event",
    "EventType",
    "KnowledgeUnit",
    "WisdomEntry",
    # Context Migration
    "ContextMigrator",
    # Embeddings
    "EmbeddingProvider",
    "SimpleEmbeddingProvider",
]
