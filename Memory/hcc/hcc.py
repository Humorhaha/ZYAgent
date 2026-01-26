"""
Hierarchical Cognitive Caching (HCC) - 核心缓存模块

论文核心设计:
    三层缓存 L1 → L2 → L3 对应:
    - Evolving Experience (短期工作记忆)
    - Refined Knowledge (中期策略记忆)  
    - Prior Wisdom (长期持久记忆)

数学符号对应:
    E_t = {e_0, e_1, ..., e_t}  # 事件序列
    L1(t) = E_{t_0-1} ∪ P_{p-1} ∪ E_{t_{p-1}+1:t}  # 工作记忆
    L2(t) = {κ_{t_{r-1}+1:t_{r-1}}}_{r=1}^{p-1}  # 知识缓存
    L3 = {(h_n, w_n)}_{n=1}^N  # 智慧缓存 (embedding-value pairs)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Any
from enum import Enum
from datetime import datetime
import numpy as np
import copy

from .embeddings import EmbeddingProvider, SimpleEmbeddingProvider, batch_cosine_similarity


class EventType(Enum):
    """事件类型 - 论文中将事件分为环境事件和Agent事件"""
    ENVIRONMENT = "environment"  # U: 任务描述、用户指令、执行反馈
    AGENT = "agent"              # A: 代码补丁、命令、计划


@dataclass
class Event:
    """事件对象 - 论文中交互序列的基本单元
    
    论文定义: E_t = {e_0, e_1, ..., e_t}
    假设交替结构: e_{2k} ∈ U (环境), e_{2k+1} ∈ A (Agent)
    """
    step: int                    # 时间步 t
    event_type: EventType        # 事件类型
    content: str                 # 事件内容
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # 标记是否为计划边界事件 (phase boundary)
    is_plan_boundary: bool = False


@dataclass
class KnowledgeUnit:
    """精炼知识单元 - L2缓存的存储单位
    
    论文定义: κ_{i:j} 表示从事件段 E_{i:j} 压缩得到的知识摘要
    包含: 关键判断、实验洞察、进度总结等
    
    示例内容:
    - "Feature X is harmful to model performance"
    - "CV leakage observed under split Y"
    - "ConvNeXt Large with 384x384 works best"
    """
    phase_id: int                # 对应的探索阶段 p
    start_step: int              # 原始事件起始步 i
    end_step: int                # 原始事件结束步 j
    summary: str                 # 精炼后的知识摘要
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass  
class WisdomEntry:
    """智慧条目 - L3缓存的存储单位
    
    论文定义: L3 = {(h_n, w_n)}_{n=1}^N
    其中 h_n = E(d_n) 是任务描述的嵌入向量，w_n 是提炼的智慧文本
    
    智慧类型示例:
    - 模型模板: "Use vit_base_patch16_224 for image classification"
    - 预处理管道: "Apply MixUp + CutMix for augmentation"
    - 超参先验: "Learning rate 1e-4 works well for fine-tuning"
    """
    task_descriptor: str         # d_n: 任务的紧凑描述
    embedding: np.ndarray        # h_n = E(d_n): 嵌入向量
    wisdom: str                  # w_n: 提炼的智慧文本
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class L1EvolvingExperience:
    """L1缓存: Evolving Experience (工作记忆)
    
    论文定义:
        L1(t) = E_{t_0-1} ∪ P_{p-1} ∪ E_{t_{p-1}+1:t}
        
    包含:
        - E_{t_0-1}: 初始代码生成前的所有事件
        - P_{p-1}: 所有阶段边界的计划事件
        - E_{t_{p-1}+1:t}: 当前活跃阶段的原始轨迹
        
    特点:
        - 保持高保真的执行轨迹
        - 支持精确调试和执行感知决策
        - 仅保留活跃执行的原始轨迹，防止上下文饱和
    """
    
    def __init__(self):
        # 初始化事件之前的事件 E_{t_0-1}
        self._initial_events: List[Event] = []
        
        # 计划边界事件 P_{p-1} = {e_{t_r}}_{r=0}^{p-1}
        self._plan_events: List[Event] = []
        
        # 当前活跃阶段的轨迹 E_{t_{p-1}+1:t}
        self._active_trace: List[Event] = []
        
        # 当前阶段ID
        self._current_phase: int = 0
        
        # 初始代码完成的时间步
        self._initial_code_step: Optional[int] = None
    
    def add_event(self, event: Event) -> None:
        """添加事件到L1缓存"""
        if self._initial_code_step is None:
            # 还在初始化阶段
            self._initial_events.append(event)
        elif event.is_plan_boundary:
            # 计划边界事件，存入P
            self._plan_events.append(event)
        else:
            # 普通事件，存入活跃轨迹
            self._active_trace.append(event)
    
    def mark_initial_code_complete(self, step: int) -> None:
        """标记初始代码生成完成
        
        论文: 此时 t_0 确定，E_{t_0-1} 固定
        """
        self._initial_code_step = step
    
    def start_new_phase(self, plan_event: Event) -> None:
        """开始新的探索阶段
        
        论文: 阶段边界 t_p 确定，清空活跃轨迹
        """
        plan_event.is_plan_boundary = True
        self._plan_events.append(plan_event)
        self._active_trace.clear()
        self._current_phase += 1
    
    def get_events_for_promotion(self) -> List[Event]:
        """获取当前活跃轨迹用于提升到L2
        
        返回后应清空活跃轨迹 (由调用方决定)
        """
        return list(self._active_trace)
    
    def clear_active_trace(self) -> None:
        """清空活跃轨迹 (提升后调用)"""
        self._active_trace.clear()
    
    def get_all_events(self) -> List[Event]:
        """获取L1中所有事件 (用于构建上下文)
        
        返回: E_{t_0-1} ∪ P_{p-1} ∪ E_{t_{p-1}+1:t}
        """
        return self._initial_events + self._plan_events + self._active_trace
    
    @property
    def current_phase(self) -> int:
        return self._current_phase
    
    def __len__(self) -> int:
        return len(self._initial_events) + len(self._plan_events) + len(self._active_trace)


class L2RefinedKnowledge:
    """L2缓存: Refined Knowledge (策略记忆)
    
    论文定义:
        L2(t) = {κ_{t_{r-1}+1:t_{r-1}}}_{r=1}^{p-1}
        
    存储:
        - 阶段完成后压缩的知识摘要
        - 关键判断、实验洞察、进度总结
        
    特点:
        - 保留决策理由，去除冗余执行细节
        - 支持跨迭代的长期规划
        - 稳定化策略推理
    """
    
    def __init__(self):
        # 按阶段ID索引的知识单元
        self._knowledge: Dict[int, KnowledgeUnit] = {}
    
    def add_knowledge(self, knowledge: KnowledgeUnit) -> None:
        """添加知识单元 (由阶段级提升产生)"""
        self._knowledge[knowledge.phase_id] = knowledge
    
    def get_knowledge(self, phase_id: int) -> Optional[KnowledgeUnit]:
        """获取特定阶段的知识"""
        return self._knowledge.get(phase_id)
    
    def get_all_knowledge(self) -> List[KnowledgeUnit]:
        """获取所有知识单元 (按阶段排序)"""
        return [self._knowledge[k] for k in sorted(self._knowledge.keys())]
    
    def has_phase(self, phase_id: int) -> bool:
        """检查是否有特定阶段的知识"""
        return phase_id in self._knowledge
    
    def __len__(self) -> int:
        return len(self._knowledge)


class L3PriorWisdom:
    """L3缓存: Prior Wisdom (长期记忆)
    
    论文定义:
        L3 = {(h_n, w_n)}_{n=1}^N
        其中 h_n = E(d_n) 是嵌入向量，w_n 是智慧文本
        
    存储:
        - 任务无关的可迁移策略
        - 可复用的预处理管道
        - 稳定的超参先验
        
    特点:
        - 跨任务持久化
        - 基于语义相似度检索
        - 支持warm-start和知识迁移
    """
    
    def __init__(self, embedding_provider: Optional[EmbeddingProvider] = None):
        """
        Args:
            embedding_provider: 嵌入提供者，默认使用简单实现
        """
        self._embedder = embedding_provider or SimpleEmbeddingProvider()
        self._entries: List[WisdomEntry] = []
        # 预计算的嵌入矩阵 (用于批量检索)
        self._embedding_matrix: Optional[np.ndarray] = None
    
    def add_wisdom(self, task_descriptor: str, wisdom: str, 
                   metadata: Optional[Dict[str, Any]] = None) -> None:
        """添加智慧条目 (由任务级提升产生)
        
        论文: L3 ← L3 ∪ {(E(d_τ), w_τ)}
        """
        embedding = self._embedder.embed(task_descriptor)
        entry = WisdomEntry(
            task_descriptor=task_descriptor,
            embedding=embedding,
            wisdom=wisdom,
            metadata=metadata or {}
        )
        self._entries.append(entry)
        # 使嵌入矩阵缓存失效
        self._embedding_matrix = None
    
    def prefetch(self, task_descriptor: str, threshold: float = 0.5) -> List[str]:
        """预取相关智慧 (Context Prefetching)
        
        论文公式:
            Ω_τ = {w_n | (h_n, w_n) ∈ L3, cos(q, h_n) > δ}
            其中 q = E(d_τ), δ 是相似度阈值
            
        Args:
            task_descriptor: 当前任务描述 d_τ
            threshold: 相似度阈值 δ
            
        Returns:
            检索到的智慧文本列表 Ω_τ
        """
        if not self._entries:
            return []
        
        # 计算查询嵌入 q = E(d_τ)
        query = self._embedder.embed(task_descriptor)
        
        # 确保嵌入矩阵已构建
        if self._embedding_matrix is None:
            self._embedding_matrix = np.stack([e.embedding for e in self._entries])
        
        # 计算余弦相似度 cos(q, h_n)
        similarities = batch_cosine_similarity(query, self._embedding_matrix)
        
        # 筛选超过阈值的条目
        results = []
        for i, sim in enumerate(similarities):
            if sim > threshold:
                results.append(self._entries[i].wisdom)
        
        return results
    
    def retrieve_top_k(
        self, 
        query: str, 
        k: int = 3,
        min_threshold: float = 0.1,
    ) -> list:
        """按 Cosine 相似度检索 Top-K 最相关的 Wisdom
        
        Args:
            query: 查询文本
            k: 返回的数量
            min_threshold: 最小相似度阈值
            
        Returns:
            List of (wisdom_text, similarity_score) 元组，按相似度降序
        """
        if not self._entries:
            return []
        
        # 计算查询嵌入
        query_embedding = self._embedder.embed(query)
        
        # 确保嵌入矩阵已构建
        if self._embedding_matrix is None:
            self._embedding_matrix = np.stack([e.embedding for e in self._entries])
        
        # 计算余弦相似度
        similarities = batch_cosine_similarity(query_embedding, self._embedding_matrix)
        
        # 按相似度排序
        scored_entries = [
            (self._entries[i].wisdom, float(sim))
            for i, sim in enumerate(similarities)
            if sim > min_threshold
        ]
        scored_entries.sort(key=lambda x: x[1], reverse=True)
        
        return scored_entries[:k]
    
    def get_all_entries(self) -> List[WisdomEntry]:
        """获取所有智慧条目"""
        return list(self._entries)
    
    def __len__(self) -> int:
        return len(self._entries)


class HierarchicalCognitiveCache:
    """层次化认知缓存 - HCC主控制器
    
    论文核心设计:
        将瞬态上下文与稳定认知状态分离，使ML-Master 2.0能够
        在不被执行细节淹没的情况下保持连贯的长时探索。
        
    使用示例:
        ```python
        hcc = HierarchicalCognitiveCache()
        
        # 预取先验智慧
        prior_wisdom = hcc.prefetch("image classification task")
        
        # 添加事件
        hcc.add_event(Event(step=0, event_type=EventType.ENVIRONMENT, content="Task: ..."))
        hcc.add_event(Event(step=1, event_type=EventType.AGENT, content="Plan: ..."))
        
        # 构建上下文 (Context Hit)
        context = hcc.build_context()
        
        # 阶段完成，提升到L2 (Phase-level Promotion)
        hcc.promote_phase(summary="Key insights from phase 1...")
        
        # 任务完成，提升到L3 (Task-level Promotion)
        hcc.promote_task(task_descriptor="...", wisdom="...")
        ```
    """
    
    def __init__(self, embedding_provider: Optional[EmbeddingProvider] = None):
        """
        Args:
            embedding_provider: 嵌入提供者，用于L3的语义检索
        """
        self.l1 = L1EvolvingExperience()
        self.l2 = L2RefinedKnowledge()
        self.l3 = L3PriorWisdom(embedding_provider)
    
    def add_event(self, event: Event) -> None:
        """添加事件到L1缓存"""
        self.l1.add_event(event)
    
    def mark_initial_code_complete(self, step: int) -> None:
        """标记初始代码生成完成"""
        self.l1.mark_initial_code_complete(step)
    
    def start_new_phase(self, plan_content: str, step: int) -> None:
        """开始新的探索阶段"""
        plan_event = Event(
            step=step,
            event_type=EventType.AGENT,
            content=plan_content,
            is_plan_boundary=True
        )
        self.l1.start_new_phase(plan_event)
    
    def prefetch(self, task_descriptor: str, threshold: float = 0.5) -> List[str]:
        """预取先验智慧 (Context Prefetching)
        
        论文: Ω_τ = {w_n | cos(q, h_n) > δ}
        """
        return self.l3.prefetch(task_descriptor, threshold)
    
    def build_context(self, max_phases: Optional[int] = None) -> str:
        """构建上下文 (Context Hit)
        
        论文公式:
            Ψ_t(k) = e_k                    if e_k ∈ L1(t)
                   = κ_{t_{r-1}+1:t_{r-1}}  if e_k ∉ L1(t), e_k ∈ L2(t)
                   = ∅                      otherwise
            
            C_{t-1} = g(E_{t-1}) = concat{Ψ_t(k)}_{k=0}^{t-1}
            
        Args:
            max_phases: Markov 约束 - 只返回最近 N 个 phase 的内容
                        None 表示返回全部
            
        策略:
            - L1中的事件以原始形式返回
            - 已完成阶段用L2中的精炼知识替代
        """
        parts = []
        
        # 计算 phase 截断点
        current_phase = self.l1.current_phase
        min_phase = 0 if max_phases is None else max(0, current_phase - max_phases + 1)
        
        # 1. 添加L1中的事件 (只添加活跃轨迹，初始事件根据 max_phases 决定)
        if max_phases is None or max_phases <= 0:
            # 无限制，返回所有 L1 事件
            for event in self.l1.get_all_events():
                parts.append(f"[Step {event.step}] {event.content}")
        else:
            # Markov 约束：只返回活跃轨迹
            for event in self.l1._active_trace:
                parts.append(f"[Step {event.step}] {event.content}")
        
        # 2. 添加L2中的精炼知识 (受 max_phases 约束)
        for knowledge in self.l2.get_all_knowledge():
            if knowledge.phase_id >= min_phase:
                parts.append(f"[Phase {knowledge.phase_id} Summary] {knowledge.summary}")
        
        return "\n\n".join(parts)
    
    def promote_phase(self, summary: str) -> None:
        """阶段级提升 (Phase-level Promotion)
        
        论文公式:
            κ_p = P1({σ_{p,i,j}}_{(i,j)∈I_p})
            L2 ← L2 ∪ {κ_p}
            L1 ← L1 \ {e | e ∈ σ_{p,i,j}}
        
        Args:
            summary: LLM生成的阶段知识摘要
        """
        # 获取当前阶段的活跃轨迹起止步
        traces = self.l1.get_events_for_promotion()
        if not traces:
            return
        
        start_step = traces[0].step if traces else 0
        end_step = traces[-1].step if traces else 0
        
        # 创建知识单元并添加到L2
        knowledge = KnowledgeUnit(
            phase_id=self.l1.current_phase,
            start_step=start_step,
            end_step=end_step,
            summary=summary
        )
        self.l2.add_knowledge(knowledge)
        
        # 清空L1中的活跃轨迹
        self.l1.clear_active_trace()
    
    def promote_task(self, task_descriptor: str, wisdom: str) -> None:
        """任务级提升 (Task-level Promotion)
        
        论文公式:
            w_τ = P2(d_τ, L1(t_max), L2(t_max), h(E_{t_max}))
            L3 ← L3 ∪ {(E(d_τ), w_τ)}
            
        Args:
            task_descriptor: 任务描述 d_τ
            wisdom: LLM提炼的任务智慧 w_τ
        """
        self.l3.add_wisdom(task_descriptor, wisdom)
    
    def get_stats(self) -> Dict[str, int]:
        """获取缓存统计信息"""
        return {
            "l1_events": len(self.l1),
            "l2_knowledge_units": len(self.l2),
            "l3_wisdom_entries": len(self.l3),
            "current_phase": self.l1.current_phase
        }

    def create_snapshot(self) -> "HierarchicalCognitiveCache":
        """创建 HCC 的深拷贝快照 (用于 Async MCTS)
        
        MCTS 在独立线程/进程中运行时，需要一份独立的状态副本，
        以免受到主线程更新的影响 (State Drift 保护)。
        """
        # 使用 deepcopy 创建完全独立的副本
        # 注意: EmbeddingProvider 通常是无状态的或共享的，
        # 如果有大模型加载，需注意 copy 行为。
        # 这里假设 embedder 可以浅拷贝或重新引用。
        
        # 暂时简单处理: 深拷贝整个对象
        # 在生产环境中，可能需要更精细的控制 (如不拷贝 Embedding 缓存)
        return copy.deepcopy(self)
