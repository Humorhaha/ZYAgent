"""
Memory 模块统一接口

将 HCC (Hierarchical Cognitive Caching) 和 TTS (Tiny Trajectory Stores) 
整合为统一的 Memory 系统，对外提供简洁的 API。

设计理念 (Teacher & Notebook Model):
    - TTS (Teacher): 只读的静态轨迹库，提供 Few-Shot 示例
    - HCC (Notebook): 读写的动态缓存，管理任务进度和积累智慧

使用示例:
    ```python
    from Memory.memory import AgentMemory
    
    memory = AgentMemory()
    memory.load_examples("./examples/")
    
    prompt = memory.build_prompt(
        task="Train a classifier",
        system_prompt="You are an AI assistant."
    )
    ```
"""

from typing import List, Dict, Optional, Any, Union, Protocol
from pathlib import Path
from dataclasses import dataclass

# HCC 模块导入
# HCC 模块导入
from .hcc import (
    HierarchicalCognitiveCache,
    ContextMigrator,
    Event,
    EventType,
    EmbeddingProvider,
    SimpleEmbeddingProvider,
)

# TTS 模块导入
from .tts import (
    TinyTrajectoryStore,
    Trajectory,
    TrajectoryStep,
    TrajectoryCategory,
)


class LLMProvider(Protocol):
    """LLM 提供者协议"""
    def generate(self, prompt: str) -> str:
        ...


@dataclass
class MemoryConfig:
    """Memory 配置"""
    # TTS 配置
    tts_max_examples: int = 2           # 检索的最大示例数
    tts_max_tokens: int = 1500          # 示例的最大 Token 数
    
    # HCC 配置
    hcc_similarity_threshold: float = 0.5  # L3 检索的相似度阈值
    hcc_auto_promote_steps: int = 20       # 自动触发 Phase Promotion 的步数
    
    # Prompt 配置
    examples_header: str = "## Reference Examples\n\nLearn from these examples:\n\n"
    context_header: str = "## Task Context\n\nYour current progress:\n\n"


class AgentMemory:
    """Agent Memory 统一接口
    
    整合 TTS 和 HCC，提供简洁的记忆管理 API。
    
    核心功能:
        1. 示例管理: 加载和检索 Few-Shot 示例 (TTS)
        2. 进度追踪: 记录任务执行过程 (HCC L1)
        3. 知识积累: 阶段性总结和提升 (HCC L2/L3)
        4. Prompt 构建: 自动组装上下文
        
    Attributes:
        tts: Tiny Trajectory Store 实例
        hcc: Hierarchical Cognitive Cache 实例
        migrator: HCC 的 Context Migrator
        config: 配置对象
    """
    
    def __init__(
        self,
        config: Optional[MemoryConfig] = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
        llm_provider: Optional[LLMProvider] = None,
    ):
        """
        Args:
            config: 配置对象，None 则使用默认配置
            embedding_provider: 嵌入提供者，用于语义检索
            llm_provider: LLM 提供者，用于 HCC 的知识压缩
        """
        self.config = config or MemoryConfig()
        self._embedder = embedding_provider or SimpleEmbeddingProvider()
        
        # 初始化 TTS
        self.tts = TinyTrajectoryStore(embedding_provider=self._embedder)
        
        # 初始化 HCC
        self.hcc = HierarchicalCognitiveCache(embedding_provider=self._embedder)
        self.migrator = ContextMigrator(
            hcc=self.hcc,
            llm_provider=llm_provider,
            similarity_threshold=self.config.hcc_similarity_threshold,
        )
        
        # 内部状态
        self._step_count = 0
        self._task_description = ""
    
    def create_snapshot(self) -> 'AgentMemory':
        """创建 Memory 的深拷贝快照 (用于 System 2 异步计算)
        
        Returns:
            独立的 AgentMemory 副本，包含当前的 TTS 和 HCC 状态。
        """
        import copy
        # 创建一个新的实例
        snapshot = AgentMemory(
            config=self.config,
            embedding_provider=self._embedder,
            # 注意: llm_provider 如果有状态需要深拷贝，通常它是无状态服务的 Wrapper
            # 这里浅拷贝引用即可
            llm_provider=self.migrator.llm_provider 
        )
        
        # 复制内部状态
        snapshot._step_count = self._step_count
        snapshot._task_description = self._task_description
        
        # 复制 HCC (使用 HCC 的深拷贝机制)
        snapshot.hcc = self.hcc.create_snapshot()
        
        # 重新绑定 Migrator 到新的 HCC
        snapshot.migrator = ContextMigrator(
            hcc=snapshot.hcc,
            llm_provider=snapshot.migrator.llm_provider,
            similarity_threshold=self.config.hcc_similarity_threshold,
        )
        
        # TTS 通常是只读的，可以共享引用，或者也深拷贝
        # 为了安全起见，如果 TTS 有动态添加示例的需求，建议深拷贝
        # 这里假设 TTS 主要是静态的，或者 add_example 很少发生
        # 如果需要完全隔离: snapshot.tts = copy.deepcopy(self.tts)
        snapshot.tts = self.tts 
        
        return snapshot

    def inject_wisdom(
        self, 
        wisdom: str, 
        search_traces: Optional[List[str]] = None,
        promote_to_l2: bool = True
    ) -> None:
        """注入外部智慧 (通过 L1 -> L2 渐进式提升)
        
        设计理念:
            MCTS 的结果被视为"前人智慧"，需要经过 HCC 的自然层级过滤。
            这样可以确保错误的建议在提升过程中被纠正或过滤。
        
        流程:
            1. 将 MCTS 搜索轨迹记录到 L1 (原始经验)
            2. 直接创建 L2 Knowledge Unit (因为 MCTS 结果已是精炼的)
            3. 当任务完成时，L2 -> L3 自然提升
        
        Args:
            wisdom: 提炼的智慧文本
            search_traces: MCTS 搜索过程的轨迹 (可选)
            promote_to_l2: 是否立即提升到 L2 (默认 True)
        """
        if not self._task_description:
            print("[Memory] Warning: Injecting wisdom without task context")
            return
        
        print(f"[Memory] Recording MCTS experience to L1...")
        
        # Step 1: 记录搜索轨迹到 L1 (如果有)
        if search_traces:
            for i, trace in enumerate(search_traces):
                self.record_event(
                    f"[MCTS-Trace-{i+1}] {trace}", 
                    EventType.AGENT
                )
        
        # Step 2: 记录最终建议到 L1
        self.record_event(
            f"[MCTS-Insight] {wisdom}", 
            EventType.AGENT
        )
        
        # Step 3: 直接创建 L2 Knowledge Unit
        # MCTS 的输出已经是结构化的洞察，无需再次 LLM 压缩
        if promote_to_l2:
            print(f"[Memory] Promoting MCTS experience: L1 -> L2")
            from .hcc import KnowledgeUnit
            
            knowledge = KnowledgeUnit(
                phase_id=self.hcc.l1.current_phase,
                start_step=self._step_count - len(search_traces or []) - 1,
                end_step=self._step_count,
                summary=f"[MCTS] {wisdom}",
            )
            self.hcc.l2.add_knowledge(knowledge)
            
            # 同时更新 L1 的 phase 计数器
            self.hcc.l1._current_phase += 1
    
    def inject_and_finalize_wisdom(
        self, 
        wisdom: str, 
        search_traces: Optional[List[str]] = None
    ) -> None:
        """注入并立即提升到 L3 (用于紧急/高置信度场景)
        
        警告: 跳过 L2 积累阶段，直接写入 L3。
        仅在 MCTS 结果高度可信时使用。
        """
        self.inject_wisdom(wisdom, search_traces, promote_to_l2=True)
        
        # 立即触发 Task Promotion (L2 -> L3)
        print(f"[Memory] Finalizing wisdom: L2 -> L3")
        self.finish_task(final_result=f"MCTS Wisdom: {wisdom}")
    
    # ==================== TTS 相关方法 ====================
    
    def load_examples(self, path: Union[str, Path]) -> int:
        """加载示例轨迹
        
        Args:
            path: 文件或目录路径
            
        Returns:
            加载的轨迹数量
        """
        path = Path(path)
        if path.is_dir():
            return self.tts.load_from_directory(path)
        elif path.suffix == ".json":
            return self.tts.load_from_json(path)
        else:
            return self.tts.load_from_text(path)
    
    def add_example(self, trajectory: Trajectory) -> None:
        """添加单个示例轨迹"""
        self.tts.add(trajectory)
    
    def get_examples(
        self,
        query: Optional[str] = None,
        category: Optional[TrajectoryCategory] = None,
        k: Optional[int] = None,
    ) -> List[Trajectory]:
        """检索示例轨迹
        
        Args:
            query: 查询文本
            category: 限定类别
            k: 返回数量，默认使用 config.tts_max_examples
        """
        return self.tts.retrieve(
            query=query,
            category=category,
            k=k or self.config.tts_max_examples,
        )
    
    # ==================== HCC 相关方法 ====================
    
    def start_task(self, task_description: str, user_instruction: str = "") -> str:
        """开始新任务
        
        初始化 HCC 上下文，并返回包含先验智慧的初始上下文。
        
        Args:
            task_description: 任务描述
            user_instruction: 用户指令
            
        Returns:
            初始上下文文本 (包含预取的 L3 智慧)
        """
        self._task_description = task_description
        self._step_count = 0
        return self.migrator.initialize_context(task_description, user_instruction)
    
    def record_event(
        self,
        content: str,
        event_type: EventType = EventType.ENVIRONMENT,
    ) -> None:
        """记录事件到 L1 缓存
        
        Args:
            content: 事件内容
            event_type: 事件类型 (ENVIRONMENT 或 AGENT)
        """
        event = Event(
            step=self._step_count,
            event_type=event_type,
            content=content,
        )
        self.hcc.add_event(event)
        self._step_count += 1
        
        # 自动阶段提升检查
        if self._step_count > 0 and self._step_count % self.config.hcc_auto_promote_steps == 0:
            self._auto_promote_phase()
    
    def record_thought_action(
        self,
        thought: str,
        action: str,
        observation: str = "",
    ) -> None:
        """记录一个完整的 Thought-Action-Observation 循环
        
        便捷方法，将一个 ReAct 步骤记录为多个事件。
        """
        self.record_event(f"Thought: {thought}", EventType.AGENT)
        self.record_event(f"Action: {action}", EventType.AGENT)
        if observation:
            self.record_event(f"Observation: {observation}", EventType.ENVIRONMENT)
    
    def get_context(self) -> str:
        """获取当前上下文 (HCC 的 L1 + L2)"""
        return self.migrator.build_context()
    
    def promote_phase(self) -> None:
        """手动触发阶段提升 (L1 -> L2)"""
        self.migrator.promote_phase()
    
    def finish_task(self, final_result: str = "") -> None:
        """结束任务，触发任务级提升 (-> L3)
        
        Args:
            final_result: 最终结果文本
        """
        self.migrator.promote_task(
            task_descriptor=self._task_description,
            final_result=final_result,
        )
    
    def _auto_promote_phase(self) -> None:
        """自动阶段提升 (Rule-based)"""
        print(f"[Memory] Auto-promoting phase at step {self._step_count}")
        self.migrator.promote_phase()
    
    # ==================== Prompt 构建 ====================
    
    def build_prompt(
        self,
        task: str,
        system_prompt: str = "",
        include_examples: bool = True,
        include_context: bool = True,
        example_category: Optional[TrajectoryCategory] = None,
    ) -> str:
        """构建完整的 Prompt (三明治结构)
        
        结构:
            1. System Prompt
            2. TTS Examples (Few-Shot)
            3. HCC Context (Progress + Memory)
            4. Current Task
            
        Args:
            task: 当前任务/问题
            system_prompt: 系统提示词
            include_examples: 是否包含 TTS 示例
            include_context: 是否包含 HCC 上下文
            example_category: 限定示例类别
            
        Returns:
            组装好的完整 Prompt
        """
        parts = []
        
        # 1. System Prompt
        if system_prompt:
            parts.append(system_prompt)
        
        # 2. TTS Examples
        if include_examples and len(self.tts) > 0:
            examples = self.get_examples(
                query=task,
                category=example_category,
            )
            if examples:
                examples_text = self.tts.format_for_prompt(
                    examples,
                    max_tokens=self.config.tts_max_tokens,
                    header=self.config.examples_header,
                )
                parts.append(examples_text)
        
        # 3. HCC Context
        if include_context:
            context = self.get_context()
            if context:
                parts.append(f"{self.config.context_header}{context}")
        
        # 4. Current Task
        parts.append(f"## Current Task\n\n{task}")
        
        return "\n\n".join(parts)
    
    # ==================== 统计与调试 ====================
    
    def get_stats(self) -> Dict[str, Any]:
        """获取 Memory 统计信息"""
        return {
            "tts": self.tts.get_stats(),
            "hcc": self.hcc.get_stats(),
            "current_step": self._step_count,
            "current_task": self._task_description,
        }
    
    def reset(self) -> None:
        """重置 Memory 状态 (保留 TTS 和 L3)
        
        适用于开始新任务时清理 L1/L2 状态。
        """
        self.hcc = HierarchicalCognitiveCache(embedding_provider=self._embedder)
        self.migrator = ContextMigrator(
            hcc=self.hcc,
            similarity_threshold=self.config.hcc_similarity_threshold,
        )
        self._step_count = 0
        self._task_description = ""
    
    def __repr__(self) -> str:
        return (
            f"AgentMemory(tts={len(self.tts)} examples, "
            f"hcc_phase={self.hcc.l1.current_phase}, "
            f"steps={self._step_count})"
        )


# ==================== 便捷导出 ====================

__all__ = [
    # 主接口
    "AgentMemory",
    "MemoryConfig",
    # HCC 组件
    "HierarchicalCognitiveCache",
    "ContextMigrator",
    "Event",
    "EventType",
    # TTS 组件
    "TinyTrajectoryStore",
    "Trajectory",
    "TrajectoryStep",
    "TrajectoryCategory",
    # 嵌入
    "EmbeddingProvider",
    "SimpleEmbeddingProvider",
]
