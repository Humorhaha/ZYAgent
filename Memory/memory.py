"""
Memory 模块统一接口

将 HCC (Hierarchical Cognitive Caching) 和 TTS (Tiny Trajectory Store) 
整合为统一的 Memory 系统，对外提供简洁的 API。

设计理念 (pipeline.tex):
    - TTS (Tiny Trajectory Store): 可读写的轨迹缓存
        - 写入: Reflect Agent 写入失败/成功的轨迹
        - 读取: ReAct Agent 获取压缩后的 Markov 轨迹
        - 压缩: 每次写入时 LLM 将新轨迹与现有摘要合并，始终只保留一条 Markov 轨迹
    - HCC (Hierarchical Cognitive Cache): 三层知识缓存
        - L1: 短期工作记忆 (Evolving Experience)
        - L2: 中期策略记忆 (Refined Knowledge)
        - L3: 长期持久记忆 (Prior Wisdom)

使用示例:
    ```python
    from Memory.memory import AgentMemory
    
    memory = AgentMemory()
    memory.start_task("Train a classifier")
    memory.record_thought_action(thought="...", action="...", observation="...")
    compressed = memory.get_tts_buffer()  # 获取压缩后的 Markov 轨迹
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
    
    # Markov 约束: 只看最近 N 轮的上下文
    markov_window: int = 1
    
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
        
        # LLM 实例 (用于 TTS 压缩和知识提炼)
        self._llm_provider = llm_provider
        self._llm = None  # 延迟初始化
        
        # Hot Start: HCC Wisdom 缓存 (只在 start_task 时提取一次)
        self._hot_start_wisdom: str = ""
        self._wisdom_fetched: bool = False
        
        # TTS Buffer: 压缩后的轨迹摘要 (每个 iteration 更新)
        # 符合 Markov 性质：使用 LLM 压缩，只保留关键信息
        self._tts_buffer_summary: str = ""
    
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
    
    def start_task(
        self, 
        task_description: str, 
        user_instruction: str = "",
        fetch_wisdom: bool = True,
    ) -> str:
        """开始新任务 (Hot Start)
        
        架构 (pipeline.tex):
            - Hot Start: 从 HCC L2/L3 提取 Wisdom (只执行一次)
            - TTS Buffer: 清空，准备记录新任务的执行轨迹
        
        Args:
            task_description: 任务描述
            user_instruction: 用户指令
            fetch_wisdom: 是否执行 Hot Start (从 HCC 获取 Wisdom)
            
        Returns:
            初始上下文文本 (包含预取的 HCC Wisdom)
        """
        self._task_description = task_description
        self._step_count = 0
        
        # 重置 TTS Buffer 摘要 (每个任务独立)
        self._tts_buffer_summary = ""
        
        # Hot Start: 从 HCC L2/L3 提取 Wisdom (只执行一次)
        if fetch_wisdom and not self._wisdom_fetched:
            self._hot_start_wisdom = self.get_wisdom_for_hot_start(
                query=task_description,
                k=3,
            )
            self._wisdom_fetched = True
            print(f"[Memory] Hot Start: Loaded wisdom from HCC")
        
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
        
        架构 (pipeline.tex):
            - 记录到 HCC L1 (长期存储)
            - 使用 LLM 压缩 TTS Buffer，保留关键信息 (Markov 约束)
        """
        # 1. 记录到 HCC L1
        self.record_event(f"Thought: {thought}", EventType.AGENT)
        self.record_event(f"Action: {action}", EventType.AGENT)
        if observation:
            self.record_event(f"Observation: {observation}", EventType.ENVIRONMENT)
        
        # 2. 构建当前轨迹
        current_trajectory = f"Thought: {thought}\nAction: {action}"
        if observation:
            current_trajectory += f"\nObservation: {observation}"
        
        # 3. 使用 LLM 压缩 (合并历史摘要与当前轨迹)
        self._tts_buffer_summary = self._compress_tts_buffer(
            current_trajectory=current_trajectory,
            previous_summary=self._tts_buffer_summary,
        )
    
    def _compress_tts_buffer(
        self,
        current_trajectory: str,
        previous_summary: str,
    ) -> str:
        """使用 LLM 压缩 TTS Buffer
        
        将当前轨迹与历史摘要合并，生成新的压缩摘要。
        
        Args:
            current_trajectory: 当前轮的执行轨迹
            previous_summary: 之前的压缩摘要
            
        Returns:
            新的压缩摘要
        """
        # 延迟初始化 LLM
        if self._llm is None:
            try:
                from LLM.llm import LLM
                if self._llm_provider:
                    self._llm = LLM(provider=self._llm_provider)
                else:
                    # 无 LLM 时使用简单拼接
                    self._llm = LLM(mock_mode=True)
            except ImportError:
                # 回退: 无 LLM 压缩，直接拼接
                if previous_summary:
                    return f"{previous_summary}\n---\n{current_trajectory}"
                return current_trajectory
        
        # 使用 LLM 压缩
        if self._llm.is_mock:
            # Mock 模式: 简单保留最新轨迹 + 历史摘要前 200 字符
            if previous_summary:
                truncated_prev = previous_summary[:200] + "..." if len(previous_summary) > 200 else previous_summary
                return f"[历史] {truncated_prev}\n[最新] {current_trajectory}"
            return current_trajectory
        
        # 真正的 LLM 压缩
        return self._llm.compress_trajectory(
            current_trajectory=current_trajectory,
            previous_summary=previous_summary,
        )
    
    def get_tts_buffer(self) -> str:
        """获取 TTS Buffer 内容 (压缩后的轨迹摘要)
        
        架构 (pipeline.tex):
            TTS Buffer 使用 LLM 压缩轨迹，只保留关键信息。
            每个 iteration 的 ReAct Agent 从这里获取上下文。
            
        Returns:
            压缩后的 TTS Buffer 摘要
        """
        return self._tts_buffer_summary
    
    def write_to_tts(
        self,
        trajectory: str,
        source: str = "reflect",
    ) -> None:
        """写入轨迹到 TTS Buffer (供 Reflect Agent 使用)
        
        架构 (pipeline.tex):
            - Reflect Agent 将失败/成功的轨迹写入 TTS
            - TTS 使用 LLM 压缩，提炼为符合 Markov 性质的摘要
        
        Args:
            trajectory: 要写入的轨迹内容
            source: 来源标记 ("reflect", "success", "failure")
        """
        print(f"[Memory] Writing to TTS from {source}")
        
        # 使用 LLM 压缩：将新轨迹与现有摘要合并
        tagged_trajectory = f"[{source.upper()}] {trajectory}"
        self._tts_buffer_summary = self._compress_tts_buffer(
            current_trajectory=tagged_trajectory,
            previous_summary=self._tts_buffer_summary,
        )
    
    def get_hot_start_wisdom(self) -> str:
        """获取 Hot Start Wisdom (从 HCC 预取的)
        
        Returns:
            Hot Start 时预取的 Wisdom 文本
        """
        return self._hot_start_wisdom
    
    def get_context(self, window: Optional[int] = None) -> str:
        """获取当前上下文 (Markov 约束)
        
        Args:
            window: 回看的轮数，None 使用配置的 markov_window
                    window=1 表示只看最近一轮的事件
        
        Returns:
            受 Markov 约束的上下文文本
        """
        w = window if window is not None else self.config.markov_window
        
        if w <= 0:
            # 无限制，返回完整上下文
            return self.migrator.build_context()
        
        # 只返回最近 w 个 phase 的事件
        return self.migrator.build_context(max_phases=w)
    
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
    
    # ==================== 轨迹管理 (架构: pipeline.tex) ====================
    
    def record_success_trajectory(
        self,
        query: str,
        trajectory: str,
        final_answer: str,
    ) -> None:
        """记录成功轨迹到 HCC L1
        
        架构位置: Success → HCC Level 1 (成功轨迹)
        
        Args:
            query: 原始查询
            trajectory: 完整执行轨迹
            final_answer: 最终答案
        """
        print(f"[Memory] Recording success trajectory to HCC L1")
        
        # 记录到 L1
        self.record_event(
            f"[SUCCESS] Task: {query[:50]}...\nAnswer: {final_answer[:100]}...",
            EventType.AGENT
        )
        
        # 创建 Wisdom 并可选提升到 L2
        self.inject_wisdom(
            wisdom=f"Successfully solved: {query[:50]}...\nApproach: {final_answer[:100]}...",
            search_traces=[trajectory],
            promote_to_l2=True,
        )
    
    def record_failure_trajectory(
        self,
        query: str,
        trajectory: str,
        failure_reason: str,
        reflect_output: str = "",
    ) -> None:
        """记录失败轨迹到 HCC L1
        
        架构位置: Failure → HCC Level 1 (失败轨迹)
        
        Args:
            query: 原始查询
            trajectory: 完整执行轨迹
            failure_reason: 失败原因
            reflect_output: Reflect Agent 的输出
        """
        print(f"[Memory] Recording failure trajectory to HCC L1")
        
        # 记录到 L1 (不立即提升)
        self.record_event(
            f"[FAILURE] Task: {query[:50]}...\nReason: {failure_reason}",
            EventType.AGENT
        )
        
        if reflect_output:
            self.record_event(
                f"[REFLECT] {reflect_output[:200]}...",
                EventType.AGENT
            )
    
    def get_wisdom_for_hot_start(
        self,
        query: str,
        k: int = 3,
    ) -> str:
        """从 HCC L3 获取相关 Wisdom (Hot Start)
        
        使用 Cosine 相似度检索最相关的 Wisdom (Embedding-Value 结构)。
        
        架构位置: HCC Level 3 (Wisdom) → Distillation Agent
        
        Args:
            query: 当前任务查询
            k: 返回的 Wisdom 数量
            
        Returns:
            格式化的 Wisdom 文本
        """
        print(f"[Memory] Hot Start: Fetching wisdom for query: {query[:50]}...")
        
        # 使用 Cosine 相似度检索 Top-K Wisdom
        # retrieve_top_k 返回 (wisdom_text, similarity_score) 列表
        results = self.hcc.l3.retrieve_top_k(query, k=k, min_threshold=0.2)
        
        if not results:
            return ""
        
        # 格式化输出 (显示相似度分数)
        lines = ["## Prior Wisdom (from HCC L3 by Cosine Similarity)\n"]
        for i, (wisdom, score) in enumerate(results, 1):
            lines.append(f"{i}. [score={score:.2f}] {wisdom}")
        
        return "\n".join(lines)
    
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
