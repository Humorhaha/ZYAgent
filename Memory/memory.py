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
from hcc import (
    HierarchicalCognitiveCache,
    ContextMigrator,
    Event,
    EventType,
    EmbeddingProvider,
    SimpleEmbeddingProvider,
)

# TTS 模块导入
from tts import (
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
