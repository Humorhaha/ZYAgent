"""
Context Migration - 上下文迁移模块

论文核心机制:
    Context Migration = (Context Prefetching, Context Hit, Context Promotion)
    
    1. Prefetching: 从L3预取相关先验智慧 Ω_τ
    2. Hit: 构建上下文时的缓存命中策略 Ψ_t(k)
    3. Promotion: P = (P1, P2)
       - P1: 阶段级提升，将执行轨迹压缩为知识
       - P2: 任务级提升，将知识蒸馏为智慧
"""

from typing import List, Optional, Protocol, Dict, Any
from dataclasses import dataclass


class LLMProvider(Protocol):
    """LLM提供者协议 - 用于知识/智慧的生成
    
    子类可实现不同的LLM后端:
    - OpenAI GPT-4
    - DeepSeek
    - 本地模型
    """
    
    def generate(self, prompt: str) -> str:
        """生成文本响应"""
        ...


@dataclass
class PromotionResult:
    """提升操作的结果"""
    summary: str           # 生成的摘要文本
    token_count: int       # 消耗的token数
    metadata: Dict[str, Any]


class MockLLMProvider:
    """Mock LLM提供者 - 用于测试
    
    生产环境应替换为真实的LLM后端
    """
    
    def generate(self, prompt: str) -> str:
        # 简单提取提示中的关键信息
        if "phase" in prompt.lower():
            return "[Mock] Phase summary: Key insights extracted from exploration traces."
        elif "task" in prompt.lower():
            return "[Mock] Task wisdom: Transferable strategies distilled from task execution."
        return "[Mock] Generated summary."


class ContextMigrator:
    """上下文迁移器 - 管理HCC中的信息流动
    
    核心职责:
    1. 初始化时预取先验智慧 (Prefetching)
    2. 构建上下文时实现缓存命中策略 (Hit)
    3. 阶段/任务完成时执行提升操作 (Promotion)
    
    使用示例:
        ```python
        from hcc import HierarchicalCognitiveCache
        
        hcc = HierarchicalCognitiveCache()
        migrator = ContextMigrator(hcc, llm_provider=my_llm)
        
        # 预取先验智慧
        initial_context = migrator.initialize_context(
            task_description="Image classification on plant leaves",
            user_instruction="Achieve high F1 score"
        )
        
        # 阶段完成时提升
        knowledge = migrator.promote_phase(traces=[...])
        
        # 任务完成时提升
        wisdom = migrator.promote_task(
            task_descriptor="...",
            final_code="...",
            final_result="..."
        )
        ```
    """
    
    # ============================================================
    # 阶段级提升的Prompt模板
    # 论文 A.3 节定义的提示词
    # ============================================================
    PHASE_PROMOTION_PROMPT = """You are a Kaggle Grandmaster summarizing exploration results.

Given the following execution traces from parallel exploration, extract key insights and learnings:

## Exploration Traces
{traces}

## Instructions
Summarize the key findings in a concise paragraph:
- What worked well and what didn't
- Critical insights about the data or model
- Recommendations for future exploration

Output only the summary paragraph, no extra formatting."""

    # ============================================================
    # 任务级提升的Prompt模板
    # ============================================================
    TASK_PROMOTION_PROMPT = """You are a Kaggle Grandmaster distilling transferable wisdom.

Given the following task completion information, extract reusable strategies:

## Task Description
{task_descriptor}

## Final Solution Code (key parts)
{final_code}

## Results
{final_result}

## Accumulated Knowledge
{knowledge_summary}

## Instructions
Distill the most valuable, transferable strategies that could help with similar future tasks.
Focus on:
- Model architecture choices that worked
- Data preprocessing techniques
- Training strategies and hyperparameters
- Common pitfalls to avoid

Output a concise wisdom paragraph, no extra formatting."""

    def __init__(
        self, 
        hcc: "HierarchicalCognitiveCache",
        llm_provider: Optional[LLMProvider] = None,
        similarity_threshold: float = 0.5
    ):
        """
        Args:
            hcc: 层次化认知缓存实例
            llm_provider: LLM提供者，用于生成摘要（默认使用Mock）
            similarity_threshold: L3检索的相似度阈值 δ
        """
        from .hcc import HierarchicalCognitiveCache
        self._hcc = hcc
        self._llm = llm_provider or MockLLMProvider()
        self._threshold = similarity_threshold
    
    @property
    def llm_provider(self) -> LLMProvider:
        """获取内部的 LLM Provider"""
        return self._llm
    
    def initialize_context(
        self, 
        task_description: str, 
        user_instruction: str = ""
    ) -> str:
        """初始化上下文 (Context Prefetching)
        
        论文公式:
            q = E(d_τ)
            Ω_τ = {w_n | (h_n, w_n) ∈ L3, cos(q, h_n) > δ}
            e_0 = concat(d_τ, u_user, Ω_τ)
            C_0 = g(E_0) = e_0
            
        Args:
            task_description: 任务描述 d_τ
            user_instruction: 用户指令 u_user
            
        Returns:
            初始上下文 C_0
        """
        # 1. 从L3预取相关先验智慧
        prior_wisdom = self._hcc.prefetch(
            task_description, 
            threshold=self._threshold
        )
        
        # 2. 构建初始上下文
        parts = []
        
        # 任务描述
        parts.append(f"## Task Description\n{task_description}")
        
        # 用户指令（如果有）
        if user_instruction:
            parts.append(f"## User Instructions\n{user_instruction}")
        
        # 先验智慧（如果有）
        if prior_wisdom:
            wisdom_text = "\n".join(f"- {w}" for w in prior_wisdom)
            parts.append(f"## Prior Wisdom (from similar tasks)\n{wisdom_text}")
        
        return "\n\n".join(parts)
    
    def build_context(self) -> str:
        """构建当前上下文 (Context Hit)
        
        论文公式:
            Ψ_t(k) = e_k                    if e_k ∈ L1(t)
                   = κ_{t_{r-1}+1:t_{r-1}}  if e_k ∉ L1(t), e_k ∈ L2(t)
                   = ∅                      otherwise
            
        委托给HCC的build_context方法
        """
        return self._hcc.build_context()
    
    def promote_phase(self, traces: Optional[List[str]] = None) -> PromotionResult:
        """阶段级提升 (Phase-level Promotion P1)
        
        论文公式:
            κ_p = P1({σ_{p,i,j}}_{(i,j)∈I_p})
            L2 ← L2 ∪ {κ_p}
            L1 ← L1 \ {执行轨迹}
            
        Args:
            traces: 可选的执行轨迹列表，如果不提供则从L1获取
            
        Returns:
            提升结果，包含生成的知识摘要
        """
        # 获取需要提升的轨迹
        if traces is None:
            events = self._hcc.l1.get_events_for_promotion()
            traces = [e.content for e in events]
        
        if not traces:
            return PromotionResult(
                summary="",
                token_count=0,
                metadata={"status": "no_traces"}
            )
        
        # 构建提示并调用LLM
        traces_text = "\n".join(f"[{i+1}] {t}" for i, t in enumerate(traces))
        prompt = self.PHASE_PROMOTION_PROMPT.format(traces=traces_text)
        
        summary = self._llm.generate(prompt)
        
        # 执行提升操作
        self._hcc.promote_phase(summary)
        
        return PromotionResult(
            summary=summary,
            token_count=len(prompt.split()) + len(summary.split()),  # 粗略估计
            metadata={"traces_count": len(traces)}
        )
    
    def promote_task(
        self,
        task_descriptor: str,
        final_code: str = "",
        final_result: str = ""
    ) -> PromotionResult:
        """任务级提升 (Task-level Promotion P2)
        
        论文公式:
            w_τ = P2(d_τ, L1(t_max), L2(t_max), h(E_{t_max}))
            L3 ← L3 ∪ {(E(d_τ), w_τ)}
            
        Args:
            task_descriptor: 任务描述 d_τ
            final_code: 最终解决方案代码
            final_result: 最终执行结果
            
        Returns:
            提升结果，包含生成的任务智慧
        """
        # 收集所有L2知识
        knowledge_units = self._hcc.l2.get_all_knowledge()
        knowledge_summary = "\n".join(k.summary for k in knowledge_units)
        
        # 构建提示并调用LLM
        prompt = self.TASK_PROMOTION_PROMPT.format(
            task_descriptor=task_descriptor,
            final_code=final_code[:2000] if final_code else "(not provided)",
            final_result=final_result[:500] if final_result else "(not provided)",
            knowledge_summary=knowledge_summary if knowledge_summary else "(none)"
        )
        
        wisdom = self._llm.generate(prompt)
        
        # 执行提升操作
        self._hcc.promote_task(task_descriptor, wisdom)
        
        return PromotionResult(
            summary=wisdom,
            token_count=len(prompt.split()) + len(wisdom.split()),
            metadata={
                "knowledge_units_used": len(knowledge_units),
                "has_code": bool(final_code),
                "has_result": bool(final_result)
            }
        )
