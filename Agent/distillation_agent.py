"""
Distillation Agent - 从 TTS 检索相关示例

根据用户查询，从 Tiny Trajectory Store 中检索最相关的 Few-Shot 示例，
用于增强 ReAct Agent 的上下文。
"""

from typing import Optional

from .base import BaseAgent, AgentContext, AgentResult, AgentStatus, LLMProvider

# TTS 导入
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from Memory.tts import TinyTrajectoryStore, TrajectoryCategory


class DistillationAgent(BaseAgent):
    """蒸馏 Agent
    
    职责:
        - 根据用户查询检索 TTS 中的相关示例
        - 格式化示例用于 Prompt 注入
    
    在 ReactXen 架构中的位置:
        - 流程入口，在 ReAct Agent 之前执行
        - 输出的 examples 会被注入到 ReAct Agent 的 Prompt 中
    """
    
    def __init__(
        self, 
        llm_provider: LLMProvider, 
        tts: TinyTrajectoryStore,
        max_examples: int = 2,
        max_tokens: int = 1500,
    ):
        """
        Args:
            llm_provider: LLM 提供者
            tts: TinyTrajectoryStore 实例
            max_examples: 最大检索示例数
            max_tokens: 示例的最大 Token 数
        """
        super().__init__(llm_provider, name="DistillationAgent")
        self.tts = tts
        self.max_examples = max_examples
        self.max_tokens = max_tokens
    
    def run(self, context: AgentContext) -> AgentResult:
        """从 TTS 检索相关示例
        
        Args:
            context: 包含 query 的上下文
            
        Returns:
            包含格式化 examples 的结果
        """
        self._log(f"Retrieving examples for query: {context.query[:50]}...")
        
        # 1. 推断任务类别 (可选：使用 LLM 进行更精确的分类)
        category = self._infer_category(context.query)
        self._log(f"Inferred category: {category.value}")
        
        # 2. 检索相关轨迹
        trajectories = self.tts.retrieve(
            query=context.query,
            category=category,
            k=self.max_examples,
        )
        
        if not trajectories:
            self._log("No relevant examples found in TTS")
            return AgentResult(
                status=AgentStatus.SUCCESS,
                output="",
                reasoning="No relevant examples found in TTS",
            )
        
        # 3. 格式化输出
        examples_text = self.tts.format_for_prompt(
            trajectories=trajectories,
            max_tokens=self.max_tokens,
            header="## Reference Examples\n\nLearn from these examples:\n\n",
        )
        
        self._log(f"Retrieved {len(trajectories)} examples")
        
        return AgentResult(
            status=AgentStatus.SUCCESS,
            output=examples_text,
            reasoning=f"Retrieved {len(trajectories)} examples from category '{category.value}'",
            metadata={"num_examples": len(trajectories), "category": category.value},
        )
    
    def _infer_category(self, query: str) -> TrajectoryCategory:
        """推断查询的任务类别
        
        简单的关键词匹配实现。
        生产环境可以使用 LLM 进行更精确的分类。
        """
        query_lower = query.lower()
        
        category_keywords = {
            TrajectoryCategory.DATA_SCIENCE: ["train", "model", "dataset", "feature", "ml", "data"],
            TrajectoryCategory.CODE_GENERATION: ["code", "function", "implement", "write", "program"],
            TrajectoryCategory.REASONING: ["why", "how", "explain", "reason", "logic", "math"],
            TrajectoryCategory.QA: ["what is", "who is", "when", "where", "question"],
            TrajectoryCategory.TOOL_USE: ["search", "calculate", "api", "tool", "call"],
        }
        
        for category, keywords in category_keywords.items():
            if any(kw in query_lower for kw in keywords):
                return category
        
        return TrajectoryCategory.GENERAL
