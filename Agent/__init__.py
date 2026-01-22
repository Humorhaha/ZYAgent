"""
Agent 模块 - ReactXen 多 Agent 系统

包含以下组件:
- BaseAgent: Agent 基类
- DistillationAgent: 从 TTS 检索示例
- ReActAgent: 执行 Thought-Action-Observation 循环
- ReflectAgent: 分析失败原因 (含 Self-Ask)
- ReviewAgent: 评估任务质量
- VerificationAgent: 验证最终结果
- ReactXenOrchestrator: 主编排器
- LLM Providers: OpenAI, Anthropic, DeepSeek
"""

from .base import BaseAgent, AgentContext, AgentResult, AgentStatus
from .distillation_agent import DistillationAgent
from .react_agent import ReActAgent, Tool
from .reflect_agent import ReflectAgent
from .review_agent import ReviewAgent
from .verification_agent import VerificationAgent
from .orchestrator import ReactXenOrchestrator
from .llm_providers import OpenAIProvider, AnthropicProvider, DeepSeekProvider

__all__ = [
    # Base
    "BaseAgent",
    "AgentContext",
    "AgentResult",
    "AgentStatus",
    "Tool",
    # Agents
    "DistillationAgent",
    "ReActAgent",
    "ReflectAgent",
    "ReviewAgent",
    "VerificationAgent",
    # Orchestrator
    "ReactXenOrchestrator",
    # LLM Providers
    "OpenAIProvider",
    "AnthropicProvider",
    "DeepSeekProvider",
]
