"""
Agent 基类与数据结构

定义所有 Agent 共用的基类、上下文和结果类型。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol


class LLMProvider(Protocol):
    """LLM 提供者协议"""
    def generate(self, prompt: str) -> str:
        """生成文本响应"""
        ...


class AgentStatus(Enum):
    """Agent 执行状态"""
    SUCCESS = "success"           # 成功完成
    FAILED = "failed"             # 失败
    NOT_FINISHED = "not_finished" # 未完成 (需要更多步骤)
    NEEDS_REVIEW = "needs_review" # 需要评审
    NEEDS_REFLECT = "needs_reflect" # 需要反思


@dataclass
class AgentContext:
    """Agent 执行上下文
    
    在 Agent 之间传递的共享状态。
    
    Attributes:
        query: 原始用户查询
        examples: TTS 检索的示例 (由 DistillationAgent 填充)
        scratchpad: ReAct 思维链记录
        reflections: 历史反思列表
        current_answer: 当前答案
        review_feedback: Review Agent 的反馈
        iteration: 当前迭代次数
        metadata: 附加元数据
    """
    query: str
    examples: str = ""
    scratchpad: str = ""
    reflections: List[str] = field(default_factory=list)
    current_answer: str = ""
    review_feedback: str = ""
    iteration: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_reflections_str(self) -> str:
        """获取格式化的反思字符串"""
        if not self.reflections:
            return ""
        
        lines = ["Previous Reflections:"]
        for i, ref in enumerate(self.reflections, 1):
            lines.append(f"{i}. {ref}")
        return "\n".join(lines)


@dataclass
class AgentResult:
    """Agent 执行结果
    
    Attributes:
        status: 执行状态
        output: 主要输出内容
        reasoning: 推理过程说明
        suggestions: 改进建议 (主要由 ReviewAgent 使用)
        metadata: 附加元数据
    """
    status: AgentStatus
    output: str = ""
    reasoning: str = ""
    suggestions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_success(self) -> bool:
        return self.status == AgentStatus.SUCCESS
    
    @property
    def needs_reflect(self) -> bool:
        return self.status in (AgentStatus.FAILED, AgentStatus.NOT_FINISHED, AgentStatus.NEEDS_REFLECT)


class BaseAgent(ABC):
    """Agent 基类
    
    所有 Agent 必须继承此类并实现 run 方法。
    
    Attributes:
        name: Agent 名称
        llm: LLM 提供者
    """
    
    def __init__(self, llm_provider: LLMProvider, name: str = "BaseAgent"):
        """
        Args:
            llm_provider: LLM 提供者实例
            name: Agent 名称，用于日志
        """
        self.llm = llm_provider
        self.name = name
    
    @abstractmethod
    def run(self, context: AgentContext) -> AgentResult:
        """执行 Agent 主逻辑
        
        Args:
            context: 执行上下文，包含查询和历史状态
            
        Returns:
            执行结果
        """
        pass
    
    def _log(self, message: str) -> None:
        """打印日志"""
        print(f"[{self.name}] {message}")
    
    def _call_llm(self, prompt: str) -> str:
        """调用 LLM"""
        return self.llm.generate(prompt)


# =============================================================================
# Mock LLM Provider (用于测试)
# =============================================================================

class MockLLMProvider:
    """Mock LLM 提供者 (用于测试和演示)"""
    
    def __init__(self, responses: Optional[Dict[str, str]] = None):
        """
        Args:
            responses: 预定义的响应字典 (prompt 关键词 -> 响应)
        """
        self.responses = responses or {}
        self.call_count = 0
        self.last_prompt = ""
    
    def generate(self, prompt: str) -> str:
        self.call_count += 1
        self.last_prompt = prompt
        
        # 检查预定义响应
        for key, response in self.responses.items():
            if key.lower() in prompt.lower():
                return response
        
        # 默认响应
        if "thought" in prompt.lower():
            return "I need to analyze this step by step."
        elif "action" in prompt.lower():
            return "Action: Search\nAction Input: relevant query"
        elif "review" in prompt.lower():
            return "Status: Accomplished\nReasoning: Task completed successfully."
        elif "reflect" in prompt.lower():
            return "I should try a different approach next time."
        else:
            return "This is a mock response."
