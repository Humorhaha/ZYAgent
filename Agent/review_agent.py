"""
Review Agent - 评审 Agent

在 ReAct Agent 标记任务完成后，独立评估结果质量，
决定是接受还是需要反思。
"""

import re
from typing import Optional

from .base import BaseAgent, AgentContext, AgentResult, AgentStatus, LLMProvider


class ReviewAgent(BaseAgent):
    """评审 Agent
    
    职责:
        - 独立评估 ReAct Agent 的执行过程和答案
        - 提供结构化反馈 (Status, Reasoning, Suggestions)
        - 决定任务是否真正完成
    
    在 ReactXen 架构中的位置:
        - 在 ReAct Agent 认为完成后执行
        - 输出决定流程走向：
            - Accomplished -> Verification Agent
            - Not/Partially Accomplished -> Reflect Agent
    """
    
    REVIEW_PROMPT = """You are a strict quality reviewer. Your job is to evaluate whether a task has been accomplished correctly.

## Original Task
{query}

## Agent's Execution Process (Scratchpad)
{scratchpad}

## Agent's Final Answer
{answer}

## Review Criteria
1. Does the answer directly address the original task?
2. Is the reasoning process logical and complete?
3. Are there any factual errors or unsupported claims?
4. Is the answer clear and helpful?

## Instructions
Evaluate the above and respond in the following EXACT format:

Status: [Accomplished | Not Accomplished | Partially Accomplished]
Reasoning: [Your detailed reasoning for the status]
Suggestions: [If not accomplished, specific suggestions for improvement. If accomplished, write "None"]
"""
    
    def __init__(self, llm_provider: LLMProvider):
        super().__init__(llm_provider, name="ReviewAgent")
    
    def run(self, context: AgentContext) -> AgentResult:
        """评估任务完成质量
        
        Args:
            context: 执行上下文
            
        Returns:
            评审结果
        """
        self._log(f"Reviewing task output: {context.current_answer[:50]}...")
        
        # 1. 构建 Review Prompt
        prompt = self.REVIEW_PROMPT.format(
            query=context.query,
            scratchpad=context.scratchpad or "No scratchpad available.",
            answer=context.current_answer or "No answer provided.",
        )
        
        # 2. 调用 LLM 进行评审
        response = self._call_llm(prompt)
        
        # 3. 解析响应
        status, reasoning, suggestions = self._parse_review(response)
        
        self._log(f"Review Status: {status}")
        self._log(f"Reasoning: {reasoning[:100]}...")
        
        # 4. 更新上下文
        context.review_feedback = f"Status: {status}\nReasoning: {reasoning}\nSuggestions: {suggestions}"
        
        # 5. 确定结果状态
        if "accomplished" in status.lower() and "not" not in status.lower():
            result_status = AgentStatus.SUCCESS
        elif "partially" in status.lower():
            result_status = AgentStatus.NEEDS_REFLECT
        else:
            result_status = AgentStatus.NEEDS_REFLECT
        
        return AgentResult(
            status=result_status,
            output=status,
            reasoning=reasoning,
            suggestions=[s.strip() for s in suggestions.split(",") if s.strip()],
            metadata={"raw_response": response},
        )
    
    def _parse_review(self, response: str) -> tuple:
        """解析评审响应
        
        Returns:
            (status, reasoning, suggestions)
        """
        status = "Not Accomplished"
        reasoning = "Could not parse review response."
        suggestions = ""
        
        # 提取 Status
        status_match = re.search(r"Status[:\s]*([^\n]+)", response, re.IGNORECASE)
        if status_match:
            status = status_match.group(1).strip()
        
        # 提取 Reasoning
        reasoning_match = re.search(r"Reasoning[:\s]*([^\n]+(?:\n(?!Suggestions)[^\n]+)*)", response, re.IGNORECASE)
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()
        
        # 提取 Suggestions
        suggestions_match = re.search(r"Suggestions[:\s]*(.+?)(?=\n\n|$)", response, re.IGNORECASE | re.DOTALL)
        if suggestions_match:
            suggestions = suggestions_match.group(1).strip()
        
        return status, reasoning, suggestions
