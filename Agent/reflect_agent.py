"""
Reflect Agent - 反思 Agent (含 Self-Ask 机制)

当任务失败或未完成时，分析原因并生成反思，
用于指导下一轮 ReAct 执行。
"""

from .base import BaseAgent, AgentContext, AgentResult, AgentStatus, LLMProvider


class ReflectAgent(BaseAgent):
    """反思 Agent
    
    职责:
        - Self-Ask: "Why is task not accomplished? What went wrong?"
        - 生成 Short Term Reflection
        - 输出用于下一轮 ReAct 的改进建议
    
    在 ReactXen 架构中的位置:
        - 当 ReAct Agent 未完成任务时触发
        - 当 Review Agent 判定结果不合格时触发
        - 输出的 reflection 会被注入到下一轮 ReAct Agent 的 Prompt 中
    """
    
    # Self-Ask Prompt
    SELF_ASK_PROMPT = """You are an expert at analyzing failed attempts and generating actionable reflections.

## Task
{query}

## Previous Attempt (Scratchpad)
{scratchpad}

## Review Feedback (if any)
{review_feedback}

## Self-Ask Questions
1. Why is the task not accomplished?
2. What went wrong in the previous attempt?
3. What specific mistakes were made?
4. What should be done differently next time?

## Instructions
Based on the above, generate a SHORT and ACTIONABLE reflection.
Focus on:
- Identify the ROOT CAUSE of failure
- Provide SPECIFIC suggestions for improvement
- Keep the reflection CONCISE (2-3 sentences max)

Your Reflection:
"""
    
    def __init__(self, llm_provider: LLMProvider):
        super().__init__(llm_provider, name="ReflectAgent")
    
    def run(self, context: AgentContext) -> AgentResult:
        """生成反思
        
        Args:
            context: 执行上下文，包含 scratchpad 和 review_feedback
            
        Returns:
            包含反思内容的结果
        """
        self._log("Self-Ask: Why is task not accomplished? What went wrong?")
        
        # 1. 构建 Self-Ask Prompt
        prompt = self.SELF_ASK_PROMPT.format(
            query=context.query,
            scratchpad=context.scratchpad or "No scratchpad available.",
            review_feedback=context.review_feedback or "No review feedback.",
        )
        
        # 2. 调用 LLM 生成反思
        reflection = self._call_llm(prompt)
        reflection = reflection.strip()
        
        self._log(f"Reflection generated: {reflection[:100]}...")
        
        # 3. 将反思添加到上下文
        context.reflections.append(reflection)
        
        return AgentResult(
            status=AgentStatus.SUCCESS,
            output=reflection,
            reasoning="Generated Short Term Reflection based on Self-Ask analysis",
            metadata={"iteration": context.iteration},
        )
