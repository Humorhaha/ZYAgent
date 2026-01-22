"""
Verification Agent - 验证 Agent

在 Review Agent 判定任务完成后，进行最终验证，
确保答案的正确性和格式。
"""

from .base import BaseAgent, AgentContext, AgentResult, AgentStatus, LLMProvider


class VerificationAgent(BaseAgent):
    """验证 Agent
    
    职责:
        - 验证最终答案的格式和正确性
        - 可选：运行测试用例
        - 提供最终确认
    
    在 ReactXen 架构中的位置:
        - 流程的最后一环
        - 只有通过 Review 后才会执行
        - 验证通过则输出最终结果
        - 验证失败且迭代次数 >= M 则返回失败
    """
    
    VERIFICATION_PROMPT = """You are a verification agent. Your job is to perform a final check on the answer.

## Original Task
{query}

## Final Answer
{answer}

## Verification Checklist
1. Is the answer formatted correctly?
2. Does the answer contain all required information?
3. Is the answer factually consistent with the reasoning?
4. Would this answer be helpful to the user?

## Instructions
Respond with:
Verified: [Yes | No]
Reason: [Brief explanation]
"""
    
    def __init__(self, llm_provider: LLMProvider):
        super().__init__(llm_provider, name="VerificationAgent")
    
    def run(self, context: AgentContext) -> AgentResult:
        """验证最终结果
        
        Args:
            context: 执行上下文
            
        Returns:
            验证结果
        """
        self._log(f"Verifying final answer: {context.current_answer[:50]}...")
        
        # 1. 构建验证 Prompt
        prompt = self.VERIFICATION_PROMPT.format(
            query=context.query,
            answer=context.current_answer or "No answer provided.",
        )
        
        # 2. 调用 LLM 验证
        response = self._call_llm(prompt)
        
        # 3. 解析响应
        verified = "yes" in response.lower() and "verified: yes" in response.lower()
        
        # 提取原因
        reason = "Verification completed."
        if "reason:" in response.lower():
            reason = response.split("Reason:")[-1].strip()
        
        self._log(f"Verified: {verified}")
        
        if verified:
            return AgentResult(
                status=AgentStatus.SUCCESS,
                output=context.current_answer,
                reasoning=reason,
                metadata={"verified": True},
            )
        else:
            return AgentResult(
                status=AgentStatus.FAILED,
                output="",
                reasoning=reason,
                metadata={"verified": False},
            )
