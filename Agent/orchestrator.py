"""
ReactXen Orchestrator - 主编排器

按照 ReactXen 架构图编排所有 Agent：
1. Distillation Agent: 从 TTS 获取示例
2. ReAct Agent: 执行任务
3. Review Agent: 评估结果
4. Reflect Agent: 分析失败 (含 Self-Ask)
5. Verification Agent: 最终验证

流程遵循双循环架构，支持最大反思迭代次数。
"""

import sys
from pathlib import Path
from typing import List, Optional, Dict

# 确保可以导入本地模块
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from Agent.base import AgentContext, AgentResult, AgentStatus, LLMProvider, MockLLMProvider
from Agent.distillation_agent import DistillationAgent
from Agent.react_agent import ReActAgent, Tool
from Agent.reflect_agent import ReflectAgent
from Agent.review_agent import ReviewAgent
from Agent.verification_agent import VerificationAgent
from Memory.tts import TinyTrajectoryStore


class ReactXenOrchestrator:
    """ReactXen 主编排器
    
    按架构图编排 Agent 执行流程：
    
    ```
    Query -> Distillation -> ReAct -> [Not Finished?] -> Reflect -> ReAct (loop)
                                   -> [Finished?] -> Review -> [Accomplished?] -> Verification -> Output
                                                            -> [Not Accomplished?] -> Reflect -> ReAct (loop)
    ```
    
    最大反思迭代次数后若仍未成功，返回失败。
    """
    
    def __init__(
        self,
        llm_provider: LLMProvider,
        tts: Optional[TinyTrajectoryStore] = None,
        tools: Optional[List[Tool]] = None,
        max_reflect_iterations: int = 3,
        max_react_steps: int = 10,
    ):
        """
        Args:
            llm_provider: LLM 提供者
            tts: TinyTrajectoryStore 实例 (可选)
            tools: ReAct Agent 可用的工具列表
            max_reflect_iterations: 最大反思迭代次数 (架构图中的 M)
            max_react_steps: ReAct Agent 的最大步数
        """
        self.llm = llm_provider
        self.tts = tts or TinyTrajectoryStore()
        self.max_iterations = max_reflect_iterations
        
        # 初始化所有 Agent
        self.distillation_agent = DistillationAgent(llm_provider, self.tts)
        self.react_agent = ReActAgent(llm_provider, tools or [], max_react_steps)
        self.reflect_agent = ReflectAgent(llm_provider)
        self.review_agent = ReviewAgent(llm_provider)
        self.verification_agent = VerificationAgent(llm_provider)
        
        self._log("Orchestrator initialized with all agents")
    
    def run(self, query: str) -> Dict:
        """执行完整的 ReactXen 流程
        
        Args:
            query: 用户查询
            
        Returns:
            结果字典，包含 success, answer, iterations, trace
        """
        self._log(f"Starting ReactXen for query: {query[:50]}...")
        
        # 1. 初始化上下文
        context = AgentContext(query=query)
        trace = []  # 记录执行轨迹
        
        # 2. Distillation Agent - 获取 TTS 示例
        self._log("=== Phase 1: Distillation ===")
        distill_result = self.distillation_agent.run(context)
        context.examples = distill_result.output
        trace.append(("Distillation", distill_result))
        
        # 3. 主循环 (最多 max_iterations 次)
        for iteration in range(1, self.max_iterations + 1):
            context.iteration = iteration
            self._log(f"=== Iteration {iteration}/{self.max_iterations} ===")
            
            # 3.1 ReAct Agent - 执行任务
            self._log("Phase 2: ReAct Execution")
            react_result = self.react_agent.run(context)
            trace.append((f"ReAct (iter {iteration})", react_result))
            
            # 3.2 检查 ReAct 结果
            if react_result.status == AgentStatus.NOT_FINISHED:
                # 未完成 -> Reflect -> 继续
                self._log("ReAct not finished, triggering Reflect")
                reflect_result = self.reflect_agent.run(context)
                trace.append((f"Reflect (iter {iteration})", reflect_result))
                continue
            
            # 3.3 ReAct 认为完成 -> Review Agent 评估
            self._log("Phase 3: Review")
            review_result = self.review_agent.run(context)
            trace.append((f"Review (iter {iteration})", review_result))
            
            # 3.4 检查 Review 结果
            if review_result.is_success:
                # Review 通过 -> Verification
                self._log("Phase 4: Verification")
                verify_result = self.verification_agent.run(context)
                trace.append((f"Verification (iter {iteration})", verify_result))
                
                if verify_result.is_success:
                    # 成功！
                    self._log(f"SUCCESS! Final answer: {context.current_answer[:100]}...")
                    return {
                        "success": True,
                        "answer": context.current_answer,
                        "iterations": iteration,
                        "trace": trace,
                    }
                else:
                    # Verification 失败
                    self._log("Verification failed, triggering Reflect")
            else:
                # Review 不通过
                self._log("Review not passed, triggering Reflect")
            
            # 3.5 需要反思
            reflect_result = self.reflect_agent.run(context)
            trace.append((f"Reflect (iter {iteration})", reflect_result))
        
        # 4. 达到最大迭代次数
        self._log(f"FAILED: Max iterations ({self.max_iterations}) reached")
        return {
            "success": False,
            "answer": context.current_answer or "Failed to complete task",
            "iterations": self.max_iterations,
            "trace": trace,
        }
    
    def _log(self, message: str) -> None:
        """打印日志"""
        print(f"[Orchestrator] {message}")


# =============================================================================
# Demo / Sandbox Test
# =============================================================================

def create_mock_tools() -> List[Tool]:
    """创建用于测试的 Mock 工具"""
    
    def search(query: str) -> str:
        return f"Search results for '{query}': Found relevant information about the topic."
    
    def calculate(expression: str) -> str:
        try:
            result = eval(expression)
            return f"Calculation result: {result}"
        except Exception as e:
            return f"Calculation error: {e}"
    
    return [
        Tool("Search", "Search for information. Input: search query", search),
        Tool("Calculate", "Perform calculations. Input: mathematical expression", calculate),
    ]


def run_sandbox_demo():
    """运行沙盒测试"""
    print("=" * 60)
    print("ReactXen Orchestrator - Sandbox Demo")
    print("=" * 60)
    
    # 1. 创建 Mock LLM
    mock_responses = {
        "thought": "I need to search for this information first.",
        "action": "Action: Finish\nAction Input: The answer is 42, based on my analysis.",
        "review": "Status: Accomplished\nReasoning: The task was completed correctly with proper reasoning.\nSuggestions: None",
        "reflect": "I should have been more thorough in my search.",
        "verification": "Verified: Yes\nReason: The answer is correct and well-formatted.",
    }
    llm = MockLLMProvider(mock_responses)
    
    # 2. 创建 TTS (空的，仅用于演示)
    tts = TinyTrajectoryStore()
    
    # 3. 创建工具
    tools = create_mock_tools()
    
    # 4. 创建编排器
    orchestrator = ReactXenOrchestrator(
        llm_provider=llm,
        tts=tts,
        tools=tools,
        max_reflect_iterations=3,
    )
    
    # 5. 运行测试
    query = "What is the meaning of life?"
    result = orchestrator.run(query)
    
    # 6. 打印结果
    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)
    print(f"Success: {result['success']}")
    print(f"Answer: {result['answer']}")
    print(f"Iterations: {result['iterations']}")
    print(f"Trace steps: {len(result['trace'])}")
    
    print("\n--- Execution Trace ---")
    for step_name, step_result in result["trace"]:
        print(f"  [{step_name}] Status: {step_result.status.value}")
    
    return result


if __name__ == "__main__":
    run_sandbox_demo()
