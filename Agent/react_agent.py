"""
ReAct Agent - 核心执行 Agent

执行标准的 Thought-Action-Observation 循环，
是 ReactXen 架构中的核心工作单元。
"""

import re
from typing import List, Dict, Any, Optional, Callable

from .base import BaseAgent, AgentContext, AgentResult, AgentStatus, LLMProvider


# =============================================================================
# Tool 定义
# =============================================================================

class Tool:
    """工具定义"""
    
    def __init__(
        self, 
        name: str, 
        description: str, 
        func: Callable[[str], str],
    ):
        """
        Args:
            name: 工具名称
            description: 工具描述
            func: 工具执行函数，接收输入字符串，返回输出字符串
        """
        self.name = name
        self.description = description
        self.func = func
    
    def execute(self, input_str: str) -> str:
        """执行工具"""
        try:
            return self.func(input_str)
        except Exception as e:
            return f"Error: {e}"


# 内置工具
def _finish_tool(input_str: str) -> str:
    """Finish 工具，标记任务完成"""
    return f"[FINISH] {input_str}"


FINISH_TOOL = Tool(
    name="Finish",
    description="Use this when you have the final answer. Input should be the final answer.",
    func=_finish_tool,
)


# =============================================================================
# ReAct Agent
# =============================================================================

class ReActAgent(BaseAgent):
    """ReAct Agent
    
    职责:
        - 执行 Thought-Action-Observation 循环
        - 调用工具获取信息
        - 生成最终答案
    
    在 ReactXen 架构中的位置:
        - 内层循环的核心执行单元
        - 接收 DistillationAgent 的 examples 和 ReflectAgent 的 reflections
    """
    
    # Prompt 模板
    SYSTEM_PROMPT = """You are an AI assistant that solves tasks using the ReAct paradigm.

{examples}

{reflections}

Available Tools:
{tools}

Instructions:
1. Think step by step about what to do
2. Choose an appropriate action
3. Observe the result
4. Repeat until you have the final answer
5. Use the Finish tool to provide your final answer

Format your response as:
Thought: [your reasoning]
Action: [tool name]
Action Input: [input for the tool]

Current Task: {query}
"""
    
    def __init__(
        self, 
        llm_provider: LLMProvider, 
        tools: Optional[List[Tool]] = None,
        max_steps: int = 10,
    ):
        """
        Args:
            llm_provider: LLM 提供者
            tools: 可用工具列表
            max_steps: 最大执行步数
        """
        super().__init__(llm_provider, name="ReActAgent")
        self.tools = tools or []
        self.tools.append(FINISH_TOOL)  # 总是包含 Finish 工具
        self.max_steps = max_steps
        self._tool_map = {t.name.lower(): t for t in self.tools}
    
    def run(self, context: AgentContext) -> AgentResult:
        """执行 ReAct 循环
        
        Args:
            context: 执行上下文
            
        Returns:
            执行结果
        """
        self._log(f"Starting ReAct loop for: {context.query[:50]}...")
        
        # 1. 构建初始 Prompt
        scratchpad = ""
        tools_desc = self._format_tools()
        
        for step in range(1, self.max_steps + 1):
            self._log(f"Step {step}/{self.max_steps}")
            
            # 2. 构建当前 Prompt
            prompt = self.SYSTEM_PROMPT.format(
                examples=context.examples or "No examples available.",
                reflections=context.get_reflections_str() or "No previous reflections.",
                tools=tools_desc,
                query=context.query,
            )
            prompt += f"\n\nScratchpad:\n{scratchpad}\n\nThought {step}:"
            
            # 3. 调用 LLM 获取 Thought
            response = self._call_llm(prompt)
            thought, action, action_input = self._parse_response(response)
            
            self._log(f"Thought: {thought[:80]}...")
            self._log(f"Action: {action}, Input: {action_input[:50]}...")
            
            # 4. 更新 Scratchpad
            scratchpad += f"\nThought {step}: {thought}"
            scratchpad += f"\nAction {step}: {action}"
            scratchpad += f"\nAction Input {step}: {action_input}"
            
            # 5. 执行 Action
            if action.lower() == "finish":
                # 任务完成
                self._log(f"Task finished with answer: {action_input[:100]}...")
                context.scratchpad = scratchpad
                context.current_answer = action_input
                
                return AgentResult(
                    status=AgentStatus.NEEDS_REVIEW,
                    output=action_input,
                    reasoning=scratchpad,
                    metadata={"steps": step},
                )
            
            # 执行工具
            observation = self._execute_tool(action, action_input)
            scratchpad += f"\nObservation {step}: {observation}"
            self._log(f"Observation: {observation[:80]}...")
        
        # 超过最大步数
        self._log("Max steps reached, task not finished")
        context.scratchpad = scratchpad
        
        return AgentResult(
            status=AgentStatus.NOT_FINISHED,
            output="",
            reasoning=f"Reached max steps ({self.max_steps}) without finishing",
            metadata={"steps": self.max_steps, "scratchpad": scratchpad},
        )
    
    def _format_tools(self) -> str:
        """格式化工具描述"""
        lines = []
        for tool in self.tools:
            lines.append(f"- {tool.name}: {tool.description}")
        return "\n".join(lines)
    
    def _parse_response(self, response: str) -> tuple:
        """解析 LLM 响应
        
        Returns:
            (thought, action, action_input)
        """
        thought = ""
        action = "Finish"
        action_input = "Could not parse response"
        
        # 提取 Thought (第一行或直到 Action)
        thought_match = re.search(r"(?:Thought[:\s]*)?(.+?)(?=\nAction|\Z)", response, re.DOTALL)
        if thought_match:
            thought = thought_match.group(1).strip()
        
        # 提取 Action
        action_match = re.search(r"Action[:\s]*([^\n]+)", response)
        if action_match:
            action = action_match.group(1).strip()
        
        # 提取 Action Input
        input_match = re.search(r"Action Input[:\s]*(.+?)(?=\n|$)", response, re.DOTALL)
        if input_match:
            action_input = input_match.group(1).strip()
        
        return thought, action, action_input
    
    def _execute_tool(self, action: str, action_input: str) -> str:
        """执行工具"""
        action_lower = action.lower()
        
        if action_lower in self._tool_map:
            tool = self._tool_map[action_lower]
            return tool.execute(action_input)
        
        return f"Unknown tool: {action}. Available tools: {list(self._tool_map.keys())}"
