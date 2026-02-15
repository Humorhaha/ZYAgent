"""
System 1 Agent - 闭环 ReAct-Review-Reflect Agent

基于 ReactXen 架构实现:
    Query -> ReAct -> Review -> [Accomplished?] -> Output
                              -> [Not Accomplished] -> Reflect -> ReAct (loop)

核心组件:
    - ReActAgent: 执行 Thought-Action-Observation 循环
    - ReviewAgent: 评估任务完成质量 (Accomplished/Not/Partially)
    - ReflectAgent: Self-Ask 分析失败原因，生成改进建议
    - TTS: 工作记忆，每轮更新和使用

设计约束:
    - 最大反思迭代次数 M (默认 3)
    - 每个 ReAct 最大步数 (默认 8)
    - TTS 作为工作记忆，每轮更新
"""

from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
import json
import re
import logging

# Local imports
from LLM.prompts import (
    REACT_IOT_SYSTEM_PROMPT,
    REVIEW_SYSTEM_PROMPT,
    REFLECT_SELF_ASK_PROMPT,
    REFLECTION_INJECTION_HEADER,
    TTS_EXAMPLE_FORMAT,
)
from core.trajectory import Trajectory, Step

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class AgentContext:
    """Agent 执行上下文 - 在各 Agent 之间传递"""
    query: str
    scratchpad: str = ""
    current_answer: str = ""
    review_feedback: str = ""
    reflections: List[str] = field(default_factory=list)
    iteration: int = 0
    tool_calls: List[Dict] = field(default_factory=list)
    
    def get_reflections_str(self) -> str:
        """获取格式化的反思字符串"""
        if not self.reflections:
            return ""
        return REFLECTION_INJECTION_HEADER.format(
            reflections="\n".join(f"{i+1}. {r}" for i, r in enumerate(self.reflections))
        )


@dataclass
class AgentResult:
    """Agent 执行结果"""
    status: str  # "success", "not_finished", "needs_reflect"
    output: str = ""
    reasoning: str = ""
    suggestions: List[str] = field(default_factory=list)
    
    @property
    def is_success(self) -> bool:
        return self.status == "success"
    
    @property
    def needs_reflect(self) -> bool:
        return self.status in ("not_finished", "needs_reflect", "failed")


# =============================================================================
# TTS Working Memory Manager
# =============================================================================

class TTSWorkingMemory:
    """TTS 工作记忆 - JSON 存储，每轮更新"""
    
    def __init__(self, json_path: str = "tts_working_memory.json"):
        self.json_path = json_path
        self.trajectories: List[Dict] = []
        self._load()
    
    def _load(self):
        """从 JSON 加载"""
        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.trajectories = data.get("trajectories", [])
        except (FileNotFoundError, json.JSONDecodeError):
            self.trajectories = []
    
    def _save(self):
        """保存到 JSON"""
        with open(self.json_path, 'w', encoding='utf-8') as f:
            json.dump({"trajectories": self.trajectories}, f, indent=2, ensure_ascii=False)
    
    def add(self, trajectory: Dict):
        """添加成功的轨迹"""
        self.trajectories.append(trajectory)
        self._save()
    
    def retrieve(self, query: str, category: str = "", k: int = 2) -> List[Dict]:
        """检索相关轨迹 (简单关键词匹配)"""
        if not self.trajectories:
            return []
        
        # 按类别过滤
        candidates = self.trajectories
        if category:
            candidates = [t for t in candidates if t.get("category") == category]
        
        # 简单相关性排序 (可扩展为 embedding)
        query_words = set(query.lower().split())
        scored = []
        for t in candidates:
            task_words = set(t.get("query", "").lower().split())
            score = len(query_words & task_words)
            scored.append((score, t))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        return [t for _, t in scored[:k]]
    
    def format_for_prompt(self, trajectories: List[Dict], max_examples: int = 2) -> str:
        """格式化为 Prompt"""
        if not trajectories:
            return "(无可用示例)"
        
        examples = []
        for i, t in enumerate(trajectories[:max_examples], 1):
            steps_text = ""
            for step in t.get("steps", [])[:3]:  # 限制步骤数
                steps_text += f"  Thought: {step.get('thought', '')[:100]}\n"
                steps_text += f"  Action: {step.get('action', '')}\n"
            
            examples.append(TTS_EXAMPLE_FORMAT.format(
                index=i,
                category=t.get("category", "General"),
                query=t.get("query", "")[:100],
                steps=steps_text,
                answer=t.get("answer", "")[:100]
            ))
        
        return "\n".join(examples)


# =============================================================================
# Component Agents
# =============================================================================

class ReActAgent:
    """ReAct Agent - 执行 Thought-Action-Observation 循环"""
    
    def __init__(self, llm, toolkit, max_steps: int = 8):
        self.llm = llm
        self.toolkit = toolkit
        self.max_steps = max_steps
    
    def run(self, context: AgentContext, tts_examples: str = "") -> AgentResult:
        """执行 ReAct 循环"""
        logger.info(f"[ReActAgent] Starting for: {context.query[:50]}...")
        
        tools_desc = self._format_tools()
        scratchpad = context.scratchpad or ""
        
        for step in range(1, self.max_steps + 1):
            # 1. 构建 Prompt
            prompt = REACT_IOT_SYSTEM_PROMPT.format(
                tool_desc=tools_desc,
                tool_names=", ".join(self.toolkit.list_tools()),
                examples=tts_examples or "(无可用示例)",
                reflections=context.get_reflections_str() or "(无历史反思)",
                question=context.query,
                scratchpad=scratchpad or "(无历史步骤)",
            )

            
            # 2. 调用 LLM
            response = self.llm.generate(prompt)
            thought, action, action_input = self._parse_response(response)
            
            logger.info(f"  Step {step}: {action}({action_input[:50]})")
            
            # 3. 更新 Scratchpad
            scratchpad += f"\nThought {step}: {thought}"
            scratchpad += f"\nAction {step}: {action}"
            scratchpad += f"\nAction Input {step}: {action_input}"
            
            # 4. 检查是否完成
            if action.lower() == "finish":
                context.scratchpad = scratchpad
                context.current_answer = action_input
                return AgentResult(
                    status="needs_review",
                    output=action_input,
                    reasoning=scratchpad,
                )
            
            # 5. 执行工具
            observation = self._execute_tool(action, action_input)
            scratchpad += f"\nObservation {step}: {observation}"
            
            context.tool_calls.append({
                "tool": action,
                "args": action_input,
                "result": observation[:200]
            })
        
        # 超过最大步数
        context.scratchpad = scratchpad
        return AgentResult(
            status="not_finished",
            output="",
            reasoning=f"达到最大步数 ({self.max_steps})",
        )
    
    def _format_tools(self) -> str:
        """格式化工具描述"""
        tools = self.toolkit.list_tools()
        lines = []
        for name in tools:
            lines.append(f"- {name}")
        return "\n".join(lines)
    
    def _parse_response(self, response: str) -> Tuple[str, str, str]:
        """解析 LLM 响应"""
        thought = ""
        action = "Finish"
        action_input = ""
        
        # 提取 Thought
        thought_match = re.search(r'(?:Thought|思路)[:\s]*(.+?)(?=\n|Action|$)', response, re.DOTALL)
        if thought_match:
            thought = thought_match.group(1).strip()
        
        # 提取 Action
        action_match = re.search(r'Action[:\s]*(\w+)', response)
        if action_match:
            action = action_match.group(1).strip()
        
        # 提取 Action Input
        input_match = re.search(r'Action Input[:\s]*(.+?)(?=\n|$)', response, re.DOTALL)
        if input_match:
            action_input = input_match.group(1).strip()
        
        return thought, action, action_input
    
    def _execute_tool(self, action: str, action_input: str) -> str:
        """执行工具"""
        available_tools = self.toolkit.list_tools()
        
        if action not in available_tools:
            return f"Error: Unknown tool '{action}'"
        
        try:
            # 解析参数
            kwargs = self._parse_tool_args(action_input)
            result = self.toolkit.call(action, **kwargs)
            
            if isinstance(result, dict):
                return json.dumps(result, indent=2, ensure_ascii=False, default=str)[:1000]
            return str(result)[:1000]
        except Exception as e:
            return f"Error: {e}"
    
    def _parse_tool_args(self, args_str: str) -> Dict[str, Any]:
        """解析工具参数"""
        kwargs = {}
        if not args_str.strip():
            return kwargs
        
        # 关键字参数: key="value"
        kv_pattern = r'(\w+)\s*=\s*(?:"([^"]*)"|\'([^\']*)\'|(\{[^}]+\})|(\w+))'
        for match in re.finditer(kv_pattern, args_str):
            key = match.group(1)
            value = match.group(2) or match.group(3) or match.group(4) or match.group(5)
            if value:
                if value.startswith('{'):
                    try:
                        value = json.loads(value.replace("'", '"'))
                    except:
                        pass
                elif value.isdigit():
                    value = int(value)
                kwargs[key] = value
        
        return kwargs


class ReviewAgent:
    """Review Agent - 评估任务完成质量"""
    
    def __init__(self, llm, max_retries: int = 2):
        self.llm = llm
        self.max_retries = max_retries
    
    def run(self, context: AgentContext) -> AgentResult:
        """评估任务完成质量"""
        logger.info(f"[ReviewAgent] Reviewing answer: {context.current_answer[:50]}...")
        
        prompt = REVIEW_SYSTEM_PROMPT.format(
            query=context.query,
            scratchpad=context.scratchpad or "(无执行过程)",
            answer=context.current_answer or "(无答案)",
        )
        
        for _ in range(self.max_retries):
            response = self.llm.generate(prompt)
            parsed = self._parse_review(response)
            
            if parsed.get("status") != "Error":
                status = parsed.get("status", "").lower()
                
                if "accomplished" in status and "not" not in status:
                    result_status = "success"
                elif "partially" in status:
                    result_status = "needs_reflect"
                else:
                    result_status = "needs_reflect"
                
                context.review_feedback = json.dumps(parsed, ensure_ascii=False)
                
                logger.info(f"  Review Status: {result_status}")
                
                return AgentResult(
                    status=result_status,
                    output=parsed.get("status", ""),
                    reasoning=parsed.get("reasoning", ""),
                    suggestions=[parsed.get("suggestions", "")]
                )
        
        return AgentResult(status="needs_reflect", output="Error", reasoning="Failed to parse review")
    
    def _parse_review(self, response: str) -> Dict:
        """解析评审结果 (JSON)"""
        try:
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if match:
                return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
        
        # Fallback: regex 提取
        status = "Not Accomplished"
        reasoning = ""
        suggestions = ""
        
        status_match = re.search(r'"?status"?\s*:\s*"?([^",}]+)', response, re.IGNORECASE)
        if status_match:
            status = status_match.group(1).strip()
        
        reasoning_match = re.search(r'"?reasoning"?\s*:\s*"?([^"]+)', response, re.IGNORECASE)
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()
        
        return {"status": status, "reasoning": reasoning, "suggestions": suggestions}


class ReflectAgent:
    """Reflect Agent - Self-Ask 分析失败原因"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def run(self, context: AgentContext) -> AgentResult:
        """生成反思"""
        logger.info("[ReflectAgent] Self-Ask: Why is task not accomplished?")
        
        prompt = REFLECT_SELF_ASK_PROMPT.format(
            query=context.query,
            scratchpad=context.scratchpad or "(无执行过程)",
            review_feedback=context.review_feedback or "(无评审反馈)",
        )
        
        reflection = self.llm.generate(prompt).strip()
        context.reflections.append(reflection)
        
        logger.info(f"  Reflection: {reflection[:100]}...")
        
        return AgentResult(
            status="success",
            output=reflection,
            reasoning="Generated Short Term Reflection via Self-Ask",
        )


# =============================================================================
# Main Orchestrator
# =============================================================================

class System1Agent:
    """System 1 主编排器 - 闭环 ReAct-Review-Reflect
    
    流程:
        Query -> ReAct -> Review -> [Accomplished?] -> Output
                                 -> [Not Accomplished?] -> Reflect -> ReAct (loop)
    
    最大 M 次反思迭代后返回失败。
    """
    
    def __init__(
        self,
        llm,
        toolkit,
        max_react_steps: int = 8,
        max_reflect_iterations: int = 3,
        tts_json_path: str = "tts_working_memory.json",
    ):
        self.llm = llm
        self.toolkit = toolkit
        self.max_iterations = max_reflect_iterations
        
        # 初始化组件 Agents
        self.react_agent = ReActAgent(llm, toolkit, max_react_steps)
        self.review_agent = ReviewAgent(llm)
        self.reflect_agent = ReflectAgent(llm)
        
        # TTS 工作记忆
        self.tts = TTSWorkingMemory(tts_json_path)
        
        logger.info(f"[System1Agent] Initialized with max_iterations={max_reflect_iterations}")
    
    def run(self, task: str, category: str = "") -> Tuple[Trajectory, float]:
        """执行闭环 Agent 流程
        
        Args:
            task: 任务描述
            category: 任务类别 (用于 TTS 检索)
            
        Returns:
            (trajectory, confidence)
        """
        logger.info(f"[System1Agent] Starting task: {task[:50]}...")
        
        # 1. 初始化上下文
        context = AgentContext(query=task)
        trajectory = Trajectory(task=task)
        
        # 2. 从 TTS 获取示例
        tts_examples = self.tts.retrieve(task, category, k=2)
        tts_examples_str = self.tts.format_for_prompt(tts_examples)
        
        # 3. 主循环
        for iteration in range(1, self.max_iterations + 1):
            context.iteration = iteration
            logger.info(f"=== Iteration {iteration}/{self.max_iterations} ===")
            
            # 3.1 ReAct 执行
            react_result = self.react_agent.run(context, tts_examples_str)
            
            # 3.2 检查 ReAct 结果
            if react_result.status == "not_finished":
                # 未完成 -> Reflect
                self.reflect_agent.run(context)
                continue
            
            # 3.3 Review 评估
            review_result = self.review_agent.run(context)
            
            if review_result.is_success:
                # 成功！
                trajectory.add_step(
                    thought="Task completed successfully",
                    action="Finish",
                    observation=context.current_answer
                )
                trajectory.mark_success()
                trajectory.metadata["final_answer"] = context.current_answer
                trajectory.metadata["tool_calls"] = context.tool_calls
                trajectory.metadata["iterations"] = iteration
                
                # 更新 TTS 工作记忆
                self._update_tts(context, category)
                
                confidence = self._calculate_confidence(context)
                return trajectory, confidence
            
            # 3.4 需要反思
            self.reflect_agent.run(context)
        
        # 4. 达到最大迭代
        logger.warning(f"[System1Agent] Max iterations ({self.max_iterations}) reached")
        trajectory.mark_failure()
        trajectory.metadata["failure_reason"] = "max_reflect_iterations"
        trajectory.metadata["tool_calls"] = context.tool_calls
        
        return trajectory, 0.3
    
    def _calculate_confidence(self, context: AgentContext) -> float:
        """计算置信度"""
        tool_count = len(context.tool_calls)
        if tool_count >= 3:
            return 0.85
        elif tool_count >= 2:
            return 0.7
        elif tool_count >= 1:
            return 0.55
        return 0.3
    
    def _update_tts(self, context: AgentContext, category: str):
        """成功后更新 TTS 工作记忆"""
        trajectory_data = {
            "query": context.query,
            "category": category or "General",
            "steps": [
                {"thought": tc.get("tool", ""), "action": tc.get("args", ""), "observation": tc.get("result", "")}
                for tc in context.tool_calls
            ],
            "answer": context.current_answer,
            "success": True,
        }
        self.tts.add(trajectory_data)
        logger.info("[System1Agent] TTS working memory updated")
