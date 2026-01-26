"""
Pipeline - Agent 执行管线

严格按照 pipeline.tex 架构实现的完整执行流程。

架构组件 (参考 pipeline.tex):
    主进程 (System 1):
        - Distillation Agent: 接收 User Query，执行 Hot Start
        - ReAct Agent: Thought-Action-Observation 循环
        - TTS Buffer: 只存上一轮轨迹 (Markov 约束)
        - Review Agent: 检查任务完成状态
        - Reflect Agent: 失败分析，短期修正 + 长期推送
    
    后台进程 (System 2):
        - Failure Queue: 接收失败轨迹
        - MCTS Simulator: Off-policy 探索
        - HCC: 分层知识存储
        - Lazy Update: 异步注入 Wisdom

设计原则:
    - 代码结构与架构图一一对应
    - 工程友好：完整类型注解、详细文档、清晰日志
    - Keep it simple: 最小化外部依赖
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Protocol, Union

from Memory.memory import AgentMemory, MemoryConfig
from .failure_queue import FailureQueue, FailureTrajectory, FailureType, create_failure_trajectory


# =============================================================================
# 日志配置
# =============================================================================

logger = logging.getLogger(__name__)


# =============================================================================
# 协议定义
# =============================================================================

class Agent(Protocol):
    """Agent 协议 - 所有 Agent 必须实现此接口"""
    
    def run(self, context: Any) -> "AgentResult":
        """执行 Agent 逻辑
        
        Args:
            context: 执行上下文 (格式由具体 Agent 定义)
            
        Returns:
            AgentResult 执行结果
        """
        ...


@dataclass
class AgentResult:
    """Agent 执行结果
    
    Attributes:
        output: 主要输出内容
        thought: ReAct 的思考过程
        action: 执行的动作
        observation: 观察结果
        accomplished: 任务是否完成
        needs_reflect: 是否需要反思
        trajectory: 完整执行轨迹 (用于 HCC 记录)
        metadata: 附加元数据
    """
    output: str
    thought: str = ""
    action: str = ""
    observation: str = ""
    accomplished: bool = False
    needs_reflect: bool = False
    trajectory: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineConfig:
    """Pipeline 配置
    
    Attributes:
        max_iterations: ReAct 最大迭代次数
        markov_window: Markov 上下文窗口大小 (默认 1)
        enable_mcts: 是否启用后台 MCTS
        failure_queue_size: Failure Queue 最大容量
        push_to_failure_queue_on_max_iter: 达到最大迭代时是否推送到 Failure Queue
    """
    max_iterations: int = 100
    markov_window: int = 1
    enable_mcts: bool = True
    failure_queue_size: int = 100
    push_to_failure_queue_on_max_iter: bool = True


@dataclass
class PipelineResult:
    """Pipeline 执行结果
    
    Attributes:
        success: 是否成功完成
        output: 最终输出
        iterations: 执行的迭代次数
        trajectory: 完整轨迹
        failure_pushed: 是否推送到 Failure Queue
    """
    success: bool
    output: str
    iterations: int = 0
    trajectory: str = ""
    failure_pushed: bool = False


# =============================================================================
# Pipeline 实现
# =============================================================================

class Pipeline:
    """ZYAgent Pipeline: 双系统架构的主进程实现
    
    严格按照 pipeline.tex 架构图实现:
    
    ```
    User Query 
        ↓
    Distillation Agent (Hot Start: HCC L2 → Wisdom)
        ↓
    ReAct Agent (TTS Buffer 提供 Markov Context)
        ↓
    Decision: 任务完成?
        ├─ YES → Review Agent → Success → HCC L1
        └─ NO  → Reflect Agent
                    ├─ 短期修正 → TTS Buffer → 继续 ReAct
                    └─ 长期记忆 → Failure Queue → MCTS (System 2)
    ```
    
    Example:
        >>> pipeline = Pipeline(
        ...     memory=AgentMemory(),
        ...     distillation_agent=distill,
        ...     react_agent=react,
        ...     review_agent=review,
        ...     reflect_agent=reflect,
        ... )
        >>> result = pipeline.run("请帮我分析这段代码...")
    """
    
    def __init__(
        self,
        memory: AgentMemory,
        react_agent: Agent,
        distillation_agent: Optional[Agent] = None,
        review_agent: Optional[Agent] = None,
        reflect_agent: Optional[Agent] = None,
        mcts_bridge: Optional[Any] = None,
        failure_queue: Optional[FailureQueue] = None,
        config: Optional[PipelineConfig] = None,
    ):
        """初始化 Pipeline
        
        Args:
            memory: AgentMemory 实例 (整合 TTS + HCC)
            react_agent: ReAct Agent (必需)
            distillation_agent: Distillation Agent (Hot Start)
            review_agent: Review Agent (质量检查)
            reflect_agent: Reflect Agent (失败分析)
            mcts_bridge: AsyncMCTSBridge 实例 (System 2 接口)
            failure_queue: Failure Queue 实例
            config: Pipeline 配置
        """
        self.memory = memory
        self.react = react_agent
        self.distillation = distillation_agent
        self.review = review_agent
        self.reflect = reflect_agent
        self.mcts_bridge = mcts_bridge
        self.config = config or PipelineConfig()
        
        # 初始化 Failure Queue
        self.failure_queue = failure_queue or FailureQueue(
            max_size=self.config.failure_queue_size
        )
        
        # Lazy Update 队列 (接收来自 MCTS 的 Wisdom)
        self._pending_wisdom: List[str] = []
        
        # 当前任务状态
        self._current_task_id: Optional[str] = None
        self._current_trajectory: List[str] = []
    
    def run(self, query: str) -> PipelineResult:
        """执行 Pipeline
        
        按照架构图的流程执行:
        1. Distillation Agent (Hot Start)
        2. ReAct Loop (Markov Context)
        3. Review/Reflect 分支
        4. Success/Failure 轨迹回流
        
        Args:
            query: 用户查询
            
        Returns:
            PipelineResult 执行结果
        """
        self._current_task_id = str(uuid.uuid4())[:8]
        self._current_trajectory = []
        
        logger.info(f"[Pipeline] Starting task {self._current_task_id}: {query[:50]}...")
        
        # =====================================================================
        # 1. Hot Start: Distillation Agent 初始化
        # =====================================================================
        context = self._hot_start(query)
        
        # =====================================================================
        # 2. Main Loop: ReAct + Review + Reflect
        # =====================================================================
        for iteration in range(1, self.config.max_iterations + 1):
            logger.info(f"[Pipeline] Iteration {iteration}/{self.config.max_iterations}")
            
            # 2.1 Lazy Update: 注入来自 MCTS 的 Wisdom
            self._inject_pending_wisdom()
            
            # 2.2 ReAct 执行
            react_result = self._execute_react(context)
            self._record_trajectory(f"[ReAct] {react_result.thought}")
            
            # 2.3 Decision: 任务完成?
            if react_result.accomplished:
                # 2.4 Review 检查
                if self.review:
                    review_result = self._execute_review(react_result)
                    
                    if review_result.accomplished:
                        # SUCCESS 路径
                        return self._handle_success(
                            query=query,
                            output=react_result.output,
                            iterations=iteration,
                        )
                    
                    # Review 判定失败 → Reflect
                    if self.reflect:
                        self._execute_reflect(
                            query=query,
                            react_result=react_result,
                            review_result=review_result,
                            iteration=iteration,
                        )
                else:
                    # 无 Review，直接 Success
                    return self._handle_success(
                        query=query,
                        output=react_result.output,
                        iterations=iteration,
                    )
            else:
                # 任务未完成 → Reflect
                if self.reflect:
                    self._execute_reflect(
                        query=query,
                        react_result=react_result,
                        review_result=None,
                        iteration=iteration,
                    )
            
            # 2.5 更新上下文 (Markov 约束)
            context = self.memory.get_context(window=self.config.markov_window)
        
        # =====================================================================
        # 3. Max Iterations → Failure
        # =====================================================================
        return self._handle_max_iterations(query)
    
    # =========================================================================
    # 内部方法: Hot Start
    # =========================================================================
    
    def _hot_start(self, query: str) -> str:
        """Hot Start: 从 HCC L2 提取 Wisdom 并初始化上下文
        
        架构 (pipeline.tex):
            - Distillation Agent 作为入口
            - Hot Start: 从 HCC L2/L3 获取 Wisdom (一次性)
            - TTS Buffer 准备记录执行轨迹
        
        Args:
            query: 用户查询
            
        Returns:
            初始上下文字符串 (包含 HCC Wisdom)
        """
        logger.info("[Pipeline] Hot Start: Fetching wisdom from HCC L2")
        
        # 1. 调用 Distillation Agent (如果有)
        if self.distillation:
            try:
                distill_result = self.distillation.run(query)
                if distill_result.output:
                    self._record_trajectory(f"[Distillation] {distill_result.output[:100]}...")
            except Exception as e:
                logger.warning(f"[Pipeline] Distillation Agent error: {e}")
        
        # 2. 从 Memory 初始化任务 (Hot Start: HCC Wisdom)
        context = self.memory.start_task(
            task_description=query,
            fetch_wisdom=True,  # 从 HCC 获取 Wisdom
        )
        
        # 3. 记录 Hot Start Wisdom
        hot_wisdom = self.memory.get_hot_start_wisdom()
        if hot_wisdom:
            self._record_trajectory(f"[Hot Start Wisdom] {hot_wisdom[:100]}...")
        
        return context
    
    # =========================================================================
    # 内部方法: Agent 执行
    # =========================================================================
    
    def _execute_react(self, context: str) -> AgentResult:
        """执行 ReAct Agent
        
        Args:
            context: 当前上下文 (受 Markov 约束)
            
        Returns:
            AgentResult
        """
        result = self.react.run(context)
        
        # 记录到 Memory
        self.memory.record_thought_action(
            thought=result.thought,
            action=result.action,
            observation=result.observation,
        )
        
        return result
    
    def _execute_review(self, react_result: AgentResult) -> AgentResult:
        """执行 Review Agent
        
        Args:
            react_result: ReAct 的执行结果
            
        Returns:
            Review 结果
        """
        logger.info("[Pipeline] Executing Review Agent")
        
        result = self.review.run(react_result.output)
        self._record_trajectory(f"[Review] accomplished={result.accomplished}")
        
        return result
    
    def _execute_reflect(
        self,
        query: str,
        react_result: AgentResult,
        review_result: Optional[AgentResult],
        iteration: int,
    ) -> None:
        """执行 Reflect Agent
        
        分支逻辑:
        - 短期修正: 更新 TTS Buffer (通过 Memory)
        - 长期记忆: 推送到 Failure Queue (供 MCTS 探索)
        
        Args:
            query: 原始查询
            react_result: ReAct 结果
            review_result: Review 结果 (可选)
            iteration: 当前迭代次数
        """
        logger.info("[Pipeline] Executing Reflect Agent")
        
        # 构建 Reflect 输入
        reflect_input = react_result.output
        if review_result:
            reflect_input += f"\n\nReview Feedback: {review_result.output}"
        
        # 调用 Reflect Agent
        reflect_result = self.reflect.run(reflect_input)
        self._record_trajectory(f"[Reflect] {reflect_result.output[:100]}...")
        
        # 短期修正: 记录到 Memory
        self.memory.record_event(f"[Reflect] {reflect_result.output}")
        
        # 长期记忆: 推送到 Failure Queue (如果需要)
        if reflect_result.needs_reflect or (review_result and not review_result.accomplished):
            self._push_to_failure_queue(
                query=query,
                trajectory="\n".join(self._current_trajectory),
                failure_type=FailureType.REVIEW_REJECTED if review_result else FailureType.TASK_NOT_ACCOMPLISHED,
                failure_reason=review_result.output if review_result else "Task not accomplished",
                reflect_output=reflect_result.output,
                iteration=iteration,
            )
    
    # =========================================================================
    # 内部方法: 结果处理
    # =========================================================================
    
    def _handle_success(
        self,
        query: str,
        output: str,
        iterations: int,
    ) -> PipelineResult:
        """处理成功场景
        
        架构位置: Success → HCC L1
        
        Args:
            query: 原始查询
            output: 最终输出
            iterations: 迭代次数
        """
        logger.info(f"[Pipeline] Task completed successfully in {iterations} iterations")
        
        # 1. 完成任务 (触发 Memory 的知识提升)
        self.memory.finish_task(final_result=output)
        
        # 2. 记录成功轨迹到 HCC L1 (通过 inject_wisdom)
        success_trajectory = "\n".join(self._current_trajectory)
        self.memory.inject_wisdom(
            wisdom=f"Successfully solved: {query[:50]}...\nApproach: {output[:100]}...",
            search_traces=[success_trajectory],
            promote_to_l2=False,  # 先存 L1，等待批量提升
        )
        
        return PipelineResult(
            success=True,
            output=output,
            iterations=iterations,
            trajectory=success_trajectory,
            failure_pushed=False,
        )
    
    def _handle_max_iterations(self, query: str) -> PipelineResult:
        """处理达到最大迭代次数的场景
        
        Args:
            query: 原始查询
        """
        logger.warning(f"[Pipeline] Max iterations ({self.config.max_iterations}) reached")
        
        trajectory = "\n".join(self._current_trajectory)
        failure_pushed = False
        
        # 推送到 Failure Queue
        if self.config.push_to_failure_queue_on_max_iter:
            self._push_to_failure_queue(
                query=query,
                trajectory=trajectory,
                failure_type=FailureType.MAX_ITERATIONS,
                failure_reason=f"Reached max iterations ({self.config.max_iterations})",
                reflect_output="",
                iteration=self.config.max_iterations,
            )
            failure_pushed = True
        
        return PipelineResult(
            success=False,
            output=f"Max iterations ({self.config.max_iterations}) reached",
            iterations=self.config.max_iterations,
            trajectory=trajectory,
            failure_pushed=failure_pushed,
        )
    
    # =========================================================================
    # 内部方法: Failure Queue
    # =========================================================================
    
    def _push_to_failure_queue(
        self,
        query: str,
        trajectory: str,
        failure_type: FailureType,
        failure_reason: str,
        reflect_output: str,
        iteration: int,
    ) -> None:
        """推送失败轨迹到 Failure Queue
        
        架构位置: Reflect Agent → Failure Queue → MCTS
        """
        logger.info(f"[Pipeline] Pushing to Failure Queue: {failure_type.value}")
        
        failure = create_failure_trajectory(
            query=query,
            trajectory=trajectory,
            failure_type=failure_type,
            failure_reason=failure_reason,
            reflect_output=reflect_output,
            iteration=iteration,
            task_id=self._current_task_id,
        )
        
        self.failure_queue.push(failure)
    
    # =========================================================================
    # 内部方法: Lazy Update
    # =========================================================================
    
    def _inject_pending_wisdom(self) -> None:
        """注入来自 MCTS 的 Wisdom (Lazy Update)
        
        架构位置: System 2 → Lazy Update → 主进程
        """
        while self._pending_wisdom:
            wisdom = self._pending_wisdom.pop(0)
            logger.info(f"[Pipeline] Injecting wisdom: {wisdom[:50]}...")
            self.memory.inject_wisdom(wisdom, promote_to_l2=True)
    
    def add_pending_wisdom(self, wisdom: str) -> None:
        """添加 pending wisdom (供 MCTS Bridge 调用)
        
        Args:
            wisdom: 来自 MCTS 的智慧
        """
        self._pending_wisdom.append(wisdom)
    
    # =========================================================================
    # 内部方法: 轨迹记录
    # =========================================================================
    
    def _record_trajectory(self, entry: str) -> None:
        """记录轨迹条目"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self._current_trajectory.append(f"[{timestamp}] {entry}")
    
    # =========================================================================
    # 公共方法: 状态查询
    # =========================================================================
    
    def get_failure_queue_stats(self) -> Dict[str, int]:
        """获取 Failure Queue 统计信息"""
        return self.failure_queue.get_stats()
