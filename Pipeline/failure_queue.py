"""
Failure Queue - 失败队列

连接 System 1 (主进程) 和 System 2 (后台进程) 的桥梁。

架构角色 (参考 pipeline.tex):
    - 接收来自 Reflect Agent 的失败轨迹
    - 供 MCTS Simulator 拉取任务进行 Off-policy 探索
    - 线程安全，支持异步读写

设计原则:
    - Keep it simple: 使用标准库 queue.Queue
    - 工程友好: 完整的类型注解和文档
"""

import threading
import queue
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum


class FailureType(Enum):
    """失败类型分类"""
    TASK_NOT_ACCOMPLISHED = "task_not_accomplished"  # 任务未完成
    REVIEW_REJECTED = "review_rejected"              # Review 判定失败
    MAX_ITERATIONS = "max_iterations"                # 超过最大迭代次数
    TOOL_ERROR = "tool_error"                        # 工具调用错误
    UNKNOWN = "unknown"


@dataclass
class FailureTrajectory:
    """失败轨迹 - Failure Queue 的存储单元
    
    包含完整的上下文信息，供 MCTS 进行 Off-policy 探索。
    
    Attributes:
        task_id: 任务唯一标识
        query: 原始用户查询
        trajectory: 执行过程的完整轨迹 (Thought-Action-Observation 序列)
        failure_type: 失败类型
        failure_reason: 失败原因描述
        reflect_output: Reflect Agent 的分析结果
        iteration: 失败时的迭代次数
        metadata: 附加元数据
    """
    task_id: str
    query: str
    trajectory: str
    failure_type: FailureType = FailureType.UNKNOWN
    failure_reason: str = ""
    reflect_output: str = ""
    iteration: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_mcts_context(self) -> str:
        """转换为 MCTS 可用的上下文格式
        
        Returns:
            格式化的上下文字符串，包含任务和失败信息
        """
        return f"""## Task
{self.query}

## Previous Attempt (Failed)
{self.trajectory}

## Failure Analysis
Type: {self.failure_type.value}
Reason: {self.failure_reason}

## Reflection
{self.reflect_output}
"""


class FailureQueue:
    """失败队列 - 线程安全的 FIFO 队列
    
    架构位置:
        - 输入: Reflect Agent 推送失败轨迹
        - 输出: AsyncMCTSBridge 拉取任务
    
    Example:
        >>> queue = FailureQueue(max_size=100)
        >>> queue.push(FailureTrajectory(...))
        >>> if not queue.is_empty():
        ...     task = queue.pop()
        ...     # 交给 MCTS 处理
    """
    
    def __init__(self, max_size: int = 100):
        """
        Args:
            max_size: 队列最大容量，超过后丢弃最早的任务
        """
        self._queue: queue.Queue[FailureTrajectory] = queue.Queue(maxsize=max_size)
        self._max_size = max_size
        self._lock = threading.Lock()
        
        # 统计信息
        self._total_pushed = 0
        self._total_popped = 0
        self._total_dropped = 0
    
    def push(self, trajectory: FailureTrajectory) -> bool:
        """推送失败轨迹到队列
        
        如果队列已满，丢弃最早的任务以腾出空间。
        
        Args:
            trajectory: 失败轨迹
            
        Returns:
            是否成功推送 (总是返回 True，除非发生异常)
        """
        with self._lock:
            # 如果队列满了，先移除最早的
            if self._queue.full():
                try:
                    self._queue.get_nowait()
                    self._total_dropped += 1
                except queue.Empty:
                    pass
            
            try:
                self._queue.put_nowait(trajectory)
                self._total_pushed += 1
                return True
            except queue.Full:
                # 理论上不会到这里，因为我们已经移除了一个
                self._total_dropped += 1
                return False
    
    def pop(self, timeout: Optional[float] = None) -> Optional[FailureTrajectory]:
        """从队列中取出一个失败轨迹
        
        Args:
            timeout: 等待超时时间 (秒)，None 表示立即返回
            
        Returns:
            失败轨迹，如果队列为空则返回 None
        """
        try:
            if timeout is None:
                trajectory = self._queue.get_nowait()
            else:
                trajectory = self._queue.get(timeout=timeout)
            
            with self._lock:
                self._total_popped += 1
            return trajectory
        except queue.Empty:
            return None
    
    def pop_batch(self, batch_size: int = 5) -> List[FailureTrajectory]:
        """批量取出失败轨迹
        
        Args:
            batch_size: 批量大小
            
        Returns:
            失败轨迹列表，可能少于 batch_size
        """
        batch = []
        for _ in range(batch_size):
            item = self.pop()
            if item is None:
                break
            batch.append(item)
        return batch
    
    def is_empty(self) -> bool:
        """检查队列是否为空"""
        return self._queue.empty()
    
    def size(self) -> int:
        """获取当前队列大小"""
        return self._queue.qsize()
    
    def get_stats(self) -> Dict[str, int]:
        """获取队列统计信息
        
        Returns:
            包含 pushed/popped/dropped/current_size 的字典
        """
        with self._lock:
            return {
                "total_pushed": self._total_pushed,
                "total_popped": self._total_popped,
                "total_dropped": self._total_dropped,
                "current_size": self._queue.qsize(),
                "max_size": self._max_size,
            }
    
    def clear(self) -> int:
        """清空队列
        
        Returns:
            被清除的任务数量
        """
        count = 0
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                count += 1
            except queue.Empty:
                break
        return count


# =============================================================================
# 工厂函数
# =============================================================================

def create_failure_trajectory(
    query: str,
    trajectory: str,
    failure_type: FailureType = FailureType.UNKNOWN,
    failure_reason: str = "",
    reflect_output: str = "",
    iteration: int = 0,
    task_id: Optional[str] = None,
) -> FailureTrajectory:
    """创建失败轨迹的便捷函数
    
    Args:
        query: 用户查询
        trajectory: 执行轨迹
        failure_type: 失败类型
        failure_reason: 失败原因
        reflect_output: 反思输出
        iteration: 迭代次数
        task_id: 任务 ID，如果不提供则自动生成
        
    Returns:
        FailureTrajectory 实例
    """
    import uuid
    
    if task_id is None:
        task_id = str(uuid.uuid4())[:8]
    
    return FailureTrajectory(
        task_id=task_id,
        query=query,
        trajectory=trajectory,
        failure_type=failure_type,
        failure_reason=failure_reason,
        reflect_output=reflect_output,
        iteration=iteration,
    )
