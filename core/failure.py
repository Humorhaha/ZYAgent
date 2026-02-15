"""
Failure Case - 失败案例定义

封装失败的 Trajectory，供 System 2 搜索使用。
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Literal
from datetime import datetime

from .trajectory import Trajectory


FailureType = Literal["hard", "soft"]


@dataclass
class FailureCase:
    """失败案例
    
    Attributes:
        task: 原始任务描述
        trajectory: 失败的轨迹
        failure_type: 失败类型 (hard: 任务未完成, soft: 低置信度)
        failure_reason: 失败原因描述
        confidence: 置信度分数 (soft failure 时有意义)
        created_at: 创建时间
    """
    task: str
    trajectory: Trajectory
    failure_type: FailureType = "hard"
    failure_reason: str = ""
    confidence: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def trajectory_id(self) -> str:
        return self.trajectory.id
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "trajectory": self.trajectory.to_dict(),
            "failure_type": self.failure_type,
            "failure_reason": self.failure_reason,
            "confidence": self.confidence,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FailureCase":
        return cls(
            task=data["task"],
            trajectory=Trajectory.from_dict(data["trajectory"]),
            failure_type=data.get("failure_type", "hard"),
            failure_reason=data.get("failure_reason", ""),
            confidence=data.get("confidence", 0.0),
            created_at=datetime.fromisoformat(data["created_at"]),
            metadata=data.get("metadata", {}),
        )


def is_failure(trajectory: Trajectory, soft_threshold: float = 0.5) -> bool:
    """判断 Trajectory 是否为失败
    
    Args:
        trajectory: 待判断的轨迹
        soft_threshold: soft failure 的置信度阈值
        
    Returns:
        是否为失败
    """
    # Hard failure: 任务未完成
    if trajectory.result == "failure":
        return True
    
    # Soft failure: 低置信度 (从 metadata 中读取)
    confidence = trajectory.metadata.get("confidence", 1.0)
    if confidence < soft_threshold:
        return True
    
    # Judge 不通过 (从 metadata 中读取)
    if trajectory.metadata.get("judge_passed") is False:
        return True
    
    return False


def create_failure_case(
    trajectory: Trajectory,
    failure_reason: str = "",
    soft_threshold: float = 0.5,
) -> Optional[FailureCase]:
    """从 Trajectory 创建 FailureCase
    
    Args:
        trajectory: 轨迹
        failure_reason: 失败原因
        soft_threshold: soft failure 阈值
        
    Returns:
        FailureCase 或 None (如果不是失败)
    """
    if not is_failure(trajectory, soft_threshold):
        return None
    
    confidence = trajectory.metadata.get("confidence", 0.0)
    
    # 判断失败类型
    if trajectory.result == "failure":
        failure_type: FailureType = "hard"
    else:
        failure_type = "soft"
    
    return FailureCase(
        task=trajectory.task,
        trajectory=trajectory,
        failure_type=failure_type,
        failure_reason=failure_reason,
        confidence=confidence,
    )
