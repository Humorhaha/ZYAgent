"""
Core Data Structures - 核心数据结构

定义论文框架中的最小研究单位：Trajectory, Step, Cost
所有模块只能读/写 Trajectory，不允许隐式状态。
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Literal, Optional
from datetime import datetime
import uuid
import json


@dataclass
class Step:
    """单步执行记录
    
    Attributes:
        thought: 思考过程
        action: 执行的动作
        observation: 观察结果
        metadata: 额外元数据
    """
    thought: str
    action: str
    observation: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "thought": self.thought,
            "action": self.action,
            "observation": self.observation,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Step":
        return cls(
            thought=data["thought"],
            action=data["action"],
            observation=data["observation"],
            metadata=data.get("metadata", {}),
        )
    
    def __str__(self) -> str:
        return f"Thought: {self.thought}\nAction: {self.action}\nObservation: {self.observation}"


@dataclass
class TrajectoryCost:
    """轨迹成本统计
    
    Attributes:
        tokens: 消耗的 token 数
        steps: 执行的步数
        tool_calls: 工具调用次数
    """
    tokens: int = 0
    steps: int = 0
    tool_calls: int = 0
    
    def to_dict(self) -> Dict[str, int]:
        return {
            "tokens": self.tokens,
            "steps": self.steps,
            "tool_calls": self.tool_calls,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, int]) -> "TrajectoryCost":
        return cls(
            tokens=data.get("tokens", 0),
            steps=data.get("steps", 0),
            tool_calls=data.get("tool_calls", 0),
        )


@dataclass
class Trajectory:
    """轨迹 - 最小研究单位
    
    任何模块都只能读/写 Trajectory，不允许隐式状态。
    
    Attributes:
        id: 唯一标识符
        task: 原始任务描述
        steps: 执行步骤列表
        result: 执行结果 (success/failure)
        cost: 成本统计
        
        # 追溯字段 (Lineage)
        triggered_system2: 是否触发了 System2
        used_wisdom: 是否使用了 wisdom
        wisdom_ids: 使用的 wisdom ID 列表
        
        # 新增字段 (Data Contract v2)
        task_id: 任务唯一标识 (用于分组)
        run_id: 运行批次标识
        prompt_hash: Prompt 哈希 (用于去重)
        model: 使用的模型名称
        metrics: 评测指标字典
        final_answer: 最终答案
        
        # 时间戳
        created_at: 创建时间
    """
    task: str
    steps: List[Step] = field(default_factory=list)
    result: Literal["success", "failure", "pending"] = "pending"
    cost: TrajectoryCost = field(default_factory=TrajectoryCost)
    
    # 追溯字段
    triggered_system2: bool = False
    used_wisdom: bool = False
    wisdom_ids: List[str] = field(default_factory=list)
    
    # 新增字段 (Data Contract v2)
    task_id: Optional[str] = None
    run_id: Optional[str] = None
    prompt_hash: Optional[str] = None
    model: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    final_answer: Optional[str] = None
    
    # 元数据
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_step(self, thought: str, action: str, observation: str, **metadata) -> None:
        """添加一个执行步骤"""
        step = Step(thought=thought, action=action, observation=observation, metadata=metadata)
        self.steps.append(step)
        self.cost.steps += 1
        if action and action.lower() not in ["think", "finish"]:
            self.cost.tool_calls += 1
    
    def mark_success(self) -> None:
        """标记为成功"""
        self.result = "success"
    
    def mark_failure(self) -> None:
        """标记为失败"""
        self.result = "failure"
    
    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            "id": self.id,
            "task": self.task,
            "task_id": self.task_id,
            "run_id": self.run_id,
            "steps": [s.to_dict() for s in self.steps],
            "result": self.result,
            "final_answer": self.final_answer,
            "cost": self.cost.to_dict(),
            "metrics": self.metrics,
            "triggered_system2": self.triggered_system2,
            "used_wisdom": self.used_wisdom,
            "wisdom_ids": self.wisdom_ids,
            "prompt_hash": self.prompt_hash,
            "model": self.model,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Trajectory":
        """从字典反序列化"""
        return cls(
            id=data["id"],
            task=data["task"],
            task_id=data.get("task_id"),
            run_id=data.get("run_id"),
            steps=[Step.from_dict(s) for s in data["steps"]],
            result=data["result"],
            final_answer=data.get("final_answer"),
            cost=TrajectoryCost.from_dict(data["cost"]),
            metrics=data.get("metrics", {}),
            triggered_system2=data.get("triggered_system2", False),
            used_wisdom=data.get("used_wisdom", False),
            wisdom_ids=data.get("wisdom_ids", []),
            prompt_hash=data.get("prompt_hash"),
            model=data.get("model"),
            created_at=datetime.fromisoformat(data["created_at"]),
            metadata=data.get("metadata", {}),
        )
    
    def to_json(self) -> str:
        """序列化为 JSON 字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False)
    
    @classmethod
    def from_json(cls, json_str: str) -> "Trajectory":
        """从 JSON 字符串反序列化"""
        return cls.from_dict(json.loads(json_str))
    
    def __repr__(self) -> str:
        return f"Trajectory(id={self.id[:8]}..., task={self.task[:30]}..., result={self.result}, steps={len(self.steps)})"
