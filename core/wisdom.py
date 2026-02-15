"""
Wisdom - 智慧定义

L2 层存储的知识单元，可反查其 source trajectories。
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any
from datetime import datetime
import uuid
import json


@dataclass
class Wisdom:
    """智慧 - L2 层知识单元
    
    Attributes:
        id: 唯一标识符
        text: 智慧文本内容
        source_trajectory_ids: 来源轨迹 ID 列表 (可反查)
        created_at: 创建时间
        metadata: 额外元数据
    """
    text: str
    source_trajectory_ids: List[str] = field(default_factory=list)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_source(self, trajectory_id: str) -> None:
        """添加来源轨迹"""
        if trajectory_id not in self.source_trajectory_ids:
            self.source_trajectory_ids.append(trajectory_id)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "source_trajectory_ids": self.source_trajectory_ids,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Wisdom":
        return cls(
            id=data["id"],
            text=data["text"],
            source_trajectory_ids=data.get("source_trajectory_ids", []),
            created_at=datetime.fromisoformat(data["created_at"]),
            metadata=data.get("metadata", {}),
        )
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)
    
    @classmethod
    def from_json(cls, json_str: str) -> "Wisdom":
        return cls.from_dict(json.loads(json_str))
    
    def __repr__(self) -> str:
        return f"Wisdom(id={self.id[:8]}..., text={self.text[:50]}..., sources={len(self.source_trajectory_ids)})"
