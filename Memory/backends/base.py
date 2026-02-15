"""
Memory Backend - 存储后端抽象接口

定义统一的 Memory 存储接口，供 JSONL 和 Neo4j 两个后端实现。
两个后端在相同 config + seed 下必须输出等价 wisdom。
"""

from abc import ABC, abstractmethod
from typing import List, Optional
from core.trajectory import Trajectory
from core.wisdom import Wisdom


class MemoryBackend(ABC):
    """Memory 存储后端抽象接口
    
    研究目标:
        - 验证是否更利于 trajectory 去重
        - 验证是否更利于 wisdom 证据追溯
        
    约束:
        - put_samples(): L1 存储
        - promote(): L1 -> L2 提升
        - retrieve(): L2 检索
    """
    
    @abstractmethod
    def put_samples(self, trajectories: List[Trajectory]) -> None:
        """存储 trajectories 到 L1
        
        Args:
            trajectories: 轨迹列表 (failure 或 success)
        """
        ...
    
    @abstractmethod
    def get_samples(self, limit: int = 100) -> List[Trajectory]:
        """获取 L1 中的 samples
        
        Args:
            limit: 最大返回数量
            
        Returns:
            轨迹列表
        """
        ...
    
    @abstractmethod
    def promote(self, min_samples: int = 5) -> List[Wisdom]:
        """从 L1 提升到 L2
        
        Args:
            min_samples: 触发 promotion 的最小样本数
            
        Returns:
            新生成的 Wisdom 列表
        """
        ...
    
    @abstractmethod
    def retrieve(self, task: str, k: int = 3) -> List[Wisdom]:
        """从 L2 检索 Wisdom
        
        Args:
            task: 任务描述 (用于相似度匹配)
            k: 返回数量
            
        Returns:
            相关的 Wisdom 列表
        """
        ...
    
    @abstractmethod
    def get_wisdom_by_id(self, wisdom_id: str) -> Optional[Wisdom]:
        """根据 ID 获取 Wisdom (用于溯源)
        
        Args:
            wisdom_id: Wisdom ID
            
        Returns:
            Wisdom 或 None
        """
        ...
    
    @abstractmethod
    def get_trajectory_by_id(self, trajectory_id: str) -> Optional[Trajectory]:
        """根据 ID 获取 Trajectory (用于溯源)
        
        Args:
            trajectory_id: Trajectory ID
            
        Returns:
            Trajectory 或 None
        """
        ...
    
    @abstractmethod
    def get_all_wisdom(self) -> List[Wisdom]:
        """获取所有 Wisdom (调试用)"""
        ...
    
    @abstractmethod
    def clear(self) -> None:
        """清空所有数据"""
        ...
    
    @abstractmethod
    def save(self) -> None:
        """持久化到存储"""
        ...
    
    @abstractmethod
    def load(self) -> None:
        """从存储加载"""
        ...
