"""
System 2 (Search) - 搜索策略统一接口

负责对 Failure Case 进行搜索，生成潜在的成功轨迹。
"""

from abc import ABC, abstractmethod
from typing import List

from config.schema import System2Config
from core.failure import FailureCase
from core.trajectory import Trajectory


class SearchStrategy(ABC):
    """搜索策略抽象基类
    
    所有搜索策略（MCTS、Beam Search、NoSearch 等）必须实现此接口。
    """
    
    @abstractmethod
    def search(
        self, 
        failure_case: FailureCase, 
        config: System2Config,
    ) -> List[Trajectory]:
        """执行搜索
        
        Args:
            failure_case: 失败案例 (搜索起点)
            config: System 2 搜索配置
            
        Returns:
            Top-K trajectories (按价值排序)
        """
        ...
