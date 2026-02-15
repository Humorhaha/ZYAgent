"""
No-Search Strategy - 无搜索策略 (对照组)

仅进行 Reflection，不进行多步搜索。
"""

from typing import List

from config.schema import System2Config
from core.failure import FailureCase
from core.trajectory import Trajectory
from .base import SearchStrategy


class NoSearchStrategy(SearchStrategy):
    """Reflect Only 策略 (不搜索)"""
    
    def search(
        self, 
        failure_case: FailureCase, 
        config: System2Config,
    ) -> List[Trajectory]:
        """仅执行反思，不搜索"""
        
        # 1. 创建新轨迹
        new_traj = Trajectory(task=failure_case.task)
        
        # 2. 生成反思 (Mock)
        reflection = self._generate_reflection(failure_case)
        
        new_traj.add_step(
            thought=reflection,
            action="Retry",
            observation="Attempting retry with reflection...",
            metadata={"reflection": reflection}
        )
        
        # 3. 模拟一次重试 (Mock)
        new_traj.add_step(
            thought="Applying reflection insights...",
            action="Finish",
            observation="Completed with reflection guidance.",
        )
        
        # Mock: 假设反思后成功
        new_traj.mark_success()
        
        # 4. 标记来源
        new_traj.triggered_system2 = True
        new_traj.metadata["search_method"] = "no_search"
        new_traj.metadata["original_failure_reason"] = failure_case.failure_reason
        
        return [new_traj]
    
    def _generate_reflection(self, failure_case: FailureCase) -> str:
        """生成反思 (Mock，实际需调用 LLM)"""
        return (
            f"Reflection on failure:\n"
            f"- Task: {failure_case.task[:50]}...\n"
            f"- Failure Type: {failure_case.failure_type}\n"
            f"- Reason: {failure_case.failure_reason}\n"
            f"- Insight: Should try a different approach..."
        )
