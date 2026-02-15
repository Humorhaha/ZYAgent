"""
Beam Search Strategy - 束搜索策略 (对照组)

作为 MCTS 的对照实验。
"""

from typing import List, Tuple
import copy

from config.schema import System2Config
from core.failure import FailureCase
from core.trajectory import Trajectory, Step
from .base import SearchStrategy


class BeamSearchStrategy(SearchStrategy):
    """Beam Search 搜索策略"""
    
    def search(
        self, 
        failure_case: FailureCase, 
        config: System2Config,
    ) -> List[Trajectory]:
        """执行 Beam Search"""
        beam_width = config.branch
        max_depth = config.max_depth
        
        # 初始 beam: [(trajectory, score)]
        initial_traj = Trajectory(task=failure_case.task)
        current_beam: List[Tuple[Trajectory, float]] = [(initial_traj, 0.0)]
        
        final_trajectories: List[Tuple[Trajectory, float]] = []
        
        for depth in range(max_depth):
            candidates: List[Tuple[Trajectory, float]] = []
            
            for traj, score in current_beam:
                # 已终止的移入 final
                if traj.result in ["success", "failure"]:
                    final_trajectories.append((traj, score))
                    continue
                
                # 扩展 (Mock)
                next_steps = self._expand(traj, beam_width, depth)
                
                for next_traj, step_score in next_steps:
                    new_score = score + step_score
                    candidates.append((next_traj, new_score))
            
            if not candidates:
                break
            
            # 剪枝
            candidates.sort(key=lambda x: x[1], reverse=True)
            current_beam = candidates[:beam_width]
            
        # 合并剩余
        final_trajectories.extend(current_beam)
        final_trajectories.sort(key=lambda x: x[1], reverse=True)
        
        # 返回 Top-K
        result = []
        for traj, score in final_trajectories[:config.top_k]:
            traj.triggered_system2 = True
            traj.metadata["search_method"] = "beam_search"
            traj.metadata["beam_score"] = score
            result.append(traj)
            
        return result

    def _expand(self, traj: Trajectory, k: int, depth: int) -> List[Tuple[Trajectory, float]]:
        """扩展轨迹 (Mock)"""
        results = []
        for i in range(k):
            # 深拷贝
            new_traj = Trajectory(task=traj.task)
            new_traj.steps = [Step(**s.to_dict()) for s in traj.steps]
            
            step_text = f"Beam-D{depth}-B{i}"
            new_traj.add_step(
                thought=f"Thinking: {step_text}",
                action=f"Action: {step_text}",
                observation="Observation...",
            )
            
            score = 0.1 + (depth * 0.05)
            
            # 模拟终止
            if len(new_traj.steps) >= 4:
                if i == 0:
                    new_traj.mark_success()
                    score += 0.5
                else:
                    new_traj.mark_failure()
                
            results.append((new_traj, score))
        return results
