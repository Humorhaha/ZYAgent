"""
JSONL Memory Backend - Baseline 存储实现

使用 JSONL 文件存储 L1 Trajectories 和 L2 Wisdom。
提供基础的基于文本匹配的检索和简单的 Promotion 逻辑。
"""

import os
import json
from typing import List, Optional, Dict, Any
from pathlib import Path
from collections import defaultdict

from core.trajectory import Trajectory
from core.wisdom import Wisdom
from .base import MemoryBackend


class JsonlBackend(MemoryBackend):
    """JSONL 文件存储后端
    
    提供基础的 L1/L2 存储、检索和 Promotion 功能。
    作为实验的 Baseline 存储方案。
    
    Attributes:
        data_dir: 数据存储目录
        l1_file: L1 样本存储路径
        l2_file: L2 智慧存储路径
    """
    
    def __init__(self, data_dir: str = "data/memory"):
        self.data_dir = Path(data_dir)
        self.l1_file = self.data_dir / "l1_samples.jsonl"
        self.l2_file = self.data_dir / "l2_wisdom.jsonl"
        
        # 内存缓存
        self.l1_samples: List[Trajectory] = []
        self.l2_wisdom: List[Wisdom] = []
        
        # ID 索引（加速查找）
        self._traj_index: Dict[str, Trajectory] = {}
        self._wisdom_index: Dict[str, Wisdom] = {}
        
        # 确保目录存在
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载已有数据
        self.load()
    
    # =========================================================================
    # 核心接口实现
    # =========================================================================
    
    def put_samples(self, trajectories: List[Trajectory]) -> None:
        """存储 trajectories 到 L1"""
        for traj in trajectories:
            # 检查重复
            if traj.id not in self._traj_index:
                self.l1_samples.append(traj)
                self._traj_index[traj.id] = traj
        self.save_l1()
    
    def get_samples(self, limit: int = 100) -> List[Trajectory]:
        """获取 L1 中的 samples"""
        return self.l1_samples[-limit:]
    
    def promote(self, min_samples: int = 5) -> List[Wisdom]:
        """从 L1 提升到 L2
        
        策略:
        1. 按 task 对 L1 samples 进行分组
        2. 如果某任务的样本数 >= min_samples，则触发 promotion
        3. 生成 Wisdom (此处为简化实现，实际应调用 LLM 进行总结)
        
        Args:
            min_samples: 触发 promotion 的最小样本数
            
        Returns:
            新生成的 Wisdom 列表
        """
        # 1. 按 task 分组（只考虑尚未被 promote 的轨迹）
        task_groups: Dict[str, List[Trajectory]] = defaultdict(list)
        
        for traj in self.l1_samples:
            if not self._is_traj_promoted(traj.id):
                task_groups[traj.task].append(traj)
        
        new_wisdoms = []
        
        # 2. 检查阈值并生成 Wisdom
        for task, group in task_groups.items():
            if len(group) >= min_samples:
                # 生成 Wisdom 文本（Mock 实现，实际需 LLM）
                success_count = sum(1 for t in group if t.result == "success")
                failure_count = len(group) - success_count
                
                wisdom_text = (
                    f"[Wisdom for task: {task[:50]}...]\n"
                    f"Based on {len(group)} trajectories "
                    f"({success_count} success, {failure_count} failure).\n"
                    f"Key insight: Review observations carefully before acting."
                )
                
                source_ids = [t.id for t in group]
                wisdom = Wisdom(
                    text=wisdom_text,
                    source_trajectory_ids=source_ids,
                    metadata={
                        "promoted_from_task": task,
                        "sample_count": len(group),
                        "success_rate": success_count / len(group) if group else 0,
                    }
                )
                
                self.l2_wisdom.append(wisdom)
                self._wisdom_index[wisdom.id] = wisdom
                new_wisdoms.append(wisdom)
        
        if new_wisdoms:
            self.save_l2()
            
        return new_wisdoms

    def _is_traj_promoted(self, traj_id: str) -> bool:
        """检查轨迹是否已作为 Source 生成过 Wisdom"""
        for w in self.l2_wisdom:
            if traj_id in w.source_trajectory_ids:
                return True
        return False
    
    def retrieve(self, task: str, k: int = 3) -> List[Wisdom]:
        """从 L2 检索 Wisdom
        
        Baseline 实现：基于简单的文本匹配评分。
        
        Args:
            task: 任务描述（用于相似度匹配）
            k: 返回数量
            
        Returns:
            相关的 Wisdom 列表
        """
        if not self.l2_wisdom:
            return []
        
        candidates = []
        task_lower = task.lower()
        
        for w in self.l2_wisdom:
            score = 0.0
            
            # 精确任务匹配
            promoted_task = w.metadata.get("promoted_from_task", "")
            if promoted_task == task:
                score = 100.0
            elif promoted_task.lower() in task_lower or task_lower in promoted_task.lower():
                score = 50.0
            
            # 文本词汇重叠
            task_words = set(task_lower.split())
            wisdom_words = set(w.text.lower().split())
            overlap = len(task_words & wisdom_words)
            score += overlap * 2
            
            # 成功率加权
            success_rate = w.metadata.get("success_rate", 0)
            score += success_rate * 10
            
            if score > 0:
                candidates.append((score, w))
        
        # 按分数降序排序
        candidates.sort(key=lambda x: x[0], reverse=True)
        return [c[1] for c in candidates[:k]]
    
    # =========================================================================
    # ID 查询（溯源支持）
    # =========================================================================
    
    def get_wisdom_by_id(self, wisdom_id: str) -> Optional[Wisdom]:
        return self._wisdom_index.get(wisdom_id)
    
    def get_trajectory_by_id(self, trajectory_id: str) -> Optional[Trajectory]:
        return self._traj_index.get(trajectory_id)
    
    def get_all_wisdom(self) -> List[Wisdom]:
        return list(self.l2_wisdom)
    
    # =========================================================================
    # 持久化
    # =========================================================================
    
    def clear(self) -> None:
        """清空所有数据"""
        self.l1_samples = []
        self.l2_wisdom = []
        self._traj_index = {}
        self._wisdom_index = {}
        
        if self.l1_file.exists():
            self.l1_file.unlink()
        if self.l2_file.exists():
            self.l2_file.unlink()
    
    def save(self) -> None:
        """保存所有数据"""
        self.save_l1()
        self.save_l2()
    
    def save_l1(self) -> None:
        """保存 L1 样本"""
        with open(self.l1_file, "w", encoding="utf-8") as f:
            for traj in self.l1_samples:
                f.write(traj.to_json() + "\n")
                
    def save_l2(self) -> None:
        """保存 L2 智慧"""
        with open(self.l2_file, "w", encoding="utf-8") as f:
            for w in self.l2_wisdom:
                f.write(w.to_json() + "\n")
    
    def load(self) -> None:
        """从文件加载数据"""
        # L1
        self.l1_samples = []
        self._traj_index = {}
        if self.l1_file.exists():
            with open(self.l1_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        traj = Trajectory.from_json(line)
                        self.l1_samples.append(traj)
                        self._traj_index[traj.id] = traj
        
        # L2
        self.l2_wisdom = []
        self._wisdom_index = {}
        if self.l2_file.exists():
            with open(self.l2_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        w = Wisdom.from_json(line)
                        self.l2_wisdom.append(w)
                        self._wisdom_index[w.id] = w
