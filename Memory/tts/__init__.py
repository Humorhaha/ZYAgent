"""
Tiny Trajectory Stores (TTS) - ReactXen 短期记忆缓存

基于 ReactXen 架构设计的静态轨迹存储系统，用于提供高质量的 Few-Shot 示例。

核心设计理念:
- 高信噪比: 只存储成功的、经过蒸馏的优质轨迹
- Token 高效: 轨迹经过压缩，占用极少的 Context Window
- 针对性引导: 支持按类别/相似度检索最匹配的轨迹

与 HCC 的关系:
- TTS 是 "教科书": 提供静态的标准解题范例 (Read-Only)
- HCC 是 "笔记本": 管理动态的任务进度和经验 (Read-Write)
"""

from .tts import (
    TinyTrajectoryStore,
    Trajectory,
    TrajectoryStep,
    TrajectoryCategory,
)

__all__ = [
    # Core Classes
    "TinyTrajectoryStore",
    # Data Types
    "Trajectory",
    "TrajectoryStep",
    "TrajectoryCategory",
]
