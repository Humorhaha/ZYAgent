"""
Pipeline Module - 主控模块

导出 Pipeline Orchestrator (异步版本)。
"""

from .async_orchestrator import (
    AsyncPipelineOrchestrator,
    BackgroundSystem2,
    FailureQueue,
)

# 主导出
PipelineOrchestrator = AsyncPipelineOrchestrator  # 默认使用异步版本

__all__ = [
    "PipelineOrchestrator",
    "AsyncPipelineOrchestrator",
    "BackgroundSystem2",
    "FailureQueue",
]
