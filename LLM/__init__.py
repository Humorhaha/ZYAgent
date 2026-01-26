"""LLM Package - 统一的 LLM 调用接口

提供:
    - LLM: 统一调用接口
    - LLMConfig: 配置类
    - create_llm: 便捷工厂函数
    - prompts: Prompt 模板模块
"""
from .llm import LLM, LLMConfig, create_llm
from . import prompts

__all__ = [
    "LLM",
    "LLMConfig",
    "create_llm",
    "prompts",
]
