"""
LLM - 统一的 LLM 调用接口

统一使用 OpenAI 兼容接口，支持:
    - OpenAI API
    - DeepSeek API
    - Qwen (DashScope) API
    - 任何 OpenAI 兼容的 API

配置方式:
    1. 创建 LLM/.env 文件 (参考 .env.example)
    2. 设置 OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass

# 加载 .env 文件
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass  # dotenv 未安装时跳过

# 导入 Prompt 模板
from . import prompts


# =============================================================================
# 配置
# =============================================================================

@dataclass
class LLMConfig:
    """LLM 配置"""
    api_key: str = ""
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: int = 2048
    
    @classmethod
    def from_env(cls) -> "LLMConfig":
        """从环境变量加载配置"""
        return cls(
            api_key=os.getenv("OPENAI_API_KEY", ""),
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            model=os.getenv("OPENAI_MODEL", "gpt-4"),
        )


# =============================================================================
# LLM 客户端
# =============================================================================

class LLM:
    """统一的 LLM 调用接口
    
    使用 OpenAI 兼容 API，支持多种 LLM 服务。
    
    Example:
        >>> llm = LLM()  # 从环境变量加载配置
        >>> response = llm.generate("Hello, world!")
        >>> 
        >>> # 使用内置 Prompt
        >>> summary = llm.compress_trajectory(
        ...     current="Thought: check file...",
        ...     previous="Previously tried A, failed."
        ... )
    """
    
    def __init__(
        self,
        config: Optional[LLMConfig] = None,
        mock_mode: bool = False,
    ):
        """
        Args:
            config: LLM 配置，None 则从环境变量加载
            mock_mode: 是否使用 Mock 模式 (无 API 调用)
        """
        self.config = config or LLMConfig.from_env()
        self._mock_mode = mock_mode
        self._client = None
        
        # 如果没有 API key，自动进入 Mock 模式
        if not self.config.api_key:
            self._mock_mode = True
            print("[LLM] No API key found, using mock mode")
    
    def _get_client(self):
        """延迟初始化 OpenAI 客户端"""
        if self._client is None and not self._mock_mode:
            try:
                import openai
                self._client = openai.OpenAI(
                    api_key=self.config.api_key,
                    base_url=self.config.base_url,
                )
            except ImportError:
                print("[LLM] OpenAI package not installed, using mock mode")
                self._mock_mode = True
        return self._client
    
    def generate(self, prompt: str) -> str:
        """生成文本响应
        
        Args:
            prompt: 完整的 Prompt
            
        Returns:
            LLM 响应文本
        """
        if self._mock_mode:
            return "[Mock Response]"
        
        client = self._get_client()
        if client is None:
            return "[Mock Response]"
        
        try:
            response = client.chat.completions.create(
                model=self.config.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            print(f"[LLM] API error: {e}")
            return f"[Error: {e}]"
    
    # =========================================================================
    # 便捷方法 (使用 prompts.py 中的模板)
    # =========================================================================
    
    def compress_trajectory(
        self,
        current_trajectory: str,
        previous_summary: str = "",
    ) -> str:
        """压缩 TTS Buffer 轨迹
        
        将当前轨迹与历史摘要合并，生成新的压缩摘要。
        
        Args:
            current_trajectory: 当前轮的执行轨迹
            previous_summary: 之前轮的压缩摘要
            
        Returns:
            新的压缩摘要
        """
        prompt = prompts.TTS_COMPRESS.format(
            current_trajectory=current_trajectory,
            previous_summary=previous_summary or "(无历史摘要)",
        )
        
        if self._mock_mode:
            # Mock 模式: 简单保留最新轨迹
            if previous_summary:
                truncated = previous_summary[:150] + "..." if len(previous_summary) > 150 else previous_summary
                return f"[历史] {truncated}\n[最新] {current_trajectory}"
            return current_trajectory
        
        return self.generate(prompt).strip()
    
    def distill_knowledge(
        self,
        raw_experience: str,
        task_type: str = "general",
    ) -> str:
        """提炼知识 (L1 -> L2)
        
        Args:
            raw_experience: 原始经验
            task_type: 任务类型
            
        Returns:
            提炼后的知识 (一句话)
        """
        prompt = prompts.KNOWLEDGE_DISTILL.format(
            raw_experience=raw_experience,
            task_type=task_type,
        )
        return self.generate(prompt).strip()
    
    def generate_wisdom(
        self,
        task_descriptor: str,
        success_experience: str,
    ) -> str:
        """生成 Wisdom (L2 -> L3)
        
        Args:
            task_descriptor: 任务描述
            success_experience: 成功经验
            
        Returns:
            可复用的 Wisdom
        """
        prompt = prompts.WISDOM_GENERATION.format(
            task_descriptor=task_descriptor,
            success_experience=success_experience,
        )
        return self.generate(prompt).strip()
    
    def analyze_failure(
        self,
        task: str,
        trajectory: str,
        failure_reason: str,
    ) -> str:
        """分析失败原因 (Reflect Agent)
        
        Args:
            task: 任务描述
            trajectory: 执行轨迹
            failure_reason: 失败原因
            
        Returns:
            分析结果和改进建议
        """
        prompt = prompts.REFLECT_ANALYSIS.format(
            task=task,
            trajectory=trajectory,
            failure_reason=failure_reason,
        )
        return self.generate(prompt).strip()
    
    def evaluate_action(
        self,
        current_state: str,
        action: str,
        next_state: str,
    ) -> tuple[float, str]:
        """评估动作的策略价值 (SPIRAL Critic)
        
        Args:
            current_state: 当前状态描述
            action: 执行的动作
            next_state: 动作后的新状态
            
        Returns:
            (score, reasoning) 元组，score 在 [0, 1] 范围
        """
        prompt = prompts.CRITIC_EVALUATE.format(
            current_state=current_state,
            action=action,
            next_state=next_state,
        )
        
        if self._mock_mode:
            # Mock 模式: 返回默认分数
            return (0.5, "[Mock] 默认评估分数")
        
        response = self.generate(prompt).strip()
        
        # 解析 JSON 响应
        try:
            import json
            import re
            # 尝试提取 JSON 块
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                score = float(data.get("score", 0.5))
                reasoning = data.get("reasoning", response)
                return (max(0.0, min(1.0, score)), reasoning)
        except (json.JSONDecodeError, ValueError, KeyError):
            pass
        
        # 解析失败时返回默认值
        return (0.5, f"[解析失败] {response}")
    
    @property
    def is_mock(self) -> bool:
        """是否处于 Mock 模式"""
        return self._mock_mode


# =============================================================================
# 工厂函数
# =============================================================================

def create_llm(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    mock_mode: bool = False,
) -> LLM:
    """创建 LLM 实例的便捷函数
    
    Args:
        api_key: API Key (可选，默认从环境变量)
        base_url: API URL (可选，默认从环境变量)
        model: 模型名称 (可选，默认从环境变量)
        mock_mode: 是否使用 Mock 模式
        
    Returns:
        LLM 实例
    """
    if mock_mode:
        return LLM(mock_mode=True)
    
    config = LLMConfig.from_env()
    if api_key:
        config.api_key = api_key
    if base_url:
        config.base_url = base_url
    if model:
        config.model = model
    
    return LLM(config=config)


# =============================================================================
# 导出
# =============================================================================

__all__ = [
    "LLM",
    "LLMConfig",
    "create_llm",
]
