"""
LLM Provider 实现

支持多种 LLM API:
- OpenAI (GPT-4/GPT-3.5)
- Anthropic (Claude)
- 自定义 API
"""

import os
from typing import Optional, Dict, List
from abc import ABC, abstractmethod


class BaseLLMProvider(ABC):
    """LLM Provider 基类"""
    
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """生成文本响应"""
        pass


class OpenAIProvider(BaseLLMProvider):
    """OpenAI API Provider (GPT-4/GPT-3.5)
    
    使用示例:
        provider = OpenAIProvider(api_key="sk-xxx", model="gpt-4")
        response = provider.generate("Hello, world!")
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        base_url: Optional[str] = None,
    ):
        """
        Args:
            api_key: OpenAI API Key (或从环境变量 OPENAI_API_KEY 读取)
            model: 模型名称 (gpt-4, gpt-4-turbo, gpt-3.5-turbo, 等)
            temperature: 生成温度 (0-2)
            max_tokens: 最大生成 Token 数
            base_url: 自定义 API 基础 URL (用于代理或兼容 API)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY env var or pass api_key.")
        
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = base_url
        
        # 延迟导入 openai
        try:
            import openai
            self._client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )
        except ImportError:
            raise ImportError("请安装 openai: pip install openai")
    
    def generate(self, prompt: str) -> str:
        """调用 OpenAI API 生成响应"""
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content or ""


class AnthropicProvider(BaseLLMProvider):
    """Anthropic API Provider (Claude)
    
    使用示例:
        provider = AnthropicProvider(api_key="sk-xxx", model="claude-3-opus-20240229")
        response = provider.generate("Hello, world!")
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-3-sonnet-20240229",
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ):
        """
        Args:
            api_key: Anthropic API Key (或从环境变量 ANTHROPIC_API_KEY 读取)
            model: 模型名称
            temperature: 生成温度
            max_tokens: 最大生成 Token 数
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key is required. Set ANTHROPIC_API_KEY env var or pass api_key.")
        
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        try:
            import anthropic
            self._client = anthropic.Anthropic(api_key=self.api_key)
        except ImportError:
            raise ImportError("请安装 anthropic: pip install anthropic")
    
    def generate(self, prompt: str) -> str:
        """调用 Anthropic API 生成响应"""
        message = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text


class DeepSeekProvider(BaseLLMProvider):
    """DeepSeek API Provider
    
    使用示例:
        provider = DeepSeekProvider(api_key="sk-xxx")
        response = provider.generate("Hello, world!")
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "deepseek-chat",
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("DeepSeek API key is required.")
        
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        try:
            import openai
            self._client = openai.OpenAI(
                api_key=self.api_key,
                base_url="https://api.deepseek.com",
            )
        except ImportError:
            raise ImportError("请安装 openai: pip install openai")
    
    def generate(self, prompt: str) -> str:
        """调用 DeepSeek API"""
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content or ""


class QwenProvider(BaseLLMProvider):
    """阿里云 DashScope Qwen API Provider
    
    使用示例:
        provider = QwenProvider(
            api_key="sk-xxx",
            model="qwen3-max",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        response = provider.generate("Hello, world!")
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "qwen3-max",
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ):
        """
        Args:
            api_key: DashScope API Key (或从环境变量 CLOUD_API_KEY 读取)
            model: 模型名称 (qwen3-max, qwen-turbo, etc.)
            base_url: API 基础 URL
            temperature: 生成温度
            max_tokens: 最大生成 Token 数
        """
        self.api_key = api_key or os.getenv("CLOUD_API_KEY")
        if not self.api_key:
            raise ValueError("Qwen API key is required. Set CLOUD_API_KEY env var or pass api_key.")
        
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        try:
            import openai
            self._client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )
        except ImportError:
            raise ImportError("请安装 openai: pip install openai")
    
    def generate(self, prompt: str) -> str:
        """调用 Qwen API"""
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content or ""


# 导出
__all__ = [
    "BaseLLMProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "DeepSeekProvider",
    "QwenProvider",
]
