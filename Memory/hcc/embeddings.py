"""
向量嵌入模块 - 支持L3 Prior Wisdom的语义检索

论文中L3缓存使用embedding进行相似度检索:
    h_n = E(d_n)  # d_n是任务描述，h_n是嵌入向量
    检索: cos(q, h_n) > δ  # 基于余弦相似度阈值检索
"""

from abc import ABC, abstractmethod
from typing import List
import numpy as np


class EmbeddingProvider(ABC):
    """嵌入向量提供者的抽象基类
    
    子类可实现不同的嵌入后端:
    - OpenAI Embedding API
    - Sentence-Transformers
    - 本地向量模型
    """
    
    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        """将文本编码为向量
        
        Args:
            text: 输入文本
            
        Returns:
            嵌入向量 (归一化后)
        """
        pass
    
    @abstractmethod
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """批量编码文本
        
        Args:
            texts: 输入文本列表
            
        Returns:
            嵌入向量矩阵 shape: (N, dim)
        """
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """嵌入向量的维度"""
        pass


class SimpleEmbeddingProvider(EmbeddingProvider):
    """简单的嵌入实现 - 基于词袋的TF-IDF风格向量
    
    仅用于测试和演示，生产环境应使用更强的模型。
    """
    
    def __init__(self, dim: int = 256, seed: int = 42):
        """
        Args:
            dim: 向量维度
            seed: 随机种子，确保相同文本产生相同向量
        """
        self._dim = dim
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        # 缓存已计算的嵌入，确保一致性
        self._cache: dict[str, np.ndarray] = {}
    
    @property
    def dimension(self) -> int:
        return self._dim
    
    def embed(self, text: str) -> np.ndarray:
        """基于文本hash生成伪随机但确定性的向量"""
        if text in self._cache:
            return self._cache[text]
        
        # 使用文本hash作为种子，确保相同文本产生相同向量
        text_hash = hash(text) & 0xFFFFFFFF
        local_rng = np.random.default_rng(text_hash)
        vec = local_rng.standard_normal(self._dim).astype(np.float32)
        
        # L2归一化，便于余弦相似度计算
        vec = vec / (np.linalg.norm(vec) + 1e-8)
        
        self._cache[text] = vec
        return vec
    
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """批量编码"""
        return np.stack([self.embed(t) for t in texts])


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """计算两个向量的余弦相似度
    
    论文公式: cos(q, h_n) > δ
    
    Args:
        a, b: 已归一化的向量
        
    Returns:
        余弦相似度 [-1, 1]
    """
    return float(np.dot(a, b))


def batch_cosine_similarity(query: np.ndarray, keys: np.ndarray) -> np.ndarray:
    """批量计算余弦相似度
    
    Args:
        query: 查询向量 shape: (dim,)
        keys: 键向量矩阵 shape: (N, dim)
        
    Returns:
        相似度数组 shape: (N,)
    """
    return keys @ query
