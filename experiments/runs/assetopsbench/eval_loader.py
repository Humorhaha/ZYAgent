"""
Evaluation Dataset Loader - AssetOpsBench 评估集加载器

从 HuggingFace 加载 IBM AssetOpsBench 评估数据集
"""

import json
import random
from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path
import urllib.request

DATASET_URL = "https://huggingface.co/datasets/ibm-research/AssetOpsBench/resolve/main/data/scenarios/all_utterance.jsonl"
CACHE_DIR = Path(__file__).parent / ".cache"


@dataclass
class EvalCase:
    """评估样本"""
    id: int
    type: str  # "IoT", "FMSA", etc.
    text: str  # 用户查询
    category: str  # "Knowledge Query", "Data Query", etc.
    deterministic: bool
    characteristic_form: str  # 期望的响应特征
    
    def __repr__(self):
        return f"EvalCase(id={self.id}, type={self.type}, text='{self.text[:50]}...')"


def download_dataset(force: bool = False) -> Path:
    """下载数据集到本地缓存"""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / "all_utterance.jsonl"
    
    if cache_file.exists() and not force:
        return cache_file
    
    print(f"[EvalLoader] Downloading dataset from HuggingFace...")
    urllib.request.urlretrieve(DATASET_URL, cache_file)
    print(f"[EvalLoader] Cached at {cache_file}")
    return cache_file


def load_all_cases() -> List[EvalCase]:
    """加载所有评估样本"""
    cache_file = download_dataset()
    cases = []
    
    with open(cache_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                cases.append(EvalCase(
                    id=data.get("id", 0),
                    type=data.get("type", ""),
                    text=data.get("text", ""),
                    category=data.get("category", ""),
                    deterministic=data.get("deterministic", True),
                    characteristic_form=data.get("characteristic_form", "")
                ))
    
    return cases


def sample_eval_cases(n: int = 5, seed: Optional[int] = None) -> List[EvalCase]:
    """
    随机采样 N 个评估样本
    
    Args:
        n: 采样数量
        seed: 随机种子 (可选, 用于复现)
        
    Returns:
        采样的 EvalCase 列表
    """
    cases = load_all_cases()
    
    if seed is not None:
        random.seed(seed)
    
    return random.sample(cases, min(n, len(cases)))


def sample_by_type(type_filter: str, n: int = 5) -> List[EvalCase]:
    """按类型采样 (IoT, FMSA, etc.)"""
    cases = [c for c in load_all_cases() if c.type == type_filter]
    return random.sample(cases, min(n, len(cases)))


def sample_by_category(category_filter: str, n: int = 5) -> List[EvalCase]:
    """按类别采样 (Knowledge Query, Data Query, etc.)"""
    cases = [c for c in load_all_cases() if c.category == category_filter]
    return random.sample(cases, min(n, len(cases)))


if __name__ == "__main__":
    # 测试: 采样 5 个样本
    samples = sample_eval_cases(5, seed=42)
    print("=" * 60)
    print("AssetOpsBench Evaluation Samples (n=5)")
    print("=" * 60)
    for s in samples:
        print(f"\n[{s.id}] ({s.type}/{s.category})")
        print(f"  Query: {s.text}")
        print(f"  Expected: {s.characteristic_form[:100]}...")
