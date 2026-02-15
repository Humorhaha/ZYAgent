"""
Configuration Schema - 实验配置定义
消融实验
统一的配置系统，所有消融实验只需修改配置，不需改代码。
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Literal
import yaml
import json
from pathlib import Path


@dataclass
class System1Config:
    """System 1 配置"""
    use_wisdom: bool = True             # 是否使用 wisdom
    wisdom_k: int = 3                   # 注入 wisdom 数量
    max_steps: int = 20                 # 最大执行步数
    
    # Markov 约束
    markov_window: int = 1              # 1 = 1-step, k = k-step, -1 = full history


@dataclass
class GraphPersistConfig:
    """MCTS 图持久化配置"""
    enabled: bool = True                 # 是否启用图持久化
    backend: str = "neo4j"               # neo4j | none
    batch_size: int = 100                # 批量写入大小
    flush_interval_ms: int = 1000        # 刷新间隔 (ms)
    persist_stats: str = "snapshot"      # off | snapshot | periodic


@dataclass
class System2Config:
    """System 2 配置"""
    enabled: bool = True                # 是否启用 System2
    search_strategy: str = "mcts"       # mcts | beam | random | none
    top_k: int = 3                      # 输出 Top-K trajectories
    
    # MCTS 超参数
    branch: int = 3                     # expand 阶段的分支数
    roll_branch: int = 3                # rollout 阶段的分支数
    roll_forward_steps: int = 5         # rollout 前进步数
    exploration_constant: float = 1.414 # UCT 探索常数
    max_depth: int = 20                 # 最大搜索深度
    iteration_limit: int = 100          # 迭代次数限制
    
    # 策略配置
    use_reflection: str = "common"      # common | simple
    roll_policy: str = "greedy"         # greedy | random
    alpha: float = 0.5                  # value 更新混合系数
    
    # 图持久化 (Neo4j)
    graph_persist: GraphPersistConfig = field(default_factory=GraphPersistConfig)


@dataclass
class MemoryConfig:
    """Memory 配置"""
    enabled: bool = True                # 是否启用 Memory
    backend: str = "jsonl"              # jsonl | neo4j
    
    # Hot Start
    enable_hot_start: bool = True       # 是否启用 Hot Start
    
    # Promotion
    enable_promotion: bool = True       # 是否启用 Promotion
    promotion_min_samples: int = 5      # 触发 promotion 的最小样本数
    
    # L1/L2 配置
    l1_max_size: int = 1000             # L1 最大存储数
    l2_max_size: int = 500              # L2 最大存储数


@dataclass
class FailureConfig:
    """Failure Queue 配置"""
    soft_failure_threshold: float = 0.5  # soft failure 置信度阈值
    queue_max_size: int = 100            # 队列最大大小


@dataclass
class EvaluationConfig:
    """评测配置"""
    output_dir: str = "experiments/runs" # 输出目录


@dataclass
class TTSConfig:
    """TTS 持久化配置"""
    persist: bool = True                 # 是否启用持久化
    backend: str = "file"                # file | sqlite | object_store
    directory: str = "data/tts"          # 存储目录


@dataclass
class ExperimentConfig:
    """完整实验配置
    
    所有消融实验只需修改此配置，不需改代码。
    
    Example:
        # 加载配置
        config = ExperimentConfig.from_yaml("config.yaml")
        
        # 消融: 禁用 System2
        config.system2.enabled = False
        
        # 消融: 使用 Beam Search
        config.system2.search_strategy = "beam"
    """
    # 随机性控制
    seed: int = 42
    
    # 子配置
    system1: System1Config = field(default_factory=System1Config)
    system2: System2Config = field(default_factory=System2Config)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    failure: FailureConfig = field(default_factory=FailureConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    
    # 实验元数据
    experiment_name: str = "default"
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            "seed": self.seed,
            "experiment_name": self.experiment_name,
            "description": self.description,
            "system1": {
                "use_wisdom": self.system1.use_wisdom,
                "wisdom_k": self.system1.wisdom_k,
                "max_steps": self.system1.max_steps,
                "markov_window": self.system1.markov_window,
            },
            "system2": {
                "enabled": self.system2.enabled,
                "search_strategy": self.system2.search_strategy,
                "top_k": self.system2.top_k,
                "branch": self.system2.branch,
                "roll_branch": self.system2.roll_branch,
                "roll_forward_steps": self.system2.roll_forward_steps,
                "exploration_constant": self.system2.exploration_constant,
                "max_depth": self.system2.max_depth,
                "iteration_limit": self.system2.iteration_limit,
                "use_reflection": self.system2.use_reflection,
                "roll_policy": self.system2.roll_policy,
                "alpha": self.system2.alpha,
                "graph_persist": {
                    "enabled": self.system2.graph_persist.enabled,
                    "backend": self.system2.graph_persist.backend,
                    "batch_size": self.system2.graph_persist.batch_size,
                    "flush_interval_ms": self.system2.graph_persist.flush_interval_ms,
                    "persist_stats": self.system2.graph_persist.persist_stats,
                },
            },
            "memory": {
                "enabled": self.memory.enabled,
                "backend": self.memory.backend,
                "enable_hot_start": self.memory.enable_hot_start,
                "enable_promotion": self.memory.enable_promotion,
                "promotion_min_samples": self.memory.promotion_min_samples,
                "l1_max_size": self.memory.l1_max_size,
                "l2_max_size": self.memory.l2_max_size,
            },
            "failure": {
                "soft_failure_threshold": self.failure.soft_failure_threshold,
                "queue_max_size": self.failure.queue_max_size,
            },
            "evaluation": {
                "output_dir": self.evaluation.output_dir,
            },
            "tts": {
                "persist": self.tts.persist,
                "backend": self.tts.backend,
                "directory": self.tts.directory,
            },
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentConfig":
        """从字典反序列化"""
        config = cls()
        
        config.seed = data.get("seed", 42)
        config.experiment_name = data.get("experiment_name", "default")
        config.description = data.get("description", "")
        
        if "system1" in data:
            s1 = data["system1"]
            config.system1 = System1Config(
                use_wisdom=s1.get("use_wisdom", True),
                wisdom_k=s1.get("wisdom_k", 3),
                max_steps=s1.get("max_steps", 20),
                markov_window=s1.get("markov_window", 1),
            )
        
        if "system2" in data:
            s2 = data["system2"]
            # 解析 graph_persist 子配置
            gp = s2.get("graph_persist", {})
            graph_persist = GraphPersistConfig(
                enabled=gp.get("enabled", True),
                backend=gp.get("backend", "neo4j"),
                batch_size=gp.get("batch_size", 100),
                flush_interval_ms=gp.get("flush_interval_ms", 1000),
                persist_stats=gp.get("persist_stats", "snapshot"),
            )
            config.system2 = System2Config(
                enabled=s2.get("enabled", True),
                search_strategy=s2.get("search_strategy", "mcts"),
                top_k=s2.get("top_k", 3),
                branch=s2.get("branch", 3),
                roll_branch=s2.get("roll_branch", 3),
                roll_forward_steps=s2.get("roll_forward_steps", 5),
                exploration_constant=s2.get("exploration_constant", 1.414),
                max_depth=s2.get("max_depth", 20),
                iteration_limit=s2.get("iteration_limit", 100),
                use_reflection=s2.get("use_reflection", "common"),
                roll_policy=s2.get("roll_policy", "greedy"),
                alpha=s2.get("alpha", 0.5),
                graph_persist=graph_persist,
            )
        
        if "memory" in data:
            m = data["memory"]
            config.memory = MemoryConfig(
                enabled=m.get("enabled", True),
                backend=m.get("backend", "jsonl"),
                enable_hot_start=m.get("enable_hot_start", True),
                enable_promotion=m.get("enable_promotion", True),
                promotion_min_samples=m.get("promotion_min_samples", 5),
                l1_max_size=m.get("l1_max_size", 1000),
                l2_max_size=m.get("l2_max_size", 500),
            )
        
        if "failure" in data:
            f = data["failure"]
            config.failure = FailureConfig(
                soft_failure_threshold=f.get("soft_failure_threshold", 0.5),
                queue_max_size=f.get("queue_max_size", 100),
            )
        
        if "evaluation" in data:
            e = data["evaluation"]
            config.evaluation = EvaluationConfig(
                output_dir=e.get("output_dir", "experiments/runs"),
            )
        
        if "tts" in data:
            t = data["tts"]
            config.tts = TTSConfig(
                persist=t.get("persist", True),
                backend=t.get("backend", "file"),
                directory=t.get("directory", "data/tts"),
            )
        
        return config
    
    def to_yaml(self, path: str) -> None:
        """保存为 YAML 文件"""
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, allow_unicode=True)
    
    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        """从 YAML 文件加载"""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)
    
    def to_json(self) -> str:
        """序列化为 JSON"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> "ExperimentConfig":
        """从 JSON 反序列化"""
        return cls.from_dict(json.loads(json_str))


# =============================================================================
# 预设配置 (Ablation Presets)
# =============================================================================

def no_system2_config() -> ExperimentConfig:
    """消融: 禁用 System2"""
    config = ExperimentConfig()
    config.experiment_name = "ablation_no_system2"
    config.system2.enabled = False
    return config


def no_mcts_config() -> ExperimentConfig:
    """消融: 禁用 MCTS (仅 reflection)"""
    config = ExperimentConfig()
    config.experiment_name = "ablation_no_mcts"
    config.system2.search_strategy = "none"
    return config


def no_memory_config() -> ExperimentConfig:
    """消融: 禁用 Memory"""
    config = ExperimentConfig()
    config.experiment_name = "ablation_no_memory"
    config.memory.enabled = False
    return config


def no_promotion_config() -> ExperimentConfig:
    """消融: 禁用 Promotion"""
    config = ExperimentConfig()
    config.experiment_name = "ablation_no_promotion"
    config.memory.enable_promotion = False
    return config


def no_hot_start_config() -> ExperimentConfig:
    """消融: 禁用 Hot Start"""
    config = ExperimentConfig()
    config.experiment_name = "ablation_no_hot_start"
    config.memory.enable_hot_start = False
    return config


def no_markov_config() -> ExperimentConfig:
    """消融: 禁用 Markov (full history)"""
    config = ExperimentConfig()
    config.experiment_name = "ablation_no_markov"
    config.system1.markov_window = -1
    return config


def neo4j_backend_config() -> ExperimentConfig:
    """实验: 使用 Neo4j 后端"""
    config = ExperimentConfig()
    config.experiment_name = "exp_neo4j_backend"
    config.memory.backend = "neo4j"
    return config
