"""
Tiny Trajectory Stores (TTS) - 核心实现模块

设计理念:
    TTS 是一个只读的、精炼的轨迹缓存，用于为 Agent 提供高质量的 Few-Shot 示例。
    它解决的是 "如何教 Agent 正确的推理模式" 的问题。
    
核心特性:
    1. 轨迹存储: 支持 JSON/文本格式的轨迹加载
    2. 分类管理: 按领域/任务类型分类存储轨迹
    3. 语义检索: 基于 Embedding 的相似度检索 (可选)
    4. Token 预算: 支持按 Token 限制截断轨迹
    
数据结构:
    Trajectory = {
        "id": "unique_id",
        "category": "data_science",
        "task": "问题描述",
        "steps": [
            {"thought": "...", "action": "...", "observation": "..."},
            ...
        ],
        "final_answer": "最终答案",
        "metadata": {"source": "...", "quality_score": 0.95}
    }
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Union
from enum import Enum
from pathlib import Path
import json
import re


class TrajectoryCategory(Enum):
    """轨迹类别 - 按任务类型分类
    
    使用示例:
        store.add(trajectory, category=TrajectoryCategory.DATA_SCIENCE)
        examples = store.retrieve(query, category=TrajectoryCategory.REASONING)
    """
    GENERAL = "general"              # 通用推理
    DATA_SCIENCE = "data_science"    # 数据科学/ML
    CODE_GENERATION = "code_gen"     # 代码生成
    REASONING = "reasoning"          # 逻辑推理
    QA = "qa"                        # 问答
    TOOL_USE = "tool_use"            # 工具调用
    REFLECTION = "reflection"        # 自我反思/纠错


@dataclass
class TrajectoryStep:
    """轨迹步骤 - 单个 Thought-Action-Observation 单元
    
    对应 ReAct 范式中的一个完整执行周期。
    
    Attributes:
        step_id: 步骤序号 (1-indexed)
        thought: Agent 的思考过程
        action: 执行的动作名称
        action_input: 动作的输入参数
        observation: 环境返回的观察结果
    """
    step_id: int
    thought: str
    action: str
    action_input: str = ""
    observation: str = ""
    
    def to_text(self, include_observation: bool = True) -> str:
        """转换为文本格式，用于 Prompt 注入
        
        Args:
            include_observation: 是否包含 Observation (最后一步可能没有)
            
        Returns:
            格式化的文本，如:
            Thought 1: I need to search for...
            Action 1: Search
            Action Input 1: query
            Observation 1: Results...
        """
        lines = [
            f"Thought {self.step_id}: {self.thought}",
            f"Action {self.step_id}: {self.action}",
        ]
        if self.action_input:
            lines.append(f"Action Input {self.step_id}: {self.action_input}")
        if include_observation and self.observation:
            lines.append(f"Observation {self.step_id}: {self.observation}")
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式，用于 JSON 序列化"""
        return {
            "step_id": self.step_id,
            "thought": self.thought,
            "action": self.action,
            "action_input": self.action_input,
            "observation": self.observation,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrajectoryStep":
        """从字典创建实例"""
        return cls(
            step_id=data.get("step_id", 0),
            thought=data.get("thought", ""),
            action=data.get("action", ""),
            action_input=data.get("action_input", ""),
            observation=data.get("observation", ""),
        )


@dataclass
class Trajectory:
    """完整轨迹 - 一个任务的完整执行过程
    
    Attributes:
        trajectory_id: 唯一标识符
        category: 轨迹类别
        task: 任务描述/问题
        steps: 执行步骤列表
        final_answer: 最终答案
        metadata: 附加元数据 (来源、质量分数等)
    """
    trajectory_id: str
    category: TrajectoryCategory
    task: str
    steps: List[TrajectoryStep]
    final_answer: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # 用于语义检索的嵌入向量 (可选)
    _embedding: Optional[Any] = field(default=None, repr=False)
    
    def to_text(self, max_steps: Optional[int] = None) -> str:
        """转换为完整的文本格式，用于 Prompt 注入
        
        Args:
            max_steps: 最大步骤数限制，None 表示全部
            
        Returns:
            格式化的完整轨迹文本
        """
        lines = [f"Question: {self.task}"]
        
        steps_to_use = self.steps[:max_steps] if max_steps else self.steps
        for i, step in enumerate(steps_to_use):
            # 最后一步可能不需要 observation (如果是 Finish action)
            include_obs = i < len(steps_to_use) - 1 or step.observation
            lines.append(step.to_text(include_observation=include_obs))
        
        if self.final_answer:
            lines.append(f"Final Answer: {self.final_answer}")
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "trajectory_id": self.trajectory_id,
            "category": self.category.value,
            "task": self.task,
            "steps": [s.to_dict() for s in self.steps],
            "final_answer": self.final_answer,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Trajectory":
        """从字典创建实例"""
        category_value = data.get("category", "general")
        try:
            category = TrajectoryCategory(category_value)
        except ValueError:
            category = TrajectoryCategory.GENERAL
            
        return cls(
            trajectory_id=data.get("trajectory_id", ""),
            category=category,
            task=data.get("task", ""),
            steps=[TrajectoryStep.from_dict(s) for s in data.get("steps", [])],
            final_answer=data.get("final_answer", ""),
            metadata=data.get("metadata", {}),
        )
    
    @classmethod
    def from_text(
        cls, 
        text: str, 
        trajectory_id: str = "",
        category: TrajectoryCategory = TrajectoryCategory.GENERAL
    ) -> "Trajectory":
        """从文本格式解析轨迹
        
        支持解析标准的 ReAct 格式文本:
            Question: ...
            Thought 1: ...
            Action 1: ...
            ...
            
        Args:
            text: 原始文本
            trajectory_id: 轨迹ID
            category: 轨迹类别
            
        Returns:
            解析后的 Trajectory 对象
        """
        # 提取 Question
        task_match = re.search(r"Question:\s*(.+?)(?=\nThought|\Z)", text, re.DOTALL)
        task = task_match.group(1).strip() if task_match else ""
        
        # 提取 Final Answer
        answer_match = re.search(r"Final Answer:\s*(.+?)(?=\n|$)", text, re.DOTALL)
        final_answer = answer_match.group(1).strip() if answer_match else ""
        
        # 提取所有步骤
        # Pattern: Thought N: ... Action N: ... [Action Input N: ...] [Observation N: ...]
        thought_pattern = re.compile(r"Thought\s+(\d+):\s*(.+?)(?=\nAction|\Z)", re.DOTALL)
        action_pattern = re.compile(r"Action\s+(\d+):\s*(.+?)(?=\n|$)")
        action_input_pattern = re.compile(r"Action Input\s+(\d+):\s*(.+?)(?=\nObservation|\nThought|\Z)", re.DOTALL)
        observation_pattern = re.compile(r"Observation\s+(\d+):\s*(.+?)(?=\nThought|\nFinal|\Z)", re.DOTALL)
        
        thoughts = {int(m.group(1)): m.group(2).strip() for m in thought_pattern.finditer(text)}
        actions = {int(m.group(1)): m.group(2).strip() for m in action_pattern.finditer(text)}
        action_inputs = {int(m.group(1)): m.group(2).strip() for m in action_input_pattern.finditer(text)}
        observations = {int(m.group(1)): m.group(2).strip() for m in observation_pattern.finditer(text)}
        
        # 组装步骤
        step_ids = sorted(set(thoughts.keys()) | set(actions.keys()))
        steps = []
        for step_id in step_ids:
            step = TrajectoryStep(
                step_id=step_id,
                thought=thoughts.get(step_id, ""),
                action=actions.get(step_id, ""),
                action_input=action_inputs.get(step_id, ""),
                observation=observations.get(step_id, ""),
            )
            steps.append(step)
        
        return cls(
            trajectory_id=trajectory_id,
            category=category,
            task=task,
            steps=steps,
            final_answer=final_answer,
        )
    
    def estimate_tokens(self) -> int:
        """粗略估算 Token 数量
        
        使用简单的 word count * 1.3 作为估算。
        生产环境应使用 tiktoken 等精确计算。
        """
        text = self.to_text()
        word_count = len(text.split())
        return int(word_count * 1.3)


class TinyTrajectoryStore:
    """轨迹存储 - TTS 的主控制器
    
    核心功能:
        1. 加载: 从文件/目录批量加载轨迹
        2. 存储: 按类别管理轨迹
        3. 检索: 支持精确匹配和语义检索
        4. 导出: 支持格式化输出用于 Prompt 注入
        
    使用示例:
        ```python
        # 初始化
        store = TinyTrajectoryStore()
        
        # 加载轨迹
        store.load_from_directory("./examples/fewshots/")
        
        # 或手动添加
        store.add(Trajectory(...))
        
        # 检索
        examples = store.retrieve(
            query="How to train a model?",
            category=TrajectoryCategory.DATA_SCIENCE,
            k=2
        )
        
        # 格式化输出
        prompt_text = store.format_for_prompt(examples, max_tokens=1000)
        ```
    """
    
    def __init__(self, embedding_provider: Optional[Any] = None):
        """
        Args:
            embedding_provider: 可选的嵌入提供者，用于语义检索。
                               应实现 `embed(text: str) -> np.ndarray` 方法。
                               如果不提供，检索将使用简单的关键词匹配。
        """
        # 按类别存储轨迹
        self._trajectories: Dict[TrajectoryCategory, List[Trajectory]] = {
            cat: [] for cat in TrajectoryCategory
        }
        # ID 索引，用于快速查找
        self._id_index: Dict[str, Trajectory] = {}
        # 嵌入提供者
        self._embedder = embedding_provider
    
    def add(self, trajectory: Trajectory) -> None:
        """添加单个轨迹
        
        Args:
            trajectory: 要添加的轨迹对象
            
        Raises:
            ValueError: 如果 trajectory_id 重复
        """
        if trajectory.trajectory_id in self._id_index:
            raise ValueError(f"Trajectory ID '{trajectory.trajectory_id}' already exists")
        
        # 如果有嵌入提供者，计算嵌入
        if self._embedder is not None and trajectory._embedding is None:
            trajectory._embedding = self._embedder.embed(trajectory.task)
        
        self._trajectories[trajectory.category].append(trajectory)
        if trajectory.trajectory_id:
            self._id_index[trajectory.trajectory_id] = trajectory
    
    def get_by_id(self, trajectory_id: str) -> Optional[Trajectory]:
        """根据 ID 获取轨迹"""
        return self._id_index.get(trajectory_id)
    
    def get_by_category(self, category: TrajectoryCategory) -> List[Trajectory]:
        """获取指定类别的所有轨迹"""
        return list(self._trajectories[category])
    
    def retrieve(
        self,
        query: Optional[str] = None,
        category: Optional[TrajectoryCategory] = None,
        k: int = 3,
        min_score: float = 0.0
    ) -> List[Trajectory]:
        """检索轨迹
        
        支持两种检索模式:
        1. 精确匹配: 如果没有 embedding_provider，使用简单的关键词匹配
        2. 语义检索: 如果有 embedding_provider，使用余弦相似度
        
        Args:
            query: 查询文本，用于语义匹配。如果为 None，则返回类别下的全部轨迹。
            category: 限定的轨迹类别。如果为 None，则搜索所有类别。
            k: 返回的最大轨迹数量
            min_score: 最低相似度阈值 (仅语义检索时有效)
            
        Returns:
            匹配的轨迹列表，按相关性排序
        """
        # 确定候选池
        if category is not None:
            candidates = list(self._trajectories[category])
        else:
            candidates = []
            for cat_trajectories in self._trajectories.values():
                candidates.extend(cat_trajectories)
        
        if not candidates:
            return []
        
        # 如果没有查询，直接返回前 k 个
        if query is None:
            return candidates[:k]
        
        # 语义检索
        if self._embedder is not None:
            return self._semantic_retrieve(query, candidates, k, min_score)
        
        # 降级到关键词匹配
        return self._keyword_retrieve(query, candidates, k)
    
    def _semantic_retrieve(
        self, 
        query: str, 
        candidates: List[Trajectory], 
        k: int,
        min_score: float
    ) -> List[Trajectory]:
        """基于 Embedding 的语义检索"""
        import numpy as np
        
        query_embedding = self._embedder.embed(query)
        
        scored = []
        for traj in candidates:
            if traj._embedding is None:
                traj._embedding = self._embedder.embed(traj.task)
            
            # 计算余弦相似度
            sim = np.dot(query_embedding, traj._embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(traj._embedding) + 1e-8
            )
            if sim >= min_score:
                scored.append((sim, traj))
        
        # 按相似度降序排列
        scored.sort(key=lambda x: x[0], reverse=True)
        return [traj for _, traj in scored[:k]]
    
    def _keyword_retrieve(
        self, 
        query: str, 
        candidates: List[Trajectory], 
        k: int
    ) -> List[Trajectory]:
        """基于关键词的简单检索 (降级方案)"""
        query_words = set(query.lower().split())
        
        scored = []
        for traj in candidates:
            task_words = set(traj.task.lower().split())
            # Jaccard 相似度
            intersection = len(query_words & task_words)
            union = len(query_words | task_words)
            score = intersection / union if union > 0 else 0
            scored.append((score, traj))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        return [traj for _, traj in scored[:k]]
    
    def format_for_prompt(
        self,
        trajectories: List[Trajectory],
        max_tokens: Optional[int] = None,
        header: str = "## Examples\n\nLearn from these high-quality examples:\n\n"
    ) -> str:
        """格式化轨迹用于 Prompt 注入
        
        Args:
            trajectories: 要格式化的轨迹列表
            max_tokens: Token 预算限制。如果超出，会截断后面的轨迹。
            header: 输出的头部文本
            
        Returns:
            格式化后的 Prompt 文本
        """
        if not trajectories:
            return ""
        
        parts = [header]
        current_tokens = len(header.split())
        
        for i, traj in enumerate(trajectories):
            traj_text = f"### Example {i + 1}\n\n{traj.to_text()}\n\n"
            traj_tokens = traj.estimate_tokens()
            
            if max_tokens is not None and current_tokens + traj_tokens > max_tokens:
                # 如果第一个都放不下，至少放一个
                if i == 0:
                    parts.append(traj_text)
                break
            
            parts.append(traj_text)
            current_tokens += traj_tokens
        
        return "".join(parts)
    
    def load_from_json(self, filepath: Union[str, Path]) -> int:
        """从 JSON 文件加载轨迹
        
        JSON 格式:
            [
                {"trajectory_id": "...", "category": "...", "task": "...", ...},
                ...
            ]
            
        Args:
            filepath: JSON 文件路径
            
        Returns:
            加载的轨迹数量
        """
        filepath = Path(filepath)
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if isinstance(data, dict):
            data = [data]
        
        count = 0
        for item in data:
            try:
                trajectory = Trajectory.from_dict(item)
                if not trajectory.trajectory_id:
                    trajectory.trajectory_id = f"{filepath.stem}_{count}"
                self.add(trajectory)
                count += 1
            except Exception as e:
                print(f"Warning: Failed to load trajectory: {e}")
        
        return count
    
    def load_from_text(
        self, 
        filepath: Union[str, Path],
        category: TrajectoryCategory = TrajectoryCategory.GENERAL
    ) -> int:
        """从文本文件加载轨迹
        
        文本格式: 标准 ReAct 格式 (Question/Thought/Action/Observation)
        支持多个轨迹，用 '---' 分隔
        
        Args:
            filepath: 文本文件路径
            category: 轨迹类别
            
        Returns:
            加载的轨迹数量
        """
        filepath = Path(filepath)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        
        # 按分隔符切分多个轨迹
        segments = re.split(r"\n---+\n", content)
        
        count = 0
        for i, segment in enumerate(segments):
            segment = segment.strip()
            if not segment:
                continue
            
            try:
                trajectory = Trajectory.from_text(
                    text=segment,
                    trajectory_id=f"{filepath.stem}_{i}",
                    category=category
                )
                self.add(trajectory)
                count += 1
            except Exception as e:
                print(f"Warning: Failed to parse trajectory segment: {e}")
        
        return count
    
    def load_from_directory(
        self, 
        dirpath: Union[str, Path],
        recursive: bool = True
    ) -> int:
        """从目录批量加载轨迹
        
        自动识别文件类型:
        - .json: JSON 格式
        - .txt/.md: 文本格式
        
        目录名可用于推断类别:
        - data_science/ -> TrajectoryCategory.DATA_SCIENCE
        - reasoning/ -> TrajectoryCategory.REASONING
        
        Args:
            dirpath: 目录路径
            recursive: 是否递归搜索子目录
            
        Returns:
            加载的轨迹总数
        """
        dirpath = Path(dirpath)
        if not dirpath.exists():
            raise FileNotFoundError(f"Directory not found: {dirpath}")
        
        total = 0
        pattern = "**/*" if recursive else "*"
        
        for filepath in dirpath.glob(pattern):
            if not filepath.is_file():
                continue
            
            # 推断类别
            category = self._infer_category_from_path(filepath)
            
            if filepath.suffix == ".json":
                total += self.load_from_json(filepath)
            elif filepath.suffix in (".txt", ".md"):
                total += self.load_from_text(filepath, category)
        
        return total
    
    def _infer_category_from_path(self, filepath: Path) -> TrajectoryCategory:
        """从文件路径推断类别"""
        path_str = str(filepath).lower()
        
        category_keywords = {
            TrajectoryCategory.DATA_SCIENCE: ["data_science", "ml", "machine_learning", "ds"],
            TrajectoryCategory.CODE_GENERATION: ["code", "code_gen", "programming"],
            TrajectoryCategory.REASONING: ["reasoning", "logic", "math"],
            TrajectoryCategory.QA: ["qa", "question_answer", "hotpot"],
            TrajectoryCategory.TOOL_USE: ["tool", "api", "function_call"],
            TrajectoryCategory.REFLECTION: ["reflection", "reflexion", "self_correct"],
        }
        
        for category, keywords in category_keywords.items():
            if any(kw in path_str for kw in keywords):
                return category
        
        return TrajectoryCategory.GENERAL
    
    def save_to_json(self, filepath: Union[str, Path]) -> None:
        """将所有轨迹保存为 JSON 文件"""
        all_trajectories = []
        for cat_trajectories in self._trajectories.values():
            for traj in cat_trajectories:
                all_trajectories.append(traj.to_dict())
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(all_trajectories, f, ensure_ascii=False, indent=2)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取存储统计信息"""
        stats = {
            "total_trajectories": len(self._id_index),
            "by_category": {},
            "has_embedder": self._embedder is not None,
        }
        
        for category, trajectories in self._trajectories.items():
            if trajectories:
                stats["by_category"][category.value] = len(trajectories)
        
        return stats
    
    def __len__(self) -> int:
        return len(self._id_index)
    
    def __repr__(self) -> str:
        return f"TinyTrajectoryStore(trajectories={len(self)}, categories={len(self._trajectories)})"
