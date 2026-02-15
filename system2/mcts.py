"""
MCTS Strategy - MCTS 搜索策略

适配现有 MCTS/base.py 到 System 2 接口。
"""

from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass, field
import time
import hashlib
import json
import re

from config.schema import System2Config
from core.failure import FailureCase
from core.trajectory import Trajectory, Step
from core.mcts_path import MCTSTreePayload, MCTSNodePayload, MCTSEdgePayload
from MCTS.base import MCTSTask, MCTS, Node
from LLM import prompts
from LLM.llm import LLM, create_llm
from .base import SearchStrategy


class MCTSAdapter(MCTSTask):
    """MCTS 任务适配器
    
    将 System 2 的配置和 Failure Case 转换为 MCTSTask 所需的接口。
    现在已接入 LLM 进行真实推理。
    
    Attributes:
        llm: LLM 实例用于生成动作和反思
        failure_case: 失败案例
        config: System2 配置
    """
    
    def __init__(
        self, 
        failure_case: FailureCase, 
        config: System2Config,
        llm: Optional[LLM] = None,
    ):
        # 转换配置参数 (注意: MCTSTask 不接受 max_depth)
        super().__init__(
            limit_type='iteration',
            iteration_limit=config.iteration_limit,
            branch=config.branch,
            roll_branch=config.roll_branch,
            roll_forward_steps=config.roll_forward_steps,
            use_reflection=config.use_reflection,
            roll_policy=config.roll_policy,
            alpha=config.alpha,
            exploration_constant=config.exploration_constant,
        )
        self.failure_case = failure_case
        self.config = config
        self.max_depth = config.max_depth
        
        # 注入 LLM 实例 (如果未提供则创建)
        self.llm = llm or create_llm()
    
    # -------------------------------------------------------------------------
    # 抽象方法实现 (LLM 接入)
    # -------------------------------------------------------------------------
    
    def get_next_step(self, history: str, step_n: int) -> str:
        """生成下一步动作 (使用 LLM)"""
        prompt = prompts.MCTS_NEXT_STEP.format(
            task=self.failure_case.task,
            history=history or "(空)",
            step_n=step_n,
        )
        return self.llm.generate(prompt).strip()
    
    def get_next_step_use_reflection(self, history: str, step_n: int, reflection: str) -> str:
        """基于反思生成改进后的动作 (使用 LLM)"""
        prompt = prompts.MCTS_NEXT_STEP_WITH_REFLECTION.format(
            task=self.failure_case.task,
            history=history or "(空)",
            reflection=reflection,
            step_n=step_n,
        )
        return self.llm.generate(prompt).strip()
    
    def get_reflection(self, history: str, step_n: int) -> str:
        """生成反思 (使用 LLM)"""
        # 深度过大时主动终止
        if step_n > self.max_depth:
            return "<end>"
        
        prompt = prompts.MCTS_REFLECTION.format(
            task=self.failure_case.task,
            history=history or "(空)",
            step_n=step_n,
        )
        return self.llm.generate(prompt).strip()
    
    def get_simple_reflection(self, history: str, step_n: int) -> str:
        """生成简单反思 (使用 LLM)"""
        if step_n > self.max_depth:
            return "<end>"
        
        prompt = prompts.MCTS_SIMPLE_REFLECTION.format(
            history=history or "(空)",
            step_n=step_n,
        )
        return self.llm.generate(prompt).strip()
    
    def get_step_value(self, history: str) -> float:
        """评估当前轨迹价值 (使用 LLM)"""
        prompt = prompts.MCTS_VALUE_EVALUATION.format(
            task=self.failure_case.task,
            history=history or "(空)",
        )
        
        response = self.llm.generate(prompt).strip()
        
        # 解析 JSON 响应
        try:
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                value = float(data.get("value", 0.5))
                return max(0.0, min(1.0, value))
        except (json.JSONDecodeError, ValueError, KeyError):
            pass
        
        # 解析失败时返回中等值
        return 0.5


@dataclass
class MCTSSearchResult:
    """MCTS 搜索结果
    
    包含候选轨迹和用于 Neo4j 批量写入的树结构。
    """
    trajectories: List[Trajectory]
    tree_payload: MCTSTreePayload
    metadata: Dict[str, Any] = field(default_factory=dict)


class MCTSStrategy(SearchStrategy):
    """MCTS 搜索策略实现
    
    Attributes:
        llm: 可选的 LLM 实例，如果未提供则创建新实例
    """
    
    def __init__(self, llm: Optional[LLM] = None):
        """
        Args:
            llm: LLM 实例，用于搜索过程中的推理。如果未提供，
                 MCTSAdapter 会在初始化时创建自己的 LLM 实例。
        """
        self.llm = llm
    
    def search(
        self, 
        failure_case: FailureCase, 
        config: System2Config,
    ) -> List[Trajectory]:
        """执行 MCTS 搜索（保持向后兼容）"""
        result = self.search_with_tree(failure_case, config)
        return result.trajectories
    
    def search_with_tree(
        self, 
        failure_case: FailureCase, 
        config: System2Config,
    ) -> MCTSSearchResult:
        """执行 MCTS 搜索并返回完整结果（包含树结构）
        
        按照 pipeline_revised.md Section 5 的要求：
        - 返回候选轨迹
        - 返回 MCTS 路径（用于批量写入 Neo4j）
        - 返回元数据（rollouts, time, etc.）
        """
        start_time = time.time()
        
        # 1. 创建任务适配器 (传入 LLM 实例)
        task_adapter = MCTSAdapter(failure_case, config, llm=self.llm)
        
        # 2. 初始化核心 MCTS
        mcts = MCTS(task_adapter)
        
        # 3. 运行搜索
        root, best_node, finish_info = mcts.search()
        
        # 4. 收集 Top-K 轨迹
        trajectories = self._collect_top_k(root, config.top_k, failure_case.task)
        
        # 5. 构建 MCTSTreePayload
        tree_payload = self._build_tree_payload(
            root=root,
            task_id=failure_case.task[:50],  # 使用任务前50字符作为 ID
            run_id=failure_case.trajectory_id,
            strategy="mcts",
        )
        
        # 标记来源
        for traj in trajectories:
            traj.triggered_system2 = True
            traj.metadata["search_method"] = "mcts"
            traj.metadata["tree_id"] = tree_payload.tree_id
        
        elapsed_time = time.time() - start_time
        
        return MCTSSearchResult(
            trajectories=trajectories,
            tree_payload=tree_payload,
            metadata={
                "elapsed_time_s": elapsed_time,
                "node_count": tree_payload.node_count,
                "edge_count": tree_payload.edge_count,
                "finish_info": finish_info,
            }
        )
    
    def _build_tree_payload(
        self, 
        root: Node, 
        task_id: str, 
        run_id: Optional[str],
        strategy: str,
    ) -> MCTSTreePayload:
        """从 MCTS 搜索树构建 Neo4j 写入 Payload"""
        payload = MCTSTreePayload(
            task_id=task_id,
            run_id=run_id,
            strategy=strategy,
        )
        
        # 遍历树收集所有节点和边
        node_id_map: Dict[int, str] = {}  # id(node) -> nodeId
        self._traverse_tree(root, payload, node_id_map, depth=0)
        
        return payload
    
    def _traverse_tree(
        self, 
        node: Node, 
        payload: MCTSTreePayload, 
        node_id_map: Dict[int, str],
        depth: int,
    ) -> str:
        """递归遍历树并收集节点/边"""
        # 生成节点 ID
        node_id = f"node_{len(node_id_map)}"
        node_id_map[id(node)] = node_id
        
        # 生成 state hash
        state_hash = hashlib.md5(
            (node.new_action or "root").encode()
        ).hexdigest()[:16]
        
        # 创建节点 Payload
        node_payload = MCTSNodePayload(
            node_id=node_id,
            state_hash=state_hash,
            depth=depth,
            is_terminal=node.is_leaf,
            N=node.visits,
            W=node.value * node.visits,  # 反推累计价值
            Q=node.value,
            P=1.0,  # Mock prior
            summary=node.new_action or "",
            reflection=getattr(node, 'reflection', ""),
        )
        payload.add_node(node_payload)
        
        # 递归处理子节点并创建边 (children 是 Dict[str, Node])
        for i, (action_key, child) in enumerate(node.children.items()):
            child_node_id = self._traverse_tree(child, payload, node_id_map, depth + 1)
            
            edge_payload = MCTSEdgePayload(
                parent_id=node_id,
                child_id=child_node_id,
                action_id=f"action_{node_id}_{i}",
                action_text=child.new_action or action_key,
                prior=1.0 / max(1, len(node.children)),
                visits_on_edge=child.visits,
            )
            payload.add_edge(edge_payload)
        
        return node_id

    def _collect_top_k(self, root: Node, k: int, task: str) -> List[Trajectory]:
        """从 MCTS 树收集 Top-K 轨迹"""
        # 收集所有叶子节点及其价值
        candidates = []
        self._collect_leaves(root, candidates)
        
        # 按价值排序
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # 转换为 Trajectory
        trajectories = []
        for node, value in candidates[:k]:
            traj = self._node_to_trajectory(node, task)
            traj.metadata["mcts_value"] = value
            trajectories.append(traj)
            
        return trajectories
    
    def _collect_leaves(self, node: Node, results: List):
        """递归收集叶子节点"""
        if node.is_leaf:
            results.append((node, node.value))
        else:
            # children 是 Dict[str, Node]，需要遍历 values()
            for child in node.children.values():
                self._collect_leaves(child, results)

    def _node_to_trajectory(self, node: Node, task: str) -> Trajectory:
        """将 MCTS 节点链转换为 Trajectory"""
        # 1. 回溯路径
        path = []
        curr = node
        while curr and curr.parent:
            path.append(curr)
            curr = curr.parent
        path.reverse()
        
        # 2. 构建 Trajectory
        traj = Trajectory(task=task)
        
        for n in path:
            traj.add_step(
                thought="",
                action=n.new_action,
                observation="",
                metadata={"node_value": n.value, "node_visits": n.visits}
            )
            
        # 根据 value 判断成功/失败
        if node.value > 0.8:
            traj.mark_success()
        else:
            traj.mark_failure()
                 
        return traj

