"""
MCTS Path Data Structures - MCTS 路径数据结构

用于将 MCTS 树结构传输到 Neo4j 的 Payload 定义。
按照 pipeline_revised.md Section 3.3 的 Data Contract。
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid


@dataclass
class MCTSNodePayload:
    """MCTS 节点 Payload
    
    用于批量写入 Neo4j 的节点数据。
    
    Attributes:
        node_id: 节点唯一标识
        state_hash: 状态哈希 (用于去重)
        depth: 节点深度
        is_terminal: 是否为终止节点
        
        # MCTS 统计
        N: 访问次数
        W: 累计价值
        Q: 平均价值 (W/N)
        P: 先验概率
        
        # 语义内容
        summary: 节点摘要 (LLM 生成)
        reflection: 节点反思 (LLM 生成)
        
        # 元数据
        model_meta: 模型相关元数据
    """
    node_id: str
    state_hash: str
    depth: int
    is_terminal: bool = False
    
    # MCTS 统计
    N: int = 0
    W: float = 0.0
    Q: float = 0.0
    P: float = 1.0
    
    # 语义内容
    summary: str = ""
    reflection: str = ""
    
    # 元数据
    model_meta: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodeId": self.node_id,
            "stateHash": self.state_hash,
            "depth": self.depth,
            "isTerminal": self.is_terminal,
            "N": self.N,
            "W": self.W,
            "Q": self.Q,
            "P": self.P,
            "summary": self.summary,
            "reflection": self.reflection,
            "modelMeta": self.model_meta,
        }


@dataclass
class MCTSEdgePayload:
    """MCTS 边 Payload
    
    用于批量写入 Neo4j 的边数据。
    
    Attributes:
        parent_id: 父节点 ID
        child_id: 子节点 ID
        action_id: 动作唯一标识
        action_text: 动作文本描述
        prior: 先验概率
        visits_on_edge: 通过此边的访问次数
    """
    parent_id: str
    child_id: str
    action_id: str
    action_text: str
    prior: float = 1.0
    visits_on_edge: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "parentId": self.parent_id,
            "childId": self.child_id,
            "actionId": self.action_id,
            "actionText": self.action_text,
            "prior": self.prior,
            "visitsOnEdge": self.visits_on_edge,
        }


@dataclass
class MCTSTreePayload:
    """MCTS 树 Payload
    
    完整的 MCTS 树写入载荷。
    
    Attributes:
        tree_id: 树唯一标识
        task_id: 关联的任务 ID
        run_id: 运行批次 ID
        strategy: 搜索策略名称
        nodes: 节点列表
        edges: 边列表
        created_at: 创建时间
        metadata: 额外元数据 (rollouts, time, etc.)
    """
    tree_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_id: Optional[str] = None
    run_id: Optional[str] = None
    strategy: str = "mcts"
    nodes: List[MCTSNodePayload] = field(default_factory=list)
    edges: List[MCTSEdgePayload] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_node(self, node: MCTSNodePayload) -> None:
        """添加节点"""
        self.nodes.append(node)
    
    def add_edge(self, edge: MCTSEdgePayload) -> None:
        """添加边"""
        self.edges.append(edge)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "treeId": self.tree_id,
            "taskId": self.task_id,
            "runId": self.run_id,
            "strategy": self.strategy,
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "createdAt": self.created_at.isoformat(),
            "metadata": self.metadata,
        }
    
    @property
    def node_count(self) -> int:
        return len(self.nodes)
    
    @property
    def edge_count(self) -> int:
        return len(self.edges)
