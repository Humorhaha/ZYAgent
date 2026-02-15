"""
Neo4j Memory Backend - 图存储实验组

使用 Neo4j 图数据库存储 L1 Trajectories 和 L2 Wisdom。
验证图结构对 trajectory 去重和 wisdom 溯源的效果。

需求文档约束：
- 两个后端在相同 config + seed 下必须输出等价 wisdom
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
import json

try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    GraphDatabase = None

from core.trajectory import Trajectory, Step
from core.wisdom import Wisdom
from core.mcts_path import MCTSTreePayload, MCTSNodePayload, MCTSEdgePayload
from .base import MemoryBackend


class Neo4jBackend(MemoryBackend):
    """Neo4j 图存储后端
    
    图模型设计:
    - (:Trajectory {id, task, result, created_at, ...})
    - (:Step {thought, action, observation})
    - (:Wisdom {id, text, created_at})
    
    关系:
    - (Trajectory)-[:HAS_STEP {order}]->(Step)
    - (Wisdom)-[:DERIVED_FROM]->(Trajectory)
    - (Trajectory)-[:SIMILAR_TO]->(Trajectory)  # 可用于去重
    
    Attributes:
        uri: Neo4j 连接地址
        user: 用户名
        password: 密码
    """
    
    def __init__(
        self, 
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "password",
    ):
        if not NEO4J_AVAILABLE:
            raise ImportError(
                "neo4j package not installed. "
                "Install with: pip install neo4j"
            )
        
        self.uri = uri
        self.user = user
        self.password = password
        
        # 连接 Neo4j
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        
        # 初始化 Schema
        self._init_schema()
        
        # 内存缓存（与 JsonlBackend 保持一致的行为）
        self._trajectory_cache: Dict[str, Trajectory] = {}
        self._wisdom_cache: Dict[str, Wisdom] = {}
        
        # 加载现有数据到缓存
        self.load()
    
    def _init_schema(self) -> None:
        """初始化图数据库 Schema 和索引"""
        with self.driver.session() as session:
            # Trajectory 和 Wisdom 约束
            session.run("""
                CREATE CONSTRAINT trajectory_id IF NOT EXISTS
                FOR (t:Trajectory) REQUIRE t.id IS UNIQUE
            """)
            session.run("""
                CREATE CONSTRAINT wisdom_id IF NOT EXISTS
                FOR (w:Wisdom) REQUIRE w.id IS UNIQUE
            """)
            session.run("""
                CREATE INDEX trajectory_task IF NOT EXISTS
                FOR (t:Trajectory) ON (t.task)
            """)
            
            # MCTS Tree 约束和索引 (按 pipeline_revised.md Section 4.3)
            session.run("""
                CREATE CONSTRAINT mcts_node_unique IF NOT EXISTS
                FOR (n:MCTSNode)
                REQUIRE (n.treeId, n.nodeId) IS UNIQUE
            """)
            session.run("""
                CREATE INDEX mcts_statehash IF NOT EXISTS
                FOR (n:MCTSNode) ON (n.treeId, n.stateHash)
            """)
            # 全文索引用于 summary/reflection 语义搜索
            try:
                session.run("""
                    CREATE FULLTEXT INDEX mcts_text IF NOT EXISTS
                    FOR (n:MCTSNode) ON EACH [n.summary, n.reflection]
                """)
            except Exception:
                pass  # 忽略已存在的索引错误
    
    # =========================================================================
    # 核心接口实现
    # =========================================================================
    
    def put_samples(self, trajectories: List[Trajectory]) -> None:
        """存储 trajectories 到图数据库"""
        with self.driver.session() as session:
            for traj in trajectories:
                if traj.id in self._trajectory_cache:
                    continue  # 已存在则跳过
                
                # 创建 Trajectory 节点
                session.run("""
                    CREATE (t:Trajectory {
                        id: $id,
                        task: $task,
                        result: $result,
                        created_at: $created_at,
                        used_wisdom: $used_wisdom,
                        triggered_system2: $triggered_system2,
                        metadata: $metadata
                    })
                """, 
                    id=traj.id,
                    task=traj.task,
                    result=traj.result,
                    created_at=traj.created_at.isoformat(),
                    used_wisdom=traj.used_wisdom,
                    triggered_system2=traj.triggered_system2,
                    metadata=json.dumps(traj.metadata),
                )
                
                # 创建 Step 节点并建立关系
                for idx, step in enumerate(traj.steps):
                    session.run("""
                        MATCH (t:Trajectory {id: $traj_id})
                        CREATE (s:Step {
                            thought: $thought,
                            action: $action,
                            observation: $observation,
                            metadata: $metadata
                        })
                        CREATE (t)-[:HAS_STEP {order: $order}]->(s)
                    """,
                        traj_id=traj.id,
                        order=idx,
                        thought=step.thought,
                        action=step.action,
                        observation=step.observation,
                        metadata=json.dumps(step.metadata),
                    )
                
                # 更新缓存
                self._trajectory_cache[traj.id] = traj
    
    def get_samples(self, limit: int = 100) -> List[Trajectory]:
        """获取 L1 中的 samples"""
        # 从缓存返回（保持与 JsonlBackend 行为一致）
        all_trajs = list(self._trajectory_cache.values())
        return all_trajs[-limit:]
    
    def promote(self, min_samples: int = 5) -> List[Wisdom]:
        """从 L1 提升到 L2
        
        使用 Cypher 按 task 分组并检查阈值。
        """
        new_wisdoms = []
        
        with self.driver.session() as session:
            # 查找满足条件的任务组
            result = session.run("""
                MATCH (t:Trajectory)
                WHERE NOT EXISTS {
                    MATCH (w:Wisdom)-[:DERIVED_FROM]->(t)
                }
                WITH t.task AS task, COLLECT(t) AS trajs
                WHERE SIZE(trajs) >= $min_samples
                RETURN task, trajs
            """, min_samples=min_samples)
            
            for record in result:
                task = record["task"]
                trajs = record["trajs"]
                
                # 生成 Wisdom
                success_count = sum(1 for t in trajs if t["result"] == "success")
                failure_count = len(trajs) - success_count
                
                wisdom_text = (
                    f"[Wisdom for task: {task[:50]}...]\n"
                    f"Based on {len(trajs)} trajectories "
                    f"({success_count} success, {failure_count} failure).\n"
                    f"Key insight: Review observations carefully before acting."
                )
                
                source_ids = [t["id"] for t in trajs]
                wisdom = Wisdom(
                    text=wisdom_text,
                    source_trajectory_ids=source_ids,
                    metadata={
                        "promoted_from_task": task,
                        "sample_count": len(trajs),
                        "success_rate": success_count / len(trajs),
                    }
                )
                
                # 存储 Wisdom 节点和关系
                session.run("""
                    CREATE (w:Wisdom {
                        id: $id,
                        text: $text,
                        created_at: $created_at,
                        metadata: $metadata
                    })
                """,
                    id=wisdom.id,
                    text=wisdom.text,
                    created_at=wisdom.created_at.isoformat(),
                    metadata=json.dumps(wisdom.metadata),
                )
                
                # 建立 DERIVED_FROM 关系
                for traj_id in source_ids:
                    session.run("""
                        MATCH (w:Wisdom {id: $wisdom_id})
                        MATCH (t:Trajectory {id: $traj_id})
                        CREATE (w)-[:DERIVED_FROM]->(t)
                    """, wisdom_id=wisdom.id, traj_id=traj_id)
                
                self._wisdom_cache[wisdom.id] = wisdom
                new_wisdoms.append(wisdom)
        
        return new_wisdoms
    
    def retrieve(self, task: str, k: int = 3) -> List[Wisdom]:
        """从 L2 检索 Wisdom
        
        使用图查询进行相似度匹配。
        """
        candidates = []
        
        with self.driver.session() as session:
            # 优先精确匹配
            result = session.run("""
                MATCH (w:Wisdom)
                WHERE w.metadata CONTAINS $task
                RETURN w.id AS id, w.text AS text, 
                       w.created_at AS created_at, w.metadata AS metadata,
                       100 AS score
                ORDER BY score DESC
                LIMIT $k
            """, task=task, k=k)
            
            for record in result:
                wisdom = self._wisdom_cache.get(record["id"])
                if wisdom:
                    candidates.append((record["score"], wisdom))
            
            # 如果不够，用模糊匹配补充
            if len(candidates) < k:
                result = session.run("""
                    MATCH (w:Wisdom)
                    RETURN w.id AS id, w.text AS text,
                           w.created_at AS created_at, w.metadata AS metadata
                    LIMIT $limit
                """, limit=k * 2)
                
                task_words = set(task.lower().split())
                for record in result:
                    wisdom = self._wisdom_cache.get(record["id"])
                    if wisdom and not any(c[1].id == wisdom.id for c in candidates):
                        wisdom_words = set(wisdom.text.lower().split())
                        overlap = len(task_words & wisdom_words)
                        if overlap > 0:
                            candidates.append((overlap, wisdom))
        
        candidates.sort(key=lambda x: x[0], reverse=True)
        return [c[1] for c in candidates[:k]]
    
    # =========================================================================
    # ID 查询（溯源支持 - 图数据库优势）
    # =========================================================================
    
    def get_wisdom_by_id(self, wisdom_id: str) -> Optional[Wisdom]:
        return self._wisdom_cache.get(wisdom_id)
    
    def get_trajectory_by_id(self, trajectory_id: str) -> Optional[Trajectory]:
        return self._trajectory_cache.get(trajectory_id)
    
    def get_all_wisdom(self) -> List[Wisdom]:
        return list(self._wisdom_cache.values())
    
    def trace_wisdom_sources(self, wisdom_id: str) -> List[Trajectory]:
        """追溯 Wisdom 的所有来源 Trajectory（图数据库优势查询）"""
        wisdom = self._wisdom_cache.get(wisdom_id)
        if not wisdom:
            return []
        
        return [
            self._trajectory_cache[tid] 
            for tid in wisdom.source_trajectory_ids 
            if tid in self._trajectory_cache
        ]
    
    def find_similar_trajectories(self, trajectory_id: str, limit: int = 5) -> List[Trajectory]:
        """查找相似的 Trajectory（图去重功能）"""
        traj = self._trajectory_cache.get(trajectory_id)
        if not traj:
            return []
        
        # 简单实现：按 task 匹配
        similar = [
            t for t in self._trajectory_cache.values()
            if t.id != trajectory_id and t.task == traj.task
        ]
        return similar[:limit]
    
    # =========================================================================
    # 持久化
    # =========================================================================
    
    def clear(self) -> None:
        """清空所有数据"""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        
        self._trajectory_cache = {}
        self._wisdom_cache = {}
    
    def save(self) -> None:
        """图数据库自动持久化，此方法为空操作"""
        pass
    
    def load(self) -> None:
        """从图数据库加载数据到缓存"""
        with self.driver.session() as session:
            # 加载 Trajectories
            result = session.run("""
                MATCH (t:Trajectory)
                OPTIONAL MATCH (t)-[r:HAS_STEP]->(s:Step)
                RETURN t, COLLECT({step: s, order: r.order}) AS steps
                ORDER BY t.created_at
            """)
            
            for record in result:
                t_node = record["t"]
                steps_data = sorted(record["steps"], key=lambda x: x["order"] or 0)
                
                steps = []
                for sd in steps_data:
                    if sd["step"]:
                        s = sd["step"]
                        steps.append(Step(
                            thought=s["thought"],
                            action=s["action"],
                            observation=s["observation"],
                            metadata=json.loads(s.get("metadata", "{}"))
                        ))
                
                traj = Trajectory(
                    id=t_node["id"],
                    task=t_node["task"],
                    result=t_node["result"],
                    created_at=datetime.fromisoformat(t_node["created_at"]),
                    used_wisdom=t_node.get("used_wisdom", False),
                    triggered_system2=t_node.get("triggered_system2", False),
                    metadata=json.loads(t_node.get("metadata", "{}")),
                )
                traj.steps = steps
                self._trajectory_cache[traj.id] = traj
            
            # 加载 Wisdom
            result = session.run("""
                MATCH (w:Wisdom)
                OPTIONAL MATCH (w)-[:DERIVED_FROM]->(t:Trajectory)
                RETURN w, COLLECT(t.id) AS source_ids
            """)
            
            for record in result:
                w_node = record["w"]
                source_ids = [sid for sid in record["source_ids"] if sid]
                
                wisdom = Wisdom(
                    id=w_node["id"],
                    text=w_node["text"],
                    source_trajectory_ids=source_ids,
                    created_at=datetime.fromisoformat(w_node["created_at"]),
                    metadata=json.loads(w_node.get("metadata", "{}")),
                )
                self._wisdom_cache[wisdom.id] = wisdom
    
    # =========================================================================
    # MCTS Tree 存储 (按 pipeline_revised.md Section 4)
    # =========================================================================
    
    def write_mcts_tree(self, payload: MCTSTreePayload) -> None:
        """批量写入 MCTS 树结构到 Neo4j
        
        按照 pipeline_revised.md Appendix A 的 Cypher 模式。
        使用 UNWIND 进行批量操作以提高性能。
        
        Args:
            payload: MCTSTreePayload 包含 tree 元数据、节点列表和边列表
        """
        with self.driver.session() as session:
            # 1. 创建或更新 SearchTree 节点
            session.run("""
                MERGE (t:SearchTree {treeId: $treeId})
                ON CREATE SET 
                    t.taskId = $taskId,
                    t.runId = $runId,
                    t.createdAt = timestamp(),
                    t.strategy = $strategy
                ON MATCH SET
                    t.updatedAt = timestamp()
            """,
                treeId=payload.tree_id,
                taskId=payload.task_id,
                runId=payload.run_id,
                strategy=payload.strategy,
            )
            
            # 2. 批量创建/更新 MCTSNode 节点
            if payload.nodes:
                nodes_data = [n.to_dict() for n in payload.nodes]
                session.run("""
                    UNWIND $nodes AS nd
                    MERGE (n:MCTSNode {treeId: $treeId, nodeId: nd.nodeId})
                    SET n.stateHash = nd.stateHash,
                        n.depth = nd.depth,
                        n.isTerminal = nd.isTerminal,
                        n.N = nd.N,
                        n.W = nd.W,
                        n.Q = nd.Q,
                        n.P = nd.P,
                        n.summary = nd.summary,
                        n.reflection = nd.reflection,
                        n.updatedAt = timestamp()
                """,
                    treeId=payload.tree_id,
                    nodes=nodes_data,
                )
            
            # 3. 创建 HAS_ROOT 关系（假设 depth=0 是 root）
            session.run("""
                MATCH (t:SearchTree {treeId: $treeId})
                MATCH (root:MCTSNode {treeId: $treeId, depth: 0})
                MERGE (t)-[:HAS_ROOT]->(root)
            """, treeId=payload.tree_id)
            
            # 4. 批量创建 CHILD 边
            if payload.edges:
                edges_data = [e.to_dict() for e in payload.edges]
                session.run("""
                    UNWIND $edges AS e
                    MATCH (p:MCTSNode {treeId: $treeId, nodeId: e.parentId})
                    MATCH (c:MCTSNode {treeId: $treeId, nodeId: e.childId})
                    MERGE (p)-[r:CHILD {treeId: $treeId, actionId: e.actionId, childId: e.childId}]->(c)
                    SET r.actionText = e.actionText,
                        r.prior = e.prior,
                        r.visitsOnEdge = e.visitsOnEdge,
                        r.updatedAt = timestamp()
                """,
                    treeId=payload.tree_id,
                    edges=edges_data,
                )
    
    def query_best_path(self, tree_id: str) -> Optional[List[Dict[str, Any]]]:
        """查询 MCTS 树中的最优终端路径
        
        按照 pipeline_revised.md Appendix B 的 Cypher 模式。
        
        Args:
            tree_id: 树的唯一标识
            
        Returns:
            路径中的节点列表（从 root 到 terminal），如果不存在则返回 None
        """
        with self.driver.session() as session:
            result = session.run("""
                MATCH (t:SearchTree {treeId: $treeId})-[:HAS_ROOT]->(root)
                MATCH p = (root)-[:CHILD*0..50]->(leaf:MCTSNode)
                WHERE leaf.isTerminal = true
                WITH p, leaf
                ORDER BY leaf.Q DESC
                LIMIT 1
                UNWIND nodes(p) AS node
                RETURN node.nodeId AS nodeId,
                       node.depth AS depth,
                       node.summary AS summary,
                       node.reflection AS reflection,
                       node.Q AS Q,
                       node.N AS N
            """, treeId=tree_id)
            
            path_nodes = []
            for record in result:
                path_nodes.append({
                    "nodeId": record["nodeId"],
                    "depth": record["depth"],
                    "summary": record["summary"],
                    "reflection": record["reflection"],
                    "Q": record["Q"],
                    "N": record["N"],
                })
            
            return path_nodes if path_nodes else None
    
    def get_mcts_tree_stats(self, tree_id: str) -> Optional[Dict[str, Any]]:
        """获取 MCTS 树的统计信息
        
        Args:
            tree_id: 树的唯一标识
            
        Returns:
            包含节点数、边数、最大深度等统计信息的字典
        """
        with self.driver.session() as session:
            result = session.run("""
                MATCH (t:SearchTree {treeId: $treeId})
                OPTIONAL MATCH (n:MCTSNode {treeId: $treeId})
                WITH t, COUNT(n) AS nodeCount, MAX(n.depth) AS maxDepth
                OPTIONAL MATCH (:MCTSNode {treeId: $treeId})-[r:CHILD]->(:MCTSNode {treeId: $treeId})
                WITH t, nodeCount, maxDepth, COUNT(r) AS edgeCount
                OPTIONAL MATCH (term:MCTSNode {treeId: $treeId})
                WHERE term.isTerminal = true
                RETURN t.strategy AS strategy,
                       t.taskId AS taskId,
                       nodeCount,
                       edgeCount,
                       maxDepth,
                       COUNT(term) AS terminalCount
            """, treeId=tree_id)
            
            record = result.single()
            if record:
                return {
                    "tree_id": tree_id,
                    "strategy": record["strategy"],
                    "task_id": record["taskId"],
                    "node_count": record["nodeCount"],
                    "edge_count": record["edgeCount"],
                    "max_depth": record["maxDepth"],
                    "terminal_count": record["terminalCount"],
                }
            return None
    
    def close(self) -> None:
        """关闭数据库连接"""
        if hasattr(self, 'driver') and self.driver:
            self.driver.close()
    
    def __del__(self):
        self.close()
