"""
Async MCTS Bridge - System 1 与 System 2 之间的桥梁

本模块实现异步 MCTS 的调度与集成逻辑。

核心组件:
    - Monitor: 监控 System 1 状态，决定何时触发System 2。
    - WisdomDistiller: 将 MCTS 搜索结果提炼为可注入的 L3 Wisdom。
    - AsyncMCTSBridge: 多线程控制器，协调 Snapshot 传输和结果回调。
"""

import threading
import queue
import time
import copy
from typing import Optional, Callable, Dict, Any, List
from dataclasses import dataclass

from Memory.memory import AgentMemory
from MCTS.mcts import MCTS, Environment, VanillaMCTS
from MCTS.config import MCTSConfig, DeepThinkerConfig


def create_deep_thinker_factory(config: MCTSConfig = None):
    """创建 DeepThinker MCTS 的工厂函数
    
    Args:
        config: MCTS 配置，默认使用 DeepThinkerConfig
        
    Returns:
        工厂函数，接受 env 并返回 MCTS 实例
    """
    if config is None:
        config = DeepThinkerConfig()
    
    def factory(env: Environment) -> MCTS:
        return VanillaMCTS(env, c_param=config.c_puct, max_depth=config.max_depth)
    
    return factory


# =============================================================================
# 1. Monitor (触发器)
# =============================================================================

@dataclass
class MonitorConfig:
    """监控配置"""
    entropy_threshold: float = 0.8       # 熵阈值 (迷茫)
    consecutive_failure_threshold: int = 3 # 连续失败次数
    critical_keywords: List[str] = None  # 危机关键词

    def __post_init__(self):
        if self.critical_keywords is None:
            self.critical_keywords = ["error", "exception", "failed", "timeout"]


class Monitor:
    """System 1 监控器
    
    职责:
        - 实时分析 Agent 的运行指标
        - 判断是否需要“呼叫支援” (Trigger System 2)
    """
    
    def __init__(self, config: MonitorConfig = None):
        self.config = config or MonitorConfig()
        self._consecutive_failures = 0
    
    def check_trigger(
        self, 
        memory_stats: Dict[str, Any], 
        recent_events: List[str] = None,
        next_token_entropy: float = 0.0
    ) -> bool:
        """检查是否应该触发 MCTS
        
        Args:
            memory_stats: Memory 统计信息
            recent_events: 最近发生的事件内容列表 (用于历史分析)
            next_token_entropy: 模型预测下一个 Token 的熵 (如果可用)
            
        Returns:
            bool: 是否触发
        """
        triggers = []
        recent_events = recent_events or []
        
        # 1. 熵检测 (Confusion)
        if next_token_entropy > self.config.entropy_threshold:
            triggers.append(f"High Entropy ({next_token_entropy:.2f})")
        
        # 2. 连续失败检测 (Frustration)
        # 倒序检查最近的事件
        current_consecutive_failures = 0
        for event_content in reversed(recent_events):
            is_failure = any(kw in event_content.lower() for kw in self.config.critical_keywords)
            if is_failure:
                current_consecutive_failures += 1
            else:
                break
        
        # 更新内部状态 (可选，但主要依赖历史检查)
        self._consecutive_failures = current_consecutive_failures
            
        if self._consecutive_failures >= self.config.consecutive_failure_threshold:
            triggers.append(f"Consecutive Failures ({self._consecutive_failures})")
        
        if triggers:
            print(f"[Monitor] System 2 Triggered by: {', '.join(triggers)}")
            return True
            
        return False


# =============================================================================
# 2. Wisdom Distiller (智慧提炼器)
# =============================================================================

class LLMProvider:
    """LLM Provider Protocol (用于类型提示)"""
    def generate(self, prompt: str) -> str:
        raise NotImplementedError


class MockLLMProvider:
    """Mock LLM Provider (用于测试)"""
    def generate(self, prompt: str) -> str:
        # 简单模拟：提取关键词并生成建议
        if "failure" in prompt.lower() or "error" in prompt.lower():
            return "Consider using error handling and retry mechanisms."
        return "Focus on the most promising paths first."


class WisdomDistiller:
    """智慧提炼器
    
    职责:
        - 分析 MCTS 搜索生成的树
        - 调用 LLM 提取关键决策逻辑
        - 将其转化为自然语言建议 (Wisdom)
    
    改进点:
        - 使用 LLM 进行语义总结，而非简单拼接动作
        - 提取失败路径的教训
        - 生成可泛化的策略建议
    """
    
    # Prompt 模板
    DISTILLATION_PROMPT = '''You are analyzing the results of a Monte Carlo Tree Search (MCTS) to extract actionable wisdom.

## Search Context
- Initial State: {initial_state}
- Best Action Found: {best_action}
- Total Simulations: {total_simulations}

## Best Path (Most Visited)
{best_path}

## Alternative Paths Explored
{alternative_paths}

## Failed Paths (Low Reward)
{failed_paths}

## Task
Based on the above search results, provide ONE concise, actionable insight that could help solve similar problems in the future. 
Focus on:
1. What strategy led to success
2. What pitfalls to avoid
3. General principles that can be reused

Output only the wisdom statement (1-2 sentences), no explanation needed.'''

    def __init__(self, llm_provider: Optional[LLMProvider] = None):
        """
        Args:
            llm_provider: LLM 提供者，用于生成语义总结
        """
        self.llm = llm_provider
    
    def distill(self, mcts_instance: MCTS, root_node, best_action) -> str:
        """从 MCTS 实例中提炼智慧
        
        Args:
            mcts_instance: MCTS 实例
            root_node: 搜索树根节点
            best_action: 最佳动作
            
        Returns:
            提炼的智慧文本
        """
        # 1. 提取搜索树信息
        search_info = self._extract_search_info(root_node)
        
        # 2. 如果没有 LLM，使用 Rule-based 回退
        if self.llm is None:
            return self._rule_based_distill(search_info, best_action)
        
        # 3. 构建 Prompt 并调用 LLM
        prompt = self.DISTILLATION_PROMPT.format(
            initial_state=search_info.get("initial_state", "Unknown"),
            best_action=best_action,
            total_simulations=search_info.get("total_visits", 0),
            best_path=search_info.get("best_path_str", "N/A"),
            alternative_paths=search_info.get("alt_paths_str", "None"),
            failed_paths=search_info.get("failed_paths_str", "None")
        )
        
        try:
            wisdom = self.llm.generate(prompt)
            return wisdom.strip()
        except Exception as e:
            print(f"[Distiller] LLM call failed: {e}, falling back to rule-based")
            return self._rule_based_distill(search_info, best_action)
    
    def _extract_search_info(self, root_node) -> Dict[str, Any]:
        """从搜索树中提取结构化信息"""
        info = {
            "initial_state": str(root_node.state) if root_node else "Unknown",
            "total_visits": root_node.visits if root_node else 0,
            "best_path": [],
            "alt_paths": [],
            "failed_paths": []
        }
        
        if not root_node or not root_node.children:
            return info
        
        # 按访问次数排序子节点
        sorted_children = sorted(root_node.children, key=lambda c: c.visits, reverse=True)
        
        # 提取最佳路径
        info["best_path"] = self._trace_path(sorted_children[0])
        
        # 提取替代路径 (第 2、3 名)
        for child in sorted_children[1:3]:
            if child.visits > 0:
                info["alt_paths"].append(self._trace_path(child))
        
        # 提取失败路径 (低 Q 值) TODO 0.3 can be adjusted
        for child in sorted_children:
            if child.visits > 0 and child.q_value < 0.3:
                info["failed_paths"].append({
                    "path": self._trace_path(child),
                    "q_value": child.q_value
                })
        
        # 格式化为字符串
        info["best_path_str"] = " -> ".join(str(a) for a in info["best_path"])
        info["alt_paths_str"] = "\n".join(
            f"- {' -> '.join(str(a) for a in p)}" for p in info["alt_paths"]
        ) or "None"
        info["failed_paths_str"] = "\n".join(
            f"- {' -> '.join(str(a) for a in fp['path'])} (Q={fp['q_value']:.2f})"
            for fp in info["failed_paths"]
        ) or "None"
        
        return info
    
    def _trace_path(self, node, max_depth: int = 5) -> List[Any]:
        """沿着最佳子节点追踪路径"""
        path = []
        current = node
        
        while current and len(path) < max_depth:
            if current.action is not None:
                path.append(current.action)
            if current.children:
                current = max(current.children, key=lambda c: c.visits)
            else:
                break
        
        return path
    
    def _rule_based_distill(self, search_info: Dict, best_action) -> str:
        """Rule-based 回退方案 (无 LLM 时使用)"""
        best_path = search_info.get("best_path", [])
        failed_paths = search_info.get("failed_paths", [])
        
        parts = []
        
        # 最佳路径建议
        if best_path:
            path_str = " -> ".join(str(a) for a in best_path[:3])
            parts.append(f"Best strategy: follow sequence [{path_str}]")
        
        # 失败教训
        if failed_paths:
            failed_actions = [fp["path"][0] for fp in failed_paths if fp["path"]]
            if failed_actions:
                avoid_str = ", ".join(str(a) for a in failed_actions[:2])
                parts.append(f"Avoid starting with: [{avoid_str}]")
        
        if not parts:
            return "Consider alternative approaches based on search results."
        
        return ". ".join(parts) + "."


# =============================================================================
# 3. Async Bridge (异步桥接器)
# =============================================================================

class AsyncMCTSBridge:
    """异步 MCTS 桥接器
    
    职责:
        - 管理后台线程
        - 从 Failure Queue 拉取失败任务
        - 运行 MCTS 进行 Off-policy 探索
        - 将结果写入 HCC 并提炼 Wisdom
    
    架构位置 (参考 pipeline.tex):
        后台进程: Failure Queue → MCTS → HCC (L1/L2) → Lazy Update
    """
    
    def __init__(
        self, 
        mcts_factory: Callable[[Environment], MCTS],
        monitor_config: MonitorConfig = None,
        llm_provider: Optional[LLMProvider] = None,
        failure_queue: Optional[Any] = None,  # FailureQueue instance
        target_memory: Optional[AgentMemory] = None,  # 主进程 Memory (用于写入 HCC)
    ):
        """
        Args:
            mcts_factory: 创建 MCTS 实例的工厂函数
            monitor_config: 监控配置
            llm_provider: LLM 提供者，用于 Wisdom 总结
            failure_queue: Failure Queue 实例 (System 2 任务来源)
            target_memory: 主进程 AgentMemory (用于 HCC 写入)
        """
        self.mcts_factory = mcts_factory
        self.monitor = Monitor(monitor_config)
        self.distiller = WisdomDistiller(llm_provider=llm_provider)
        
        # Failure Queue 整合
        self.failure_queue = failure_queue
        self.target_memory = target_memory
        
        # 任务队列
        self._queue = queue.Queue()
        self._stop_event = threading.Event()
        self._worker_thread = None
        
        # Lazy Update: pending wisdom 队列
        self._pending_wisdom: List[str] = []
        self._pending_lock = threading.Lock()
        
        # 启动后台线程
        self.start()
    
    def start(self):
        """启动后台工作线程"""
        if self._worker_thread and self._worker_thread.is_alive():
            return
        
        self._stop_event.clear()
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()
        print("[Bridge] Async MCTS service started.")
    
    def stop(self):
        """停止后台服务"""
        self._stop_event.set()
        if self._worker_thread:
            self._worker_thread.join(timeout=2.0)
            print("[Bridge] Async MCTS service stopped.")
    
    def has_pending(self) -> bool:
        """检查是否有 pending wisdom"""
        with self._pending_lock:
            return len(self._pending_wisdom) > 0
    
    def pop_wisdom(self) -> Optional[str]:
        """弹出一个 pending wisdom (FIFO)"""
        with self._pending_lock:
            if self._pending_wisdom:
                return self._pending_wisdom.pop(0)
            return None
    
    def trigger(self, memory: AgentMemory, env: Environment) -> bool:
        """主线程调用: 尝试触发 MCTS
        
        Args:
            memory: 主 Agent 的 Memory 实例
            env: 当前环境 (需支持深拷贝或重建)
            
        Returns:
            bool: 任务是否被接受
        """
        # 1. 检查条件
        # 获取最近事件列表 (合并 initial 和 active)
        recent_events = []
        
        # 获取所有事件 (last 10 should be enough for monitoring)
        all_events = []
        if hasattr(memory.hcc.l1, 'get_all_events'):
            all_events = memory.hcc.l1.get_all_events()
        else:
            # Fallback access
            all_events = memory.hcc.l1._initial_events + memory.hcc.l1._active_trace
            
        # Extract content from last 10 events
        recent_events = [e.content for e in all_events[-10:]]
            
        should_run = self.monitor.check_trigger(
            memory_stats=memory.get_stats(),
            recent_events=recent_events
        )
        
        if not should_run:
            return False
        
        # 2. 创建快照 (Snapshot)
        print("[Bridge] Creating memory snapshot...")
        memory_snapshot = memory.create_snapshot()
        
        # 3. 提交任务
        task = {
            "memory": memory_snapshot,
            "env": env,  # 注意: env 也需要在 worker 中是独立的
            "callback_target": memory,  # 原始 memory 实例，用于回调注入
        }
        self._queue.put(task)
        return True

    def _worker_loop(self):
        """后台线程循环"""
        while not self._stop_event.is_set():
            try:
                task = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue
            
            try:
                self._process_task(task)
            except Exception as e:
                print(f"[Bridge] Error processing task: {e}")
            finally:
                self._queue.task_done()
    
    def _process_task(self, task: Dict[str, Any]):
        """处理单个 MCTS 任务"""
        memory_snapshot = task["memory"]
        env = task["env"]
        target_memory = task["callback_target"]
        
        print("[Bridge] Starting System 2 Deep Thinking...")
        
        # 1. 初始化 MCTS
        mcts = self.mcts_factory(env)
        
        # 2. 运行搜索
        # 假设从当前状态开始搜索
        # 这里需要 Environment 能够从 Memory Snapshot 恢复状态
        # 为了简化，假设 env 已经包含了状态
        # 实际工程中，可能需要 env.load_state(memory_snapshot.get_state())
        initial_state = 50 # Mock state for demo compatibility
        
        best_action = mcts.search(initial_state, num_simulations=50)
        
        # 3. 提取搜索轨迹 (用于 L1 记录)
        search_traces = self._extract_search_traces(mcts.root)
        
        # 4. 提炼智慧
        wisdom = self.distiller.distill(mcts, mcts.root, best_action)
        
        # 5. Lazy Update: 将 wisdom 放入队列，而非直接注入
        print(f"[Bridge] Wisdom distilled. Queuing for lazy update...")
        with self._pending_lock:
            self._pending_wisdom.append(wisdom)
        
        # 可选: 同时记录搜索轨迹到快照 memory (不影响主线程)
        for trace in search_traces:
            memory_snapshot.record_event(f"[MCTS] {trace}")
    
    def _extract_search_traces(self, root_node) -> List[str]:
        """从搜索树中提取关键轨迹 (用于 L1 记录)"""
        traces = []
        
        if not root_node or not root_node.children:
            return traces
        
        # 按访问次数排序
        sorted_children = sorted(root_node.children, key=lambda c: c.visits, reverse=True)
        
        # 记录 Top-3 路径
        for i, child in enumerate(sorted_children[:3]):
            if child.visits > 0:
                path = self._trace_path_simple(child)
                q_value = child.q_value
                traces.append(f"Path-{i+1}: {' -> '.join(map(str, path))} (Q={q_value:.2f}, visits={child.visits})")
        
        return traces
    
    def _trace_path_simple(self, node, max_depth: int = 3) -> List[Any]:
        """简化的路径追踪"""
        path = []
        current = node
        
        while current and len(path) < max_depth:
            if current.action is not None:
                path.append(current.action)
            if current.children:
                current = max(current.children, key=lambda c: c.visits)
            else:
                break
        
        return path
    
    # =========================================================================
    # Failure Queue 整合 (架构: Failure Queue → MCTS)
    # =========================================================================
    
    def process_failure_queue(self, env_factory: Callable[[], Environment]) -> int:
        """从 Failure Queue 拉取任务并处理
        
        架构位置: Failure Queue → MCTS Simulator → HCC
        
        Args:
            env_factory: 创建 Environment 的工厂函数
            
        Returns:
            处理的任务数量
        """
        if not self.failure_queue:
            print("[Bridge] No failure queue configured.")
            return 0
        
        processed = 0
        
        while not self.failure_queue.is_empty():
            failure = self.failure_queue.pop()
            if failure is None:
                break
            
            print(f"[Bridge] Processing failure: {failure.task_id}")
            
            try:
                # 1. 创建 Environment
                env = env_factory()
                
                # 2. 运行 MCTS
                mcts = self.mcts_factory(env)
                initial_state = 50  # Mock state
                best_action = mcts.search(initial_state, num_simulations=100)
                
                # 3. 提取轨迹
                search_traces = self._extract_search_traces(mcts.root)
                
                # 4. 提炼 Wisdom
                wisdom = self.distiller.distill(mcts, mcts.root, best_action)
                
                # 5. 写入 HCC (如果有 target_memory)
                if self.target_memory:
                    self._write_to_hcc(
                        wisdom=wisdom,
                        traces=search_traces,
                        failure=failure,
                        success=(mcts.root.q_value > 0.5 if mcts.root else False),
                    )
                
                # 6. 添加到 pending wisdom
                with self._pending_lock:
                    self._pending_wisdom.append(wisdom)
                
                processed += 1
                
            except Exception as e:
                print(f"[Bridge] Error processing failure {failure.task_id}: {e}")
        
        return processed
    
    def _write_to_hcc(
        self,
        wisdom: str,
        traces: List[str],
        failure: Any,  # FailureTrajectory
        success: bool,
    ) -> None:
        """将 MCTS 结果写入 HCC
        
        架构位置: MCTS → HCC Level 1 (Success/Failure)
        
        Args:
            wisdom: 提炼的智慧
            traces: 搜索轨迹
            failure: 原始失败信息
            success: MCTS 是否找到成功路径
        """
        if not self.target_memory:
            return
        
        # 记录到 HCC (通过 inject_wisdom)
        trace_summary = "\n".join(traces[:3])  # 只取 Top-3
        
        if success:
            # Success Path: 写入 L1 并标记可提升
            self.target_memory.inject_wisdom(
                wisdom=f"[MCTS Success] {wisdom}\n\nTask: {failure.query[:50]}...",
                search_traces=traces,
                promote_to_l2=True,  # 成功路径优先提升
            )
            print(f"[Bridge] SUCCESS trajectory written to HCC L1 → L2")
        else:
            # Failure Path: 只写入 L1 备查
            self.target_memory.inject_wisdom(
                wisdom=f"[MCTS Explored] {wisdom}\n\nOriginal Failure: {failure.failure_reason}",
                search_traces=traces,
                promote_to_l2=False,  # 不立即提升
            )
            print(f"[Bridge] FAILURE trajectory written to HCC L1")
    
    # =========================================================================
    # Wisdom Promotion (架构: HCC L1 → L2)
    # =========================================================================
    
    def promote_wisdom(self) -> int:
        """触发 HCC L1 → L2 的知识提升
        
        Returns:
            提升的条目数量
        """
        if not self.target_memory:
            return 0
        
        # 调用 Memory 的 promote 方法
        self.target_memory.promote_phase()
        print("[Bridge] Wisdom promotion triggered (L1 → L2)")
        return 1  # 简化返回
