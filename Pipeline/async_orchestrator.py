"""
异步 Pipeline Orchestrator - 支持后台 System 2 和惰性更新

核心设计:
1. System 2 作为后台进程，不阻塞主流程
2. 惰性更新：Review Agent 判断完成时批量存储到图数据库
3. 失败队列异步处理
4. System 1 使用 ReactXenOrchestrator 双循环架构
"""

import asyncio
import json
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from queue import Queue, Empty

from config.schema import ExperimentConfig
from core.trajectory import Trajectory
from core.failure import is_failure, create_failure_case, FailureCase
from core.wisdom import Wisdom

# 使用现有 Agent 模块 - ReactXenOrchestrator 双循环架构
from Agent.base import AgentContext, AgentResult, AgentStatus, BaseAgent
from Agent.react_agent import ReActAgent, Tool
from Agent.orchestrator import ReactXenOrchestrator

# 使用现有 LLM 模块
from LLM.llm import LLM, create_llm

# 使用 TTS 作为工作记忆
from Memory.tts import TinyTrajectoryStore, Trajectory as TTSTrajectory, TrajectoryStep

# System 2 策略
from system2.base import SearchStrategy
from system2.mcts import MCTSStrategy
from system2.beam_search import BeamSearchStrategy
from system2.no_search import NoSearchStrategy

# HCC Memory Backend
from Memory.backends.jsonl_backend import JsonlBackend


class FailureQueue:
    """线程安全的失败案例队列"""
    
    def __init__(self, max_size: int = 100):
        self._queue: Queue = Queue(maxsize=max_size)
        self._lock = threading.Lock()
        self._all_failures: List[FailureCase] = []  # 惰性存储缓存
    
    def enqueue(self, case: FailureCase) -> bool:
        try:
            self._queue.put_nowait(case)
            with self._lock:
                self._all_failures.append(case)
            return True
        except:
            return False
    
    def dequeue(self, timeout: float = 0.1) -> Optional[FailureCase]:
        try:
            return self._queue.get(timeout=timeout)
        except Empty:
            return None
    
    def get_all_pending(self) -> List[FailureCase]:
        """获取所有待处理的失败案例（用于惰性更新）"""
        with self._lock:
            result = list(self._all_failures)
            self._all_failures = []
            return result
    
    def __len__(self):
        return self._queue.qsize()


class BackgroundSystem2:
    """后台 System 2 处理器
    
    作为独立线程运行，持续处理失败队列中的案例。
    """
    
    def __init__(
        self, 
        failure_queue: FailureQueue,
        config: ExperimentConfig,
        result_callback: callable,
    ):
        self.failure_queue = failure_queue
        self.config = config
        self.result_callback = result_callback
        
        # 初始化策略
        self.strategy = self._init_strategy()
        
        # 控制标志
        self._running = False
        self._thread: Optional[threading.Thread] = None
        
        # 结果缓存（惰性更新用）
        self._pending_results: List[Trajectory] = []
        self._results_lock = threading.Lock()
    
    def _init_strategy(self) -> Optional[SearchStrategy]:
        if not self.config.system2.enabled:
            return None
            
        strategy_name = self.config.system2.search_strategy
        if strategy_name == "mcts":
            return MCTSStrategy()
        elif strategy_name == "beam":
            return BeamSearchStrategy()
        elif strategy_name in ["none", "no_search"]:
            return NoSearchStrategy()
        return None
    
    def start(self):
        """启动后台处理线程"""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._process_loop, daemon=True)
        self._thread.start()
        print("  [Background] System 2 后台进程已启动")
    
    def stop(self):
        """停止后台处理"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
        print("  [Background] System 2 后台进程已停止")
    
    def _process_loop(self):
        """后台处理循环"""
        while self._running:
            case = self.failure_queue.dequeue(timeout=0.5)
            
            if case is None:
                continue
            
            if self.strategy is None:
                continue
            
            try:
                print(f"  [Background] 处理失败案例: {case.task[:30]}...")
                
                # 执行搜索
                trajectories = self.strategy.search(case, self.config.system2)
                
                # 缓存结果
                with self._results_lock:
                    self._pending_results.extend(trajectories)
                
                # 回调通知
                best = next((t for t in trajectories if t.result == "success"), None)
                if best:
                    self.result_callback("success", best)
                else:
                    self.result_callback("failure", trajectories[0] if trajectories else None)
                    
            except Exception as e:
                print(f"  [Background] 处理失败: {e}")
    
    def get_pending_results(self) -> List[Trajectory]:
        """获取待存储的结果（用于惰性更新）"""
        with self._results_lock:
            results = list(self._pending_results)
            self._pending_results = []
            return results


class AsyncPipelineOrchestrator:
    """异步双系统架构主控制器
    
    核心特性:
    1. System 2 作为后台进程
    2. 惰性更新：Review Agent 判断完成时批量存储
    3. 线程安全的队列和缓存
    
    架构图:
    
    Task → System 1 → [Success] → 成功缓存
                    → [Failure] → 失败队列 → [后台] System 2 → 结果缓存
                    
    Review Agent 判断完成 → 批量存储到图数据库
    """
    
    def __init__(self, config: ExperimentConfig, mock_llm: bool = False):
        self.config = config
        
        # 设置随机种子
        random.seed(config.seed)
        
        # 1. 初始化 LLM
        self.llm = create_llm(mock_mode=mock_llm)
        print(f"  LLM 初始化: {'Mock 模式' if self.llm.is_mock else self.llm.config.model}")
        
        # 2. 初始化 TTS 工作记忆
        self.tts = TinyTrajectoryStore()
        
        # 3. 初始化 System 1 (使用 ReactXenOrchestrator 双循环架构)
        # 包含: Distillation -> ReAct -> Review -> Reflect -> Verification
        self.agent_orchestrator = ReactXenOrchestrator(
            llm_provider=self._create_llm_adapter(),
            tts=self.tts,
            tools=[],  # 可根据需要添加工具
            max_reflect_iterations=config.system1.max_steps // 3,  # 映射到反思轮数
            max_react_steps=config.system1.max_steps,
        )
        print(f"  System 1 初始化: ReactXenOrchestrator (双循环架构)")
        
        # 4. 初始化 HCC Memory
        if config.memory.enabled:
            if config.memory.backend == "jsonl":
                self.memory = JsonlBackend(data_dir=f"{config.evaluation.output_dir}/memory")
            elif config.memory.backend == "neo4j":
                from Memory.backends.neo4j_backend import Neo4jBackend
                self.memory = Neo4jBackend()
            else:
                raise NotImplementedError(f"Backend {config.memory.backend} not implemented")
        else:
            self.memory = None
            
        # 5. 初始化失败队列（线程安全）
        self.failure_queue = FailureQueue(max_size=config.failure.queue_max_size)
        
        # 6. 初始化后台 System 2
        self.background_s2 = BackgroundSystem2(
            failure_queue=self.failure_queue,
            config=config,
            result_callback=self._on_s2_result,
        )
        
        # 7. 准备输出目录
        self.output_dir = Path(config.evaluation.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        config.to_yaml(str(self.output_dir / "config.yaml"))
        
        # 运行统计
        self.run_stats = {
            "total_tasks": 0,
            "success_s1": 0,
            "success_s2": 0,
            "failures": 0,
            "total_tokens": 0,
            "total_steps": 0,
            "wisdom_promoted": 0,
            "lazy_updates": 0,
        }
        
        # ========================================
        # 惰性更新缓存
        # ========================================
        self._pending_success: List[Trajectory] = []  # 待存储的成功轨迹
        self._pending_lock = threading.Lock()
        
        # TTS 压缩摘要
        self.tts_summary = ""
        
        # 启动后台进程
        if config.system2.enabled:
            self.background_s2.start()
    
    def _create_llm_adapter(self):
        """创建 LLM 适配器"""
        class LLMAdapter:
            def __init__(self, llm: LLM):
                self.llm = llm
            
            def generate(self, prompt: str) -> str:
                return self.llm.generate(prompt)
        
        return LLMAdapter(self.llm)
    
    def _on_s2_result(self, status: str, trajectory: Optional[Trajectory]):
        """System 2 结果回调"""
        if status == "success" and trajectory:
            self.run_stats["success_s2"] += 1
            print(f"  [Background] System 2 成功解决!")
        else:
            self.run_stats["failures"] += 1
    
    # =========================================================================
    # 主流程
    # =========================================================================
    
    def run_task(self, task: str) -> Trajectory:
        """执行单个任务（同步，System 2 在后台）
        
        System 1 使用 ReactXenOrchestrator 双循环架构:
        Query -> Distillation -> ReAct -> Review -> [Success?] -> Verification
                                               -> [Failure?] -> Reflect -> ReAct (loop)
        """
        self.run_stats["total_tasks"] += 1
        task_num = self.run_stats["total_tasks"]
        print(f"[Task {task_num}] {task[:50]}...")
        
        # === Step 1: Hot Start (Wisdom 注入) ===
        injected_wisdom = self._hot_start(task)
        
        # === Step 2: System 1 Execution (使用 ReactXenOrchestrator) ===
        # 内部已包含：Distillation, ReAct, Review, Reflect, Verification
        result = self.agent_orchestrator.run(task)
        
        # === Step 3: 转换结果为 Trajectory ===
        trajectory = self._orchestrator_result_to_trajectory(task, result, injected_wisdom)
        
        # === Step 4: TTS 压缩 (Markov 约束) ===
        if self.config.system1.markov_window > 0:
            self._compress_tts_from_trace(result.get("trace", []))
        
        # === Step 5: 分流 ===
        if result["success"]:
            # Success - 加入待存储缓存
            print(f"  ✓ System 1 Success (iterations: {result['iterations']})")
            self.run_stats["success_s1"] += 1
            with self._pending_lock:
                self._pending_success.append(trajectory)
        else:
            # Failure - 加入失败队列，后台处理
            print(f"  ✗ System 1 Failed (iterations: {result['iterations']}) -> 后台 System 2")
            failure = create_failure_case(
                trajectory, 
                soft_threshold=self.config.failure.soft_failure_threshold
            )
            if failure:
                self.failure_queue.enqueue(failure)
        
        return trajectory
    
    def _orchestrator_result_to_trajectory(
        self, 
        task: str, 
        result: Dict,
        injected_wisdom: List[str],
    ) -> Trajectory:
        """将 ReactXenOrchestrator 结果转换为 Trajectory"""
        trajectory = Trajectory(task=task)
        
        # 从 trace 中提取步骤
        for step_name, step_result in result.get("trace", []):
            trajectory.add_step(
                thought=f"[{step_name}] {step_result.reasoning[:200] if step_result.reasoning else ''}...",
                action=step_name,
                observation=step_result.output[:200] if step_result.output else "",
            )
        
        # 设置结果状态
        if result["success"]:
            trajectory.mark_success()
        else:
            trajectory.mark_failure()
        
        # 记录 wisdom 使用
        trajectory.used_wisdom = bool(injected_wisdom)
        if injected_wisdom:
            trajectory.metadata["injected_wisdom"] = injected_wisdom
        
        trajectory.metadata["iterations"] = result["iterations"]
        trajectory.metadata["answer"] = result["answer"]
        
        self._update_stats(trajectory)
        return trajectory
    
    def _compress_tts_from_trace(self, trace: List) -> None:
        """从执行 trace 中压缩 TTS"""
        if not trace:
            return
        
        # 构建轨迹文本
        trajectory_text = "\n".join([
            f"{name}: {r.output[:100]}..." 
            for name, r in trace 
            if r.output
        ])
        
        if trajectory_text:
            new_summary = self.llm.compress_trajectory(
                current_trajectory=trajectory_text,
                previous_summary=self.tts_summary,
            )
            self.tts_summary = new_summary
    
    # =========================================================================
    # 惰性更新 (Review Agent 调用)
    # =========================================================================
    
    def flush_to_memory(self) -> Dict[str, int]:
        """惰性更新：批量存储到图数据库
        
        由 Review Agent 在判断任务完成时调用。
        将所有待存储的成功轨迹和 System 2 结果存入 Memory。
        
        Returns:
            存储统计 {"success": n, "s2_results": m, "failures": k}
        """
        if not self.memory:
            return {"success": 0, "s2_results": 0, "failures": 0}
        
        stats = {"success": 0, "s2_results": 0, "failures": 0}
        
        print("\n[Lazy Update] 批量存储到图数据库...")
        
        # 1. 存储成功轨迹
        with self._pending_lock:
            success_trajs = list(self._pending_success)
            self._pending_success = []
        
        if success_trajs:
            self.memory.put_samples(success_trajs)
            stats["success"] = len(success_trajs)
            print(f"  - 成功轨迹: {len(success_trajs)} 条")
        
        # 2. 存储 System 2 结果
        s2_results = self.background_s2.get_pending_results()
        if s2_results:
            self.memory.put_samples(s2_results)
            stats["s2_results"] = len(s2_results)
            print(f"  - System 2 结果: {len(s2_results)} 条")
        
        # 3. 存储失败案例（用于分析）
        failures = self.failure_queue.get_all_pending()
        if failures:
            # 将失败案例的轨迹也存入（标记为 failure）
            failure_trajs = [f.trajectory for f in failures if f.trajectory]
            if failure_trajs:
                self.memory.put_samples(failure_trajs)
            stats["failures"] = len(failures)
            print(f"  - 失败轨迹: {len(failures)} 条")
        
        # 4. 尝试 Promotion
        self._try_promote()
        
        # 5. 落盘 JSONL
        for traj in success_trajs + s2_results:
            self._save_trajectory(traj)
        
        self.run_stats["lazy_updates"] += 1
        print(f"[Lazy Update] 完成 (第 {self.run_stats['lazy_updates']} 次)")
        
        return stats
    
    # =========================================================================
    # 辅助方法
    # =========================================================================
    
    def _hot_start(self, task: str) -> List[str]:
        """Hot Start: 从 Memory 检索 Wisdom"""
        if not self.config.memory.enabled or not self.config.memory.enable_hot_start:
            return []
        
        if not self.memory:
            return []
        
        wisdom_objs = self.memory.retrieve(task, k=self.config.system1.wisdom_k)
        return [w.text for w in wisdom_objs]
    
    def _get_tts_examples(self, task: str) -> str:
        """从 TTS 获取 Few-Shot 示例"""
        if self.tts.count() == 0:
            return ""
        
        examples = self.tts.retrieve(query=task, k=3)
        if not examples:
            return ""
        
        lines = ["Here are some relevant examples:"]
        for i, traj in enumerate(examples, 1):
            lines.append(f"\n--- Example {i} ---")
            lines.append(traj.to_text(max_steps=5))
        
        return "\n".join(lines)
    
    def _update_tts(self, trajectory: Trajectory, context: AgentContext) -> None:
        """将成功轨迹添加到 TTS"""
        if trajectory.result != "success":
            return
        
        steps = []
        for i, step in enumerate(trajectory.steps, 1):
            tts_step = TrajectoryStep(
                step_id=i,
                thought=step.thought,
                action=step.action.split(":")[0] if ":" in step.action else step.action,
                action_input=step.action.split(":", 1)[1].strip() if ":" in step.action else "",
                observation=step.observation,
            )
            steps.append(tts_step)
        
        tts_traj = TTSTrajectory(
            trajectory_id=trajectory.id,
            task=trajectory.task,
            steps=steps,
            is_successful=True,
            metadata={"source": "pipeline"}
        )
        
        try:
            self.tts.add(tts_traj)
        except ValueError:
            pass
    
    def _compress_tts(self, context: AgentContext) -> None:
        """TTS 压缩"""
        if not context.scratchpad:
            return
        
        new_summary = self.llm.compress_trajectory(
            current_trajectory=context.scratchpad,
            previous_summary=self.tts_summary,
        )
        
        self.tts_summary = new_summary
    
    def _agent_result_to_trajectory(
        self, 
        task: str, 
        result: AgentResult, 
        context: AgentContext,
        injected_wisdom: List[str],
    ) -> Trajectory:
        """将 AgentResult 转换为 Trajectory"""
        trajectory = Trajectory(task=task)
        
        if context.scratchpad:
            trajectory.add_step(
                thought=context.scratchpad,
                action="ReAct",
                observation=result.output,
            )
        else:
            trajectory.add_step(
                thought=result.reasoning,
                action="Execute",
                observation=result.output,
            )
        
        if result.status == AgentStatus.SUCCESS:
            trajectory.mark_success()
        elif result.status == AgentStatus.NEEDS_REVIEW:
            trajectory.metadata["needs_review"] = True
            trajectory.metadata["confidence"] = 0.7
        else:
            trajectory.mark_failure()
        
        trajectory.used_wisdom = bool(injected_wisdom)
        if injected_wisdom:
            trajectory.metadata["injected_wisdom"] = injected_wisdom
        
        self._update_stats(trajectory)
        return trajectory
    
    def _try_promote(self):
        """尝试 Promotion"""
        if not self.memory or not self.config.memory.enable_promotion:
            return
        
        new_wisdom = self.memory.promote(
            min_samples=self.config.memory.promotion_min_samples
        )
        if new_wisdom:
            print(f"  ↳ Promoted {len(new_wisdom)} new wisdom")
            self.run_stats["wisdom_promoted"] += len(new_wisdom)
            self._save_wisdom(new_wisdom)
    
    def _update_stats(self, trajectory: Trajectory):
        """更新统计"""
        self.run_stats["total_tokens"] += trajectory.cost.tokens
        self.run_stats["total_steps"] += trajectory.cost.steps
    
    def _save_trajectory(self, trajectory: Trajectory):
        """落盘轨迹"""
        file_path = self.output_dir / "trajectories.jsonl"
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(trajectory.to_json() + "\n")
    
    def _save_wisdom(self, wisdom_list: List[Wisdom]):
        """落盘 Wisdom"""
        file_path = self.output_dir / "wisdom.jsonl"
        with open(file_path, "a", encoding="utf-8") as f:
            for w in wisdom_list:
                f.write(w.to_json() + "\n")
    
    # =========================================================================
    # 批量执行
    # =========================================================================
    
    def run_batch(self, tasks: List[str], flush_interval: int = 10) -> Dict[str, Any]:
        """批量执行任务，定期惰性更新"""
        for i, task in enumerate(tasks, 1):
            self.run_task(task)
            
            # 定期惰性更新
            if i % flush_interval == 0:
                self.flush_to_memory()
        
        # 最终更新
        self.flush_to_memory()
        
        return self.get_metrics()
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取评测指标"""
        total = self.run_stats["total_tasks"]
        if total == 0:
            return self.run_stats
        
        metrics = dict(self.run_stats)
        metrics["success_rate"] = (metrics["success_s1"] + metrics["success_s2"]) / total
        metrics["s1_success_rate"] = metrics["success_s1"] / total
        metrics["s2_success_rate"] = metrics["success_s2"] / total
        metrics["avg_steps"] = metrics["total_steps"] / total
        
        return metrics
    
    def close(self):
        """清理资源"""
        # 停止后台进程
        self.background_s2.stop()
        
        # 最终惰性更新
        self.flush_to_memory()
        
        # 保存 metrics
        metrics = self.get_metrics()
        metrics_path = self.output_dir / "metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        
        print(f"\n=== Experiment Complete ===")
        print(f"Success Rate: {metrics.get('success_rate', 0):.2%}")
        print(f"S1 Success: {metrics['success_s1']}, S2 Success: {metrics['success_s2']}")
        print(f"Failures: {metrics['failures']}")
        print(f"Lazy Updates: {metrics['lazy_updates']}")
        print(f"Results saved to: {self.output_dir}")
