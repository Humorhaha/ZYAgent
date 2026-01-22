"""
Pipeline Unit Tests - 逐模块测试 Bridge 组件
"""

import sys
import os
import time
import unittest
from pathlib import Path

project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

from Memory.memory import AgentMemory, EventType
from MCTS.mcts import VanillaMCTS
from Pipeline.bridge import Monitor, MonitorConfig, WisdomDistiller, AsyncMCTSBridge


# =============================================================================
# Mock Environment
# =============================================================================

class MockEnv:
    """Mock 环境"""
    def __init__(self, target=42):
        self.target = target
    
    def get_actions(self, state): return [-1, 1]
    def step(self, state, action): return state + action
    def is_terminal(self, state): return state == self.target
    def get_reward(self, state): return 1.0 if state == self.target else 0.0
    def get_prior(self, state, action): return 0.5


# =============================================================================
# 1. Monitor Tests
# =============================================================================

class TestMonitor(unittest.TestCase):
    """Monitor 单元测试"""
    
    def test_no_trigger_on_empty_events(self):
        """无事件时不应触发"""
        monitor = Monitor()
        result = monitor.check_trigger(
            memory_stats={},
            recent_events=[],
            next_token_entropy=0.0
        )
        self.assertFalse(result, "Should not trigger on empty events")
    
    def test_trigger_on_high_entropy(self):
        """高熵时应触发"""
        monitor = Monitor(MonitorConfig(entropy_threshold=0.5))
        result = monitor.check_trigger(
            memory_stats={},
            recent_events=[],
            next_token_entropy=0.9  # 高于阈值
        )
        self.assertTrue(result, "Should trigger on high entropy")
    
    def test_trigger_on_consecutive_failures(self):
        """连续失败时应触发"""
        monitor = Monitor(MonitorConfig(consecutive_failure_threshold=2))
        
        # 两个连续失败
        result = monitor.check_trigger(
            memory_stats={},
            recent_events=["task failed", "error occurred"]
        )
        self.assertTrue(result, "Should trigger on 2 consecutive failures")
    
    def test_no_trigger_on_interrupted_failures(self):
        """失败被成功打断时不应触发"""
        monitor = Monitor(MonitorConfig(consecutive_failure_threshold=3))
        
        # 失败被成功打断
        result = monitor.check_trigger(
            memory_stats={},
            recent_events=["error", "success", "error"]  # 只有 1 个连续失败
        )
        self.assertFalse(result, "Should not trigger when failure chain is broken")
    
    def test_consecutive_failures_count_from_end(self):
        """连续失败应从末尾计数"""
        monitor = Monitor(MonitorConfig(consecutive_failure_threshold=2))
        
        # 末尾两个是失败
        result = monitor.check_trigger(
            memory_stats={},
            recent_events=["success", "error", "failed"]
        )
        self.assertTrue(result, "Should count failures from the end")
        self.assertEqual(monitor._consecutive_failures, 2)


# =============================================================================
# 2. WisdomDistiller Tests
# =============================================================================

class TestWisdomDistiller(unittest.TestCase):
    """WisdomDistiller 单元测试"""
    
    def test_distill_returns_string(self):
        """提炼结果应为字符串"""
        distiller = WisdomDistiller()
        env = MockEnv()
        mcts = VanillaMCTS(env, max_depth=3)
        mcts.search(50, num_simulations=10)
        
        wisdom = distiller.distill(mcts, mcts.root, 1)
        
        self.assertIsInstance(wisdom, str)
        self.assertTrue(len(wisdom) > 0)
    
    def test_distill_with_empty_root(self):
        """空根节点应返回默认建议"""
        distiller = WisdomDistiller()
        
        # 模拟一个没有子节点的根
        class MockRoot:
            children = []
            state = "mock_state"
            visits = 0
        
        wisdom = distiller.distill(None, MockRoot(), None)
        self.assertEqual(wisdom, "Consider alternative approaches based on search results.")


# =============================================================================
# 3. AsyncMCTSBridge Tests
# =============================================================================

class TestAsyncMCTSBridge(unittest.TestCase):
    """AsyncMCTSBridge 单元测试"""
    
    def setUp(self):
        def mcts_factory(env):
            return VanillaMCTS(env, max_depth=5)
        
        self.bridge = AsyncMCTSBridge(mcts_factory, MonitorConfig(consecutive_failure_threshold=2))
    
    def tearDown(self):
        self.bridge.stop()
    
    def test_bridge_starts_worker_thread(self):
        """Bridge 应启动后台线程"""
        self.assertTrue(self.bridge._worker_thread.is_alive())
    
    def test_trigger_without_failures_returns_false(self):
        """无失败时 trigger 应返回 False"""
        memory = AgentMemory()
        memory.start_task("test")
        memory.record_event("success", EventType.ENVIRONMENT)
        
        env = MockEnv()
        triggered = self.bridge.trigger(memory, env)
        
        self.assertFalse(triggered, "Should not trigger without failures")
    
    def test_trigger_with_failures_returns_true(self):
        """有连续失败时 trigger 应返回 True"""
        memory = AgentMemory()
        memory.start_task("test")
        memory.record_event("error 1", EventType.ENVIRONMENT)
        memory.record_event("failed 2", EventType.ENVIRONMENT)
        
        env = MockEnv()
        triggered = self.bridge.trigger(memory, env)
        
        self.assertTrue(triggered, "Should trigger with consecutive failures")
    
    def test_full_async_flow(self):
        """完整异步流程测试"""
        memory = AgentMemory()
        memory.start_task("Find 42")
        
        # 触发条件
        for _ in range(2):
            memory.record_event("error", EventType.ENVIRONMENT)
        
        env = MockEnv()
        triggered = self.bridge.trigger(memory, env)
        self.assertTrue(triggered)
        
        # 等待结果 - 现在检查 L2 而非 L3
        initial_l2_count = len(memory.hcc.l2)
        start = time.time()
        while len(memory.hcc.l2) == initial_l2_count and time.time() - start < 5:
            time.sleep(0.1)
        
        # MCTS 经验应该被记录到 L2
        self.assertGreater(len(memory.hcc.l2), initial_l2_count)


# =============================================================================
# 4. Edge Case Tests
# =============================================================================

class TestEdgeCases(unittest.TestCase):
    """边界条件测试"""
    
    def test_multiple_triggers_queue_tasks(self):
        """多次触发应排队任务"""
        def mcts_factory(env):
            return VanillaMCTS(env, max_depth=3)
        
        bridge = AsyncMCTSBridge(mcts_factory, MonitorConfig(consecutive_failure_threshold=1))
        
        memory = AgentMemory()
        memory.start_task("test")
        memory.record_event("error", EventType.ENVIRONMENT)
        
        env = MockEnv()
        
        # 连续触发 3 次
        for _ in range(3):
            bridge.trigger(memory, env)
        
        # 等待所有任务完成
        time.sleep(3)
        
        # 由于新的渐进式提升设计，MCTS 经验会记录到 L2
        # 每次触发应该增加一个 L2 knowledge unit
        self.assertGreaterEqual(len(memory.hcc.l2), 1)
        
        bridge.stop()


if __name__ == "__main__":
    print("=" * 60)
    print("Pipeline Unit Tests")
    print("=" * 60)
    unittest.main(verbosity=2)
