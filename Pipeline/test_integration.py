"""
Integration Test for Async MCTS Bridge
"""

import sys
import os
import time
import unittest
from pathlib import Path

# 添加项目根目录到路径
project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

from Memory.memory import AgentMemory, MemoryConfig, EventType
from MCTS.mcts import VanillaMCTS
from MCTS.config import MCTSConfig
from Pipeline.bridge import AsyncMCTSBridge, MonitorConfig


# Mock Environment for testing
class MockEnv:
    def __init__(self, target=42):
        self.target = target
    
    def get_actions(self, state): return [-1, 1]
    def step(self, state, action): return state + action
    def is_terminal(self, state): return state == self.target
    def get_reward(self, state): return 1.0 if state == self.target else 0.0
    def get_prior(self, state, action): return 0.5


class TestAsyncBridge(unittest.TestCase):
    
    def test_end_to_end_flow(self):
        print("\n=== Testing Async MCTS Integration ===")
        
        # 1. Setup System 1 (Memory)
        print("[Test] Initializing Memory...")
        memory = AgentMemory()
        memory.start_task("Find the number 42")
        
        # Simulate some failures to trigger monitor
        for _ in range(3):
            memory.record_event("Execution failed: timeout", EventType.ENVIRONMENT)
            
        print(f"[Test] Memory Stats: {memory.get_stats()}")
        
        # 2. Setup System 2 (Bridge)
        print("[Test] Initializing Bridge...")
        
        def mcts_factory(env):
            return VanillaMCTS(env, max_depth=5)
            
        monitor_config = MonitorConfig(consecutive_failure_threshold=3)
        bridge = AsyncMCTSBridge(mcts_factory, monitor_config)
        
        # 3. Trigger Flow
        print("[Test] Triggering Bridge...")
        env = MockEnv()
        triggered = bridge.trigger(memory, env)
        self.assertTrue(triggered, "Bridge should be triggered by 3 consecutive failures")
        
        # 4. Wait for Async Execution
        print("[Test] Waiting for Async Worker...")
        # L3 count should increase from 0 to 1
        initial_l3_count = len(memory.hcc.l3)
        
        # Poll for result (max 5 seconds)
        start_time = time.time()
        while len(memory.hcc.l3) == initial_l3_count:
            if time.time() - start_time > 5:
                break
            time.sleep(0.1)
            
        # 5. Verify Result
        final_l3_count = len(memory.hcc.l3)
        print(f"[Test] L3 Count: {initial_l3_count} -> {final_l3_count}")
        
        self.assertEqual(final_l3_count, initial_l3_count + 1, "Wisdom should be injected into L3")
        
        wisdom_entry = memory.hcc.l3.get_all_entries()[-1]
        print(f"[Test] Injected Wisdom: {wisdom_entry.wisdom}")
        self.assertIn("MCTS Suggestion", wisdom_entry.wisdom)
        
        # 6. Clean up
        bridge.stop()
        print("[Test] Passed!")


if __name__ == "__main__":
    unittest.main()
