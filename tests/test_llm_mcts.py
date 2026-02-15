"""
LLM Integration Test - 验证 LLM 接入和 MCTS 集成
此测试旨在验证:
1. LLM 能否正确加载配置 (.env)
2. LLM 能否进行真实调用 (与 Mock 模式区分)
3. MCTSAdapter 是否真正使用了 LLM 生成内容 (基于 Prompt 模板)
4. 解析逻辑是否健壮 (针对 JSON 输出)

使用 unittest 以避免额外依赖。
"""

import unittest
import re
from unittest.mock import MagicMock
from typing import Optional

from LLM.llm import LLM, create_llm
from system2.mcts import MCTSAdapter
from core.failure import FailureCase
from core.trajectory import Trajectory
from config import ExperimentConfig

class TestLLMIntegration(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """创建 LLM 实例 (优先尝试真实调用)"""
        cls.llm = create_llm()
        if cls.llm.is_mock:
            print("\n[WARNING] LLM is in Mock mode. Ensure LLM/.env exists and has API key.")
        else:
            print(f"\n[INFO] LLM Configured: model={cls.llm.config.model}, base_url={cls.llm.config.base_url}")
            
    def setUp(self):
        """每个测试前初始化 Adapter"""
        # IoT 场景任务: 诊断传感器异常
        self.failure_case = MagicMock()
        self.failure_case.task = "Diagnose intermittent 0-value readings from Temperature Sensor T-204 on Boiler B."
        
        # 模拟配置
        self.config = ExperimentConfig()
        
        # 实例化 Adapter
        self.adapter = MCTSAdapter(self.failure_case, self.config.system2, llm=self.llm)

    def test_llm_connectivity(self):
        """测试 LLM 基本连通性"""
        if self.llm.is_mock:
            print("Skipping connectivity test in Mock mode")
            return
            
        prompt = "Hello, reply with the single word 'World' only."
        response = self.llm.generate(prompt)
        print(f"\n[LLM Response] {response}")
        
        self.assertIsNotNone(response)
        self.assertTrue(len(response.strip()) > 0)
        self.assertTrue("World" in response or "world" in response)

    def test_get_next_step_structure(self):
        """测试下一步生成 (MCTS_NEXT_STEP)"""
        if self.llm.is_mock:
            print("Skipping LLM generation test in Mock mode")
            return
            
        history = "Step 1: Check sensor connection status. Action: run_command('ping sensor-t204-gateway')"
        
        action = self.adapter.get_next_step(history, step_n=2)
        print(f"\n[MCTS Next Step] {action}")
        
        # 验证输出格式包含 Thought 和 Action
        self.assertIn("Thought:", action)
        self.assertIn("Action:", action)
        # 验证内容与 IoT 相关
        self.assertTrue(any(kw in action.lower() for kw in ['log', 'status', 'connection', 'voltage', 'check', 'read', 'packet', 'ping']))

    def test_get_next_step_with_reflection(self):
        """测试带反思的下一步生成 (MCTS_NEXT_STEP_WITH_REFLECTION)"""
        if self.llm.is_mock:
            return

        history = "Step 1: Checked logs. Found timeout errors."
        reflection = "Don't just check logs, also verify physical link integration."
        
        # FIX: 参数顺序 (history, step_n, reflection)
        action = self.adapter.get_next_step_use_reflection(history, 2, reflection)
        print(f"\n[MCTS Next Step (Reflected)] {action}")
        
        self.assertIn("Thought:", action)
        # 验证是否采纳了反思 (提及物理连接/链路)
        self.assertTrue(any(kw in action.lower() for kw in ['physical', 'link', 'cable', 'connection', 'wire']))

    def test_multimodal_iot_prompts(self):
        """测试多模态 IoT 任务生成 (MCTS_NEXT_STEP)"""
        if self.llm.is_mock:
            return

        # Case 1: Diagnosis
        h1 = "Step 1: check_connection('gateway') -> Timeout"
        a1 = self.adapter.get_next_step(h1, 2)
        print(f"\n[Diagnosis Step] {a1}")
        self.assertTrue("Thought" in a1)

        # Case 2: QA
        # 我们临时修改 task 来模拟 QA 场景
        original_task = self.adapter.failure_case.task
        self.adapter.failure_case.task = "Explain the difference between MQTT QoS 0, 1, and 2."
        a2 = self.adapter.get_next_step("", 1)
        print(f"\n[QA Step] {a2}")
        self.assertTrue(any(kw in a2.lower() for kw in ['search', 'knowledge', 'doc', 'manual']))
        
        # Case 3: Prediction
        self.adapter.failure_case.task = "Predict the boiler pressure trend for the next 24 hours."
        a3 = self.adapter.get_next_step("", 1)
        print(f"\n[Prediction Step] {a3}")
        self.assertTrue(any(kw in a3.lower() for kw in ['history', 'data', 'trend', 'model', 'time']))

        # Restore
        self.adapter.failure_case.task = original_task

    def test_reflection_multimodal(self):
        """测试多模态反思 (MCTS_REFLECTION)"""
        if self.llm.is_mock:
            return
            
        # 模拟一个"知识幻觉"的 QA 场景
        self.adapter.failure_case.task = " Explain Modbus Error Code 04"
        history = "Step 1: Thought: Code 04 means Device Busy. Action: Reply to user."
        # 实际 04 是 Slave Device Failure，06 才是 Busy。LLM 应该指出没查文档。
        
        ref = self.adapter.get_reflection(history, 2)
        print(f"\n[QA Reflection] {ref}")
        self.assertTrue(any(kw in ref.lower() for kw in ['knowledge', 'search', 'verify', 'hallucination', 'check', 'doc', '幻觉', '查']))

    def test_get_step_value_parsing(self):
        """测试价值评估及分数解析 (MCTS_VALUE_EVALUATION)"""
        # 模拟一个接近解决的 History
        history = """
        Step 1: Analyzed error logs. Found CRC errors.
        Step 2: Inspected extensive cabling. Found loose shielded wire on T-204.
        Step 3: Re-crimped the connector.
        """
        
        val = self.adapter.get_step_value(history)
        print(f"\n[MCTS Value] {val}")
        
        if self.llm.is_mock:
            self.assertEqual(val, 0.5)
        else:
            self.assertIsInstance(val, float)
            self.assertTrue(0.0 <= val <= 1.0)
            
            # 如果解析得到 0.5 (默认值)，且模型输出了有效文本但 regex 没匹配到，这里可能会失败
            # 我们先尝试断言 > 0.6，如果失败则说明 parsing 需要改进
            if val == 0.5:
                print("[WARNING] Value parsing might have failed (returned default 0.5)")
            
            self.assertTrue(val > 0.5, f"Value {val} is too low for a solved case")

if __name__ == "__main__":
    unittest.main(verbosity=2)
