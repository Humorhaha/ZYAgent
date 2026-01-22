#!/usr/bin/env python3
"""
ReactXen 集成测试脚本

使用真实 LLM API 测试完整的 Agent 编排流程。

使用方式:
    # 设置环境变量
    export CLOUD_API_KEY="sk-xxx"
    
    # 运行测试
    python3 test_reactxen.py
"""

import os
import sys
from pathlib import Path

# 添加项目根目录
project_root = str(Path(__file__).parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from Agent.llm_providers import QwenProvider
from Agent.orchestrator import ReactXenOrchestrator
from Agent.react_agent import Tool
from Memory.tts import TinyTrajectoryStore


# =============================================================================
# 配置
# =============================================================================

# Qwen API 配置
QWEN_CONFIG = {
    "api_key": os.getenv("CLOUD_API_KEY", "sk-a2d89ee90e954ee68e82a387e2d43d69"),
    "model": os.getenv("CLOUD_MODEL", "qwen3-max"),
    "base_url": os.getenv("CLOUD_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
}


# =============================================================================
# 工具定义
# =============================================================================

def create_tools():
    """创建测试工具"""
    
    def calculate(expression: str) -> str:
        """数学计算"""
        try:
            # 安全地执行计算
            result = eval(expression, {"__builtins__": {}}, {})
            return f"计算结果: {result}"
        except Exception as e:
            return f"计算错误: {e}"
    
    def search(query: str) -> str:
        """模拟搜索"""
        # 模拟一些预设答案
        knowledge = {
            "python": "Python 是一种高级编程语言，以简洁易读著称。",
            "机器学习": "机器学习是人工智能的一个分支，通过数据训练模型。",
            "reactxen": "ReactXen 是一个多 Agent 系统框架，支持反思和自我纠错。",
        }
        
        for key, value in knowledge.items():
            if key.lower() in query.lower():
                return f"搜索结果: {value}"
        
        return f"搜索结果: 未找到关于 '{query}' 的详细信息，但可以基于常识回答。"
    
    return [
        Tool("Calculate", "进行数学计算。输入: 数学表达式 (如 2+2 或 3*4)", calculate),
        Tool("Search", "搜索信息。输入: 搜索关键词", search),
    ]


# =============================================================================
# 测试用例
# =============================================================================

TEST_CASES = [
    {
        "name": "简单计算",
        "query": "请计算 25 * 4 的结果",
        "expected_keywords": ["100"],
    },
    {
        "name": "复杂计算",
        "query": "计算 (15 + 25) * 2 - 10 等于多少",
        "expected_keywords": ["70"],
    },
    {
        "name": "知识问答",
        "query": "什么是 Python 编程语言？请简要介绍",
        "expected_keywords": ["Python", "编程"],
    },
    {
        "name": "推理任务",
        "query": "如果一个苹果 3 元，买 5 个需要多少钱？",
        "expected_keywords": ["15"],
    },
]


# =============================================================================
# 测试运行器
# =============================================================================

class TestRunner:
    """测试运行器"""
    
    def __init__(self, orchestrator: ReactXenOrchestrator):
        self.orchestrator = orchestrator
        self.results = []
    
    def run_test(self, test_case: dict) -> dict:
        """运行单个测试用例"""
        print(f"\n{'=' * 60}")
        print(f"测试: {test_case['name']}")
        print(f"查询: {test_case['query']}")
        print('=' * 60)
        
        result = self.orchestrator.run(test_case["query"])
        
        # 检查结果
        answer = result.get("answer", "")
        expected = test_case.get("expected_keywords", [])
        passed = any(kw in answer for kw in expected) if expected else result["success"]
        
        test_result = {
            "name": test_case["name"],
            "query": test_case["query"],
            "answer": answer,
            "success": result["success"],
            "iterations": result["iterations"],
            "passed": passed,
        }
        
        self.results.append(test_result)
        
        # 打印结果
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"\n{status}")
        print(f"答案: {answer[:200]}...")
        print(f"迭代次数: {result['iterations']}")
        
        return test_result
    
    def run_all(self, test_cases: list) -> dict:
        """运行所有测试用例"""
        print("\n" + "=" * 60)
        print("ReactXen 集成测试")
        print("=" * 60)
        
        for test_case in test_cases:
            self.run_test(test_case)
        
        # 汇总
        passed = sum(1 for r in self.results if r["passed"])
        total = len(self.results)
        
        print("\n" + "=" * 60)
        print("测试汇总")
        print("=" * 60)
        print(f"通过: {passed}/{total}")
        
        for r in self.results:
            status = "✅" if r["passed"] else "❌"
            print(f"  {status} {r['name']}")
        
        return {
            "passed": passed,
            "total": total,
            "results": self.results,
        }


# =============================================================================
# 主函数
# =============================================================================

def main():
    print("=" * 60)
    print("ReactXen 集成测试 - 使用 Qwen API")
    print("=" * 60)
    
    # 1. 初始化 LLM
    print(f"\n[Init] 使用模型: {QWEN_CONFIG['model']}")
    llm = QwenProvider(
        api_key=QWEN_CONFIG["api_key"],
        model=QWEN_CONFIG["model"],
        base_url=QWEN_CONFIG["base_url"],
    )
    
    # 2. 创建工具
    tools = create_tools()
    print(f"[Init] 已加载 {len(tools)} 个工具")
    
    # 3. 创建编排器
    orchestrator = ReactXenOrchestrator(
        llm_provider=llm,
        tools=tools,
        max_reflect_iterations=2,
        max_react_steps=5,
    )
    
    # 4. 运行测试
    runner = TestRunner(orchestrator)
    summary = runner.run_all(TEST_CASES)
    
    # 5. 返回状态码
    return 0 if summary["passed"] == summary["total"] else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
