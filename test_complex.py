#!/usr/bin/env python3
"""
ReactXen 复杂任务测试

测试需要多步推理、反思和工具组合的复杂任务。

使用方式:
    python3 test_complex.py
"""

import os
import sys
from pathlib import Path

project_root = str(Path(__file__).parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from Agent.llm_providers import QwenProvider
from Agent.orchestrator import ReactXenOrchestrator
from Agent.react_agent import Tool
from Memory.tts import TinyTrajectoryStore, Trajectory, TrajectoryStep, TrajectoryCategory


# =============================================================================
# 配置
# =============================================================================

QWEN_CONFIG = {
    "api_key": os.getenv("CLOUD_API_KEY", "sk-a2d89ee90e954ee68e82a387e2d43d69"),
    "model": os.getenv("CLOUD_MODEL", "qwen3-max"),
    "base_url": os.getenv("CLOUD_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
}


# =============================================================================
# 高级工具定义
# =============================================================================

# 模拟数据库
DATABASE = {
    "products": {
        "laptop": {"price": 5999, "stock": 10, "category": "electronics"},
        "mouse": {"price": 99, "stock": 50, "category": "electronics"},
        "keyboard": {"price": 299, "stock": 30, "category": "electronics"},
        "book": {"price": 49, "stock": 100, "category": "office"},
        "notebook": {"price": 15, "stock": 200, "category": "office"},
    },
    "discounts": {
        "electronics": 0.1,  # 10% off
        "office": 0.05,      # 5% off
    },
    "shipping": {
        "standard": 10,
        "express": 30,
    },
}


def create_advanced_tools():
    """创建高级工具集"""
    
    def calculate(expression: str) -> str:
        """安全的数学计算"""
        try:
            # 允许基本数学运算
            allowed = {"__builtins__": {}, "round": round, "abs": abs, "min": min, "max": max}
            result = eval(expression, allowed, {})
            return f"计算结果: {result}"
        except Exception as e:
            return f"计算错误: {e}"
    
    def query_product(product_name: str) -> str:
        """查询产品信息"""
        product_name = product_name.lower().strip()
        if product_name in DATABASE["products"]:
            p = DATABASE["products"][product_name]
            discount = DATABASE["discounts"].get(p["category"], 0)
            return f"产品: {product_name}\n原价: {p['price']}元\n库存: {p['stock']}件\n类别: {p['category']}\n折扣: {int(discount*100)}%"
        return f"未找到产品: {product_name}\n可用产品: {list(DATABASE['products'].keys())}"
    
    def check_inventory(product_name: str) -> str:
        """检查库存"""
        product_name = product_name.lower().strip()
        if product_name in DATABASE["products"]:
            stock = DATABASE["products"][product_name]["stock"]
            return f"{product_name} 库存: {stock} 件"
        return f"未找到产品: {product_name}"
    
    def get_shipping_cost(method: str) -> str:
        """获取运费"""
        method = method.lower().strip()
        if method in DATABASE["shipping"]:
            return f"{method} 运费: {DATABASE['shipping'][method]} 元"
        return f"运费方式: {list(DATABASE['shipping'].keys())}"
    
    def place_order(order_details: str) -> str:
        """模拟下单"""
        return f"订单已提交: {order_details}\n订单号: ORD-2026-{hash(order_details) % 100000:05d}"
    
    return [
        Tool("Calculate", "数学计算。输入: 表达式 (如 100*0.9 或 5999*3)", calculate),
        Tool("QueryProduct", "查询产品信息。输入: 产品名称", query_product),
        Tool("CheckInventory", "检查库存。输入: 产品名称", check_inventory),
        Tool("GetShippingCost", "获取运费。输入: standard 或 express", get_shipping_cost),
        Tool("PlaceOrder", "提交订单。输入: 订单详情", place_order),
    ]


# =============================================================================
# 创建 Few-Shot 示例
# =============================================================================

def create_example_trajectories():
    """创建示例轨迹用于 TTS"""
    tts = TinyTrajectoryStore()
    
    # 示例 1: 多步计算
    example1 = Trajectory(
        trajectory_id="order_calc_1",
        category=TrajectoryCategory.TOOL_USE,
        task="计算购买 2 个笔记本电脑的总价（含 10% 折扣）",
        steps=[
            TrajectoryStep(
                step_id=1,
                thought="首先需要查询笔记本电脑的价格",
                action="QueryProduct",
                action_input="laptop",
                observation="产品: laptop\n原价: 5999元\n库存: 10件\n类别: electronics\n折扣: 10%",
            ),
            TrajectoryStep(
                step_id=2,
                thought="现在计算 2 台电脑的价格，并应用 10% 折扣",
                action="Calculate",
                action_input="5999 * 2 * 0.9",
                observation="计算结果: 10798.2",
            ),
            TrajectoryStep(
                step_id=3,
                thought="总价是 10798.2 元",
                action="Finish",
                action_input="购买 2 个笔记本电脑（含 10% 折扣）的总价是 10798.2 元",
            ),
        ],
        final_answer="购买 2 个笔记本电脑（含 10% 折扣）的总价是 10798.2 元",
    )
    tts.add(example1)
    
    return tts


# =============================================================================
# 复杂测试任务
# =============================================================================

COMPLEX_TASKS = [
    {
        "name": "多产品订单计算",
        "query": """
        我想购买以下商品:
        - 1 个笔记本电脑 (laptop)
        - 2 个键盘 (keyboard)
        - 3 个鼠标 (mouse)
        
        请帮我计算:
        1. 每种商品的小计（应用对应类别的折扣）
        2. 总计金额
        3. 加上快递运费后的最终价格
        """,
        "expected_keywords": ["laptop", "keyboard", "mouse", "运费", "总"],
    },
    {
        "name": "库存检查与决策",
        "query": """
        我需要采购 15 个键盘给公司员工。
        请帮我:
        1. 检查键盘库存是否充足
        2. 计算总价（含折扣）
        3. 如果库存充足，帮我下单
        """,
        "expected_keywords": ["库存", "30", "keyboard"],
    },
    {
        "name": "预算优化问题",
        "query": """
        我有 1000 元预算，想购买尽可能多的办公用品。
        可选商品: book (书，49元) 和 notebook (笔记本，15元)
        办公用品类有 5% 折扣。
        
        请帮我计算如何分配预算，最大化购买数量。
        """,
        "expected_keywords": ["book", "notebook", "预算", "数量"],
    },
]


# =============================================================================
# 主函数
# =============================================================================

def main():
    print("=" * 70)
    print("ReactXen 复杂任务测试")
    print("=" * 70)
    
    # 初始化
    llm = QwenProvider(
        api_key=QWEN_CONFIG["api_key"],
        model=QWEN_CONFIG["model"],
        base_url=QWEN_CONFIG["base_url"],
    )
    
    tools = create_advanced_tools()
    tts = create_example_trajectories()
    
    print(f"[Init] 模型: {QWEN_CONFIG['model']}")
    print(f"[Init] 工具: {[t.name for t in tools]}")
    print(f"[Init] TTS 示例: {len(tts)} 条")
    
    orchestrator = ReactXenOrchestrator(
        llm_provider=llm,
        tts=tts,
        tools=tools,
        max_reflect_iterations=3,
        max_react_steps=10,  # 允许更多步骤
    )
    
    # 运行测试
    for i, task in enumerate(COMPLEX_TASKS, 1):
        print(f"\n{'=' * 70}")
        print(f"任务 {i}: {task['name']}")
        print(f"{'=' * 70}")
        print(f"查询:\n{task['query']}")
        print("-" * 70)
        
        result = orchestrator.run(task["query"])
        
        # 结果
        status = "✅ 成功" if result["success"] else "❌ 失败"
        print(f"\n{status} (迭代: {result['iterations']})")
        print(f"\n最终答案:\n{result['answer']}")
        
        # 简要轨迹
        print(f"\n执行轨迹 ({len(result['trace'])} 步):")
        for step_name, step_result in result["trace"]:
            print(f"  → {step_name}: {step_result.status.value}")
    
    print("\n" + "=" * 70)
    print("测试完成")
    print("=" * 70)


if __name__ == "__main__":
    main()
