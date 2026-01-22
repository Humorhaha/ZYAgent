#!/usr/bin/env python3
"""
ReactXen 主入口 - 生产环境使用

使用方式:
    # 设置环境变量
    export OPENAI_API_KEY="sk-xxx"
    
    # 运行
    python3 main.py "你的问题"
    
或在 Python 中使用:
    from Agent import ReactXenOrchestrator, OpenAIProvider
    
    llm = OpenAIProvider(api_key="sk-xxx")
    orchestrator = ReactXenOrchestrator(llm_provider=llm)
    result = orchestrator.run("你的问题")
"""

import sys
import argparse
from pathlib import Path

# 添加项目根目录
project_root = str(Path(__file__).parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def create_default_tools():
    """创建默认工具集"""
    from Agent.react_agent import Tool
    
    def search(query: str) -> str:
        """模拟搜索 (生产环境替换为真实搜索 API)"""
        return f"搜索结果: 找到了关于 '{query}' 的相关信息。"
    
    def calculate(expression: str) -> str:
        """计算器工具"""
        try:
            result = eval(expression)
            return f"计算结果: {result}"
        except Exception as e:
            return f"计算错误: {e}"
    
    def read_file(filepath: str) -> str:
        """读取文件内容"""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            return f"文件内容:\n{content[:2000]}"
        except Exception as e:
            return f"读取文件失败: {e}"
    
    return [
        Tool("Search", "搜索信息。输入: 搜索关键词", search),
        Tool("Calculate", "进行数学计算。输入: 数学表达式", calculate),
        Tool("ReadFile", "读取文件内容。输入: 文件路径", read_file),
    ]


def main():
    parser = argparse.ArgumentParser(
        description="ReactXen Multi-Agent System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python3 main.py "什么是机器学习?"
    python3 main.py --provider openai --model gpt-4 "解释一下量子计算"
    python3 main.py --provider deepseek "帮我分析这个问题"
        """,
    )
    
    parser.add_argument("query", nargs="?", help="用户查询")
    parser.add_argument("--provider", choices=["openai", "anthropic", "deepseek"], 
                       default="openai", help="LLM 提供者")
    parser.add_argument("--model", help="模型名称")
    parser.add_argument("--api-key", help="API Key (或使用环境变量)")
    parser.add_argument("--max-iterations", type=int, default=3, 
                       help="最大反思迭代次数")
    parser.add_argument("--verbose", action="store_true", help="详细输出")
    
    args = parser.parse_args()
    
    # 交互模式
    if not args.query:
        print("=" * 60)
        print("ReactXen Multi-Agent System")
        print("=" * 60)
        print("输入 'quit' 或 'exit' 退出\n")
        
        # 初始化
        llm, orchestrator = initialize(args)
        
        while True:
            try:
                query = input("\n> 你的问题: ").strip()
                if query.lower() in ("quit", "exit", "q"):
                    print("再见!")
                    break
                if not query:
                    continue
                
                result = orchestrator.run(query)
                print_result(result, args.verbose)
                
            except KeyboardInterrupt:
                print("\n再见!")
                break
    else:
        # 单次查询模式
        llm, orchestrator = initialize(args)
        result = orchestrator.run(args.query)
        print_result(result, args.verbose)


def initialize(args):
    """初始化 LLM 和 Orchestrator"""
    from Agent.llm_providers import OpenAIProvider, AnthropicProvider, DeepSeekProvider
    from Agent.orchestrator import ReactXenOrchestrator
    from Memory.tts import TinyTrajectoryStore
    
    # 选择 LLM Provider
    if args.provider == "openai":
        model = args.model or "gpt-4"
        llm = OpenAIProvider(api_key=args.api_key, model=model)
    elif args.provider == "anthropic":
        model = args.model or "claude-3-sonnet-20240229"
        llm = AnthropicProvider(api_key=args.api_key, model=model)
    elif args.provider == "deepseek":
        model = args.model or "deepseek-chat"
        llm = DeepSeekProvider(api_key=args.api_key, model=model)
    else:
        raise ValueError(f"Unknown provider: {args.provider}")
    
    print(f"[Init] Using {args.provider} ({model})")
    
    # 创建 TTS (可以加载预定义示例)
    tts = TinyTrajectoryStore()
    # 可选: tts.load_from_directory("./examples/")
    
    # 创建工具
    tools = create_default_tools()
    
    # 创建 Orchestrator
    orchestrator = ReactXenOrchestrator(
        llm_provider=llm,
        tts=tts,
        tools=tools,
        max_reflect_iterations=args.max_iterations,
    )
    
    return llm, orchestrator


def print_result(result: dict, verbose: bool = False):
    """打印结果"""
    print("\n" + "=" * 60)
    print("结果")
    print("=" * 60)
    
    if result["success"]:
        print(f"✅ 成功 (迭代次数: {result['iterations']})")
    else:
        print(f"❌ 失败 (迭代次数: {result['iterations']})")
    
    print(f"\n答案: {result['answer']}")
    
    if verbose:
        print("\n--- 执行轨迹 ---")
        for step_name, step_result in result["trace"]:
            print(f"  [{step_name}] {step_result.status.value}")


if __name__ == "__main__":
    main()
