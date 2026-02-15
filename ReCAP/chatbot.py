from dotenv import load_dotenv
from pathlib import Path
import os
from typing_extensions import TypedDict
from typing import Annotated

import requests
from bs4 import BeautifulSoup

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()

from langchain_openai import ChatOpenAI
env_path = Path(__file__).parent/'.env'
if env_path.exists():
    load_dotenv(env_path)


class State(TypedDict):
    messages: Annotated[list, add_messages]


if __name__ == '__main__':
    graph = StateGraph(State)
    llm = ChatOpenAI(
        model = os.getenv("QWEN_MODEL"),
        api_key = os.getenv("QWEN_API_KEY"),
        base_url=os.getenv("QWEN_URL"),
        temperature=0.2,
    )

    def chat(state: State):
        result = llm_with_tools.invoke(state['messages'])
        return {
            'messages': [result]
        }
    

    graph.add_node("chat-node", chat)
    # graph.add_edge(START, "chat-node")
    # graph.add_edge("chat-node", END)

    # app = graph.compile()
    # result = app.invoke({"messages": [{"role": "user", "content": "你好, 你是什么模型?"}]})

    # print(result.get("messages")[-1].content)

    # pic = app.get_graph().draw_mermaid_png(output_file_path="chatbot.png")


    @tool
    def baidu_search(query: str):
        """搜索百度并返回结果"""
        url = "https://www.baidu.com/s"
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/121.0.0.0 Safari/537.36"
            )
        }

        try:
            response = requests.get(
                url,
                params={"wd": query},
                headers=headers,
                timeout=30,
            )
            response.encoding = 'utf-8'

            soup = BeautifulSoup(response.text, 'html.parser')

            results = []
            # 查找前 3 条结果（避免内容过多撑爆 LLM 上下文）
            for item in soup.select('#content_left .result')[:3]:
                # 获取标题
                title_tag = item.select_one('h3')
                if not title_tag:
                    continue
                title = title_tag.get_text().strip()
                
                # 获取链接
                anchor = title_tag.select_one('a')
                link = anchor.get('href') if anchor else "无链接"
                
                # 获取摘要 (通常在 class="c-abstract" 或类似结构中)
                abstract_tag = item.select_one('.c-abstract') or item.select_one('.content-right_8Zs40')
                abstract = abstract_tag.get_text().strip() if abstract_tag else "无摘要"
                
                results.append(f"标题: {title}\n摘要: {abstract}\n链接: {link}\n")
                
            if not results:
                return f"在百度上没有找到关于 '{query}' 的相关结果。"
                
            return "---\n".join(results)
        
        except Exception as e:
            return f"搜索时发生错误: {str(e)}"




    tools = [baidu_search]
    llm_with_tools = llm.bind_tools(tools)
    tool_node = ToolNode(tools)
    graph.add_node("tool-node", tool_node)


    graph.add_edge(START, "chat-node")
    graph.add_conditional_edges(
        "chat-node",
        tools_condition,
        {
            "tools":"tool-node",
            "__end__": END
        }
    )

    graph.add_edge("tool-node", "chat-node")

    app = graph.compile(checkpointer=memory, interrupt_before=["tool-node"])
    config = {
        'configurable': {'thread_id': 'cli-session-1'}
    }

    print("开始对话，输入 exit / quit 结束。")
    while True:
        query = input("你: ").strip()
        if query.lower() in {"exit", "quit"}:
            print("对话结束。")
            break
        if not query:
            continue

        events = app.stream(
            {'messages': [{'role': 'user', 'content': query}]},
            config,
            stream_mode='values'
        )

        for event in events:
            event.get('messages')[-1].pretty_print()

        # 如果下一步是工具节点，说明命中了 interrupt_before，在这里人工确认
        snapshot = app.get_state(config)
        while snapshot.next == ("tool-node",):
            last_message = snapshot.values.get("messages", [])[-1]
            tool_calls = getattr(last_message, "tool_calls", []) or []
            if tool_calls:
                print("\n检测到待执行工具调用:")
                for call in tool_calls:
                    print(f"- {call.get('name')} args={call.get('args')}")
            else:
                print("\n检测到工具中断，但未解析到 tool_calls。")

            approval = input("是否继续执行工具? [Y/n]: ").strip().lower()
            if approval in {"n", "no"}:
                print("已取消本次工具执行。")
                break

            # 继续执行工具节点
            events = app.stream(None, config, stream_mode='values')
            for event in events:
                event.get('messages')[-1].pretty_print()

            snapshot = app.get_state(config)

    pic = app.get_graph().draw_mermaid_png(output_file_path="chatbot.png")

    # snapshot = app.get_state(config)
    # print(snapshot)
