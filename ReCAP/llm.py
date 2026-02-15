import os
from pathlib import Path
import json
from dotenv import load_dotenv
from openai import OpenAI

env_path =Path(__file__).parent/'.env'
if os.path.exists(env_path):
    load_dotenv(env_path)


def chat(query:str):
    api_key = os.getenv("QWEN_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("QWEN_URL") or os.getenv("OPENAI_BASE_URL")
    model = os.getenv("QWEN_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-4o-mini"

    if not api_key:
        raise RuntimeError(
            "缺少 API Key。请在 ReCAP/.env 配置 QWEN_API_KEY（或 OPENAI_API_KEY）。"
        )

    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": query}],
        response_format={"type": "json_object"},
    )

    try:
        data = json.loads(response.choices[0].message.content)
        return data
    except json.JSONDecodeError as e:
        print(f"JSONDecodeError: {str(e)}")
        return response.choices[0].message.content
