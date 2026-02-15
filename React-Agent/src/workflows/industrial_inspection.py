from typing import Any, Dict

from src.agent.react_agent import IndustrialReActAgent
from src.config import load_settings


def run_industrial_inspection(task: str, asset_context: Dict[str, Any]) -> Dict[str, Any]:
    settings = load_settings()
    agent = IndustrialReActAgent(settings)
    return agent.run(task=task, asset_context=asset_context)
