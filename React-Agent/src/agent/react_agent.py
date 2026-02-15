import json
from typing import Any, Dict, List

from openai import OpenAI

from src.agent.prompts import REACT_USER_TEMPLATE, SYSTEM_PROMPT
from src.agent.tools import IndustrialTools
from src.config import Settings


class IndustrialReActAgent:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )
        self.tools = IndustrialTools()

    def _run_tool(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        if not hasattr(self.tools, name):
            return {"error": f"unknown tool: {name}"}
        fn = getattr(self.tools, name)
        try:
            return {"ok": True, "result": fn(**args)}
        except Exception as exc:  # pragma: no cover
            return {"ok": False, "error": str(exc)}

    def run(self, task: str, asset_context: Dict[str, Any]) -> Dict[str, Any]:
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": REACT_USER_TEMPLATE.format(
                    task=task,
                    asset_context=json.dumps(asset_context, ensure_ascii=False, indent=2),
                ),
            },
        ]

        for _ in range(self.settings.max_steps):
            completion = self.client.chat.completions.create(
                model=self.settings.openai_model,
                temperature=self.settings.temperature,
                messages=messages,
                response_format={"type": "json_object"},
            )
            content = completion.choices[0].message.content or "{}"
            payload = json.loads(content)

            if "final_answer" in payload:
                return payload["final_answer"]

            if "action" not in payload:
                return {
                    "summary": "Agent returned invalid intermediate step.",
                    "risk_level": "high",
                    "suspected_root_causes": ["invalid_reasoning_output"],
                    "evidence": [content],
                    "recommended_actions": ["Escalate to reliability engineer for manual review."],
                    "escalation_needed": True,
                    "missing_data": ["valid_action_or_final_answer"],
                }

            action = payload["action"]
            tool_name = action.get("tool_name", "")
            tool_input = action.get("tool_input", {})
            observation = self._run_tool(tool_name, tool_input)

            messages.append({"role": "assistant", "content": json.dumps(payload, ensure_ascii=False)})
            messages.append(
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "observation": observation,
                            "instruction": "继续ReAct。若信息充分，请输出 final_answer。",
                        },
                        ensure_ascii=False,
                    ),
                }
            )

        return {
            "summary": "Agent reached max reasoning steps before final conclusion.",
            "risk_level": "medium",
            "suspected_root_causes": ["incomplete_diagnosis"],
            "evidence": ["max_steps_exceeded"],
            "recommended_actions": ["Collect missing telemetry and rerun diagnosis."],
            "escalation_needed": False,
            "missing_data": ["more_diagnostic_signals"],
        }
