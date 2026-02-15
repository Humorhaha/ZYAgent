from typing import Any, Dict, List, Literal
from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    tool_name: str
    tool_input: Dict[str, Any] = Field(default_factory=dict)


class ReActStep(BaseModel):
    thought: str
    action: ToolCall | None = None
    observation: Dict[str, Any] | None = None


class InspectionReport(BaseModel):
    summary: str
    risk_level: Literal["low", "medium", "high", "critical"]
    suspected_root_causes: List[str]
    evidence: List[str]
    recommended_actions: List[str]
    escalation_needed: bool
    missing_data: List[str]
