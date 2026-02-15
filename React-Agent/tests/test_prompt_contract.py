from src.agent.prompts import SYSTEM_PROMPT


def test_system_prompt_contains_output_contract() -> None:
    assert "summary" in SYSTEM_PROMPT
    assert "risk_level" in SYSTEM_PROMPT
    assert "recommended_actions" in SYSTEM_PROMPT
