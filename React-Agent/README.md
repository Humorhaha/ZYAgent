# Industrial ReAct Agent

This folder contains a full ReAct Agent architecture designed for industrial inspection workflows.

## 1. Architecture

- `src/config.py`: env loading and runtime config.
- `src/agent/schemas.py`: structured data contracts.
- `src/agent/prompts.py`: industrial inspection prompt design.
- `src/agent/tools.py`: tool layer for sensor/history/knowledge lookup.
- `src/agent/react_agent.py`: ReAct reasoning loop implementation.
- `src/workflows/industrial_inspection.py`: task-oriented workflow entry.
- `src/main.py`: CLI demo entrypoint.
- `tests/test_prompt_contract.py`: prompt contract sanity checks.

## 2. Setup

```bash
cd React-Agent
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# fill OPENAI_API_KEY
python -m src.main
```

## 3. Industrial Prompt Design Goals

- Safety-first decisions.
- Evidence-based conclusions.
- Explicit uncertainty and escalation rules.
- Actionable field instructions.

## 4. Output Contract

The final answer is always JSON with:

- `summary`: diagnosis summary.
- `risk_level`: low/medium/high/critical.
- `suspected_root_causes`: ranked list.
- `evidence`: observations and tool findings.
- `recommended_actions`: immediate and follow-up actions.
- `escalation_needed`: boolean.
- `missing_data`: what must be collected next.

