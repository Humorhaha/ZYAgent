# Industrial ReAct Agent Architecture

## 1. Layered Design

- Interface Layer
  - `src/main.py`
  - 接收任务与设备上下文，触发工业检测流程。

- Workflow Layer
  - `src/workflows/industrial_inspection.py`
  - 编排一次完整诊断请求。

- Agent Core Layer
  - `src/agent/react_agent.py`
  - 执行 ReAct 推理循环，维护 `messages` 上下文，控制最大步数。

- Prompt Layer
  - `src/agent/prompts.py`
  - 提供工业检测专用系统提示词与用户模板。
  - 强制输出结构化 JSON，避免自由文本歧义。

- Tool Layer
  - `src/agent/tools.py`
  - 统一封装工业数据查询接口（传感器、告警、维护、知识库）。

- Schema Layer
  - `src/agent/schemas.py`
  - 定义工具调用和最终报告字段，确保上下游契约一致。

- Config Layer
  - `src/config.py`
  - 从 `.env` 读取 API、模型、推理参数。

## 2. ReAct Execution Flow

1. 注入工业检测系统提示词（安全优先、证据优先、可执行优先）。
2. 模型输出中间步骤 JSON（`thought + action`）或最终结果 JSON（`final_answer`）。
3. Agent 根据 `action` 调用单个工具，获取 observation。
4. 将 observation 回灌给模型，继续下一轮推理。
5. 输出 `InspectionReport` 格式结果。

## 3. Industrial Risk Control

- 当数据出现高危组合（高温 + 高振动 + 高频告警）时，建议默认升级。
- 对每个根因必须给出对应证据，避免“无证据结论”。
- 若关键数据缺失（如频谱、润滑检测），必须写入 `missing_data`。

## 4. Prompt Contract

- 中间步骤：
  - `thought`
  - `action.tool_name`
  - `action.tool_input`

- 终局输出：
  - `summary`
  - `risk_level`
  - `suspected_root_causes`
  - `evidence`
  - `recommended_actions`
  - `escalation_needed`
  - `missing_data`

## 5. Extension Points

- Tool Layer 接入真实 OT/IT 数据源（MQTT, OPC-UA, Historian, CMMS）。
- 增加规则引擎，对 `risk_level` 与 `escalation_needed` 做二次校验。
- 增加离线评测集，验证根因诊断准确率和误报率。
