# Industrial ReAct Prompt Contract

## Intermediate Step JSON

```json
{
  "thought": "接下来最有信息增益的检查是什么",
  "action": {
    "tool_name": "get_sensor_snapshot",
    "tool_input": {
      "asset_id": "COMP-A17",
      "tags": ["bearing_temp_c", "vibration_mm_s"]
    }
  }
}
```

## Final Answer JSON

```json
{
  "final_answer": {
    "summary": "驱动端轴承温升+振动共振，怀疑润滑失效并伴随轻微不对中。",
    "risk_level": "high",
    "suspected_root_causes": [
      "bearing lubrication failure",
      "misalignment after replacement"
    ],
    "evidence": [
      "bearing_temp_c=98.7C 超过报警阈值",
      "vibration_mm_s=9.8 高于设备基线",
      "高等级振动告警与高温告警并发"
    ],
    "recommended_actions": [
      "立即降载并安排点检",
      "2小时内完成润滑状态确认",
      "执行激光对中复核"
    ],
    "escalation_needed": true,
    "missing_data": [
      "润滑脂取样结果",
      "1x/2x频谱分量"
    ]
  }
}
```
