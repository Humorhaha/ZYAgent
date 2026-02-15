import json

from src.workflows.industrial_inspection import run_industrial_inspection


def main() -> None:
    task = "对压缩机A-17进行异常诊断，并给出风险等级和处置建议。"
    asset_context = {
        "asset_id": "COMP-A17",
        "asset_type": "air_compressor",
        "line": "Line-3",
        "recent_event": "温度和振动连续2小时超阈值",
        "shift": "night",
    }

    result = run_industrial_inspection(task=task, asset_context=asset_context)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
