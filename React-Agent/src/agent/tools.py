from typing import Any, Dict, List


class IndustrialTools:
    """Tool layer stub for industrial inspection. Replace with real data connectors."""

    def get_sensor_snapshot(self, asset_id: str, tags: List[str]) -> Dict[str, Any]:
        mock = {
            "asset_id": asset_id,
            "readings": {
                "bearing_temp_c": 98.7,
                "vibration_mm_s": 9.8,
                "motor_current_a": 43.2,
                "discharge_pressure_kpa": 1270,
            },
            "tags_requested": tags,
            "timestamp": "2026-02-06T10:15:00Z",
        }
        return mock

    def get_alarm_history(self, asset_id: str, hours: int) -> Dict[str, Any]:
        return {
            "asset_id": asset_id,
            "hours": hours,
            "alarms": [
                {"code": "ALM-HT-002", "severity": "high", "count": 4},
                {"code": "ALM-VIB-007", "severity": "high", "count": 6},
            ],
        }

    def get_maintenance_history(self, asset_id: str, days: int) -> Dict[str, Any]:
        return {
            "asset_id": asset_id,
            "days": days,
            "events": [
                {"date": "2026-01-28", "type": "bearing_replacement", "note": "drive-end"},
                {"date": "2026-01-30", "type": "alignment_check", "note": "within tolerance"},
            ],
        }

    def get_failure_knowledge(self, symptom: str) -> Dict[str, Any]:
        kb = {
            "high_vibration_and_temperature": {
                "likely_causes": [
                    "bearing lubrication failure",
                    "misalignment after replacement",
                    "shaft imbalance",
                ],
                "recommended_checks": [
                    "verify lubrication route and grease type",
                    "perform laser alignment",
                    "check resonance band around operating speed",
                ],
            }
        }
        return kb.get(symptom, {"likely_causes": [], "recommended_checks": []})


TOOL_REGISTRY = {
    "get_sensor_snapshot": IndustrialTools.get_sensor_snapshot,
    "get_alarm_history": IndustrialTools.get_alarm_history,
    "get_maintenance_history": IndustrialTools.get_maintenance_history,
    "get_failure_knowledge": IndustrialTools.get_failure_knowledge,
}
