"""
Case Success Checker - 语义校验模块

对每个 Case 的最终输出结果进行语义校验，而非过程校验。
用于 IoT/Data Query 场景的"任务成功"判定。
"""

import re
from dataclasses import dataclass
from typing import Any, Literal, Optional, List


@dataclass
class CaseResult:
    """Case 验证结果"""
    status: Literal["SUCCESS", "FAIL", "NOT_FOUND"]
    reason: str
    evidence: Optional[dict] = None


class CaseSuccessChecker:
    """
    Case 成功校验器
    
    对 Agent 最终输出进行语义校验，判断是否真正完成任务目标。
    """
    
    def __init__(self):
        # IoT 相关关键词
        self.iot_keywords = ["chiller", "facility", "asset", "equipment", "meter", "sensor"]
        
    def check(self, query: str, agent_output: Any, category: str = "") -> CaseResult:
        """
        校验 Agent 输出是否满足任务目标
        
        Args:
            query: 原始查询
            agent_output: Agent 最终输出（结构化或自然语言）
            category: 任务类型（IoT, FMSA, Workorder 等）
            
        Returns:
            CaseResult 包含 status, reason, evidence
        """
        # 根据类别选择校验策略
        if category.lower() in ["iot", "data query", "iot/data query"]:
            return self._check_iot_query(query, agent_output)
        else:
            # 默认宽松校验
            return self._check_generic(query, agent_output)
    
    def _check_iot_query(self, query: str, agent_output: Any) -> CaseResult:
        """
        IoT/Data Query 专用校验
        
        成功条件（满足其一）：
        1. 输出包含 asset_name + facility
        2. 明确声明 NOT_FOUND 并列出已检查数据源
        
        失败条件：
        - 仅包含 schema/preview/sample 信息
        - 未出现任何资产级或设施级实体
        """
        output_str = self._normalize_output(agent_output)
        
        # 提取查询中的目标实体
        target_asset = self._extract_asset_from_query(query)
        target_facility = self._extract_facility_from_query(query)
        
        # 检查是否为 NOT_FOUND 声明
        if self._is_not_found_declaration(output_str):
            # 验证是否列出了检查的数据源
            checked_sources = self._extract_checked_sources(output_str)
            if checked_sources:
                return CaseResult(
                    status="NOT_FOUND",
                    reason=f"Agent explicitly declared data not found after checking {len(checked_sources)} sources",
                    evidence={"checked_sources": checked_sources, "target_asset": target_asset}
                )
            else:
                return CaseResult(
                    status="FAIL",
                    reason="NOT_FOUND declared but no evidence of data sources checked",
                    evidence=None
                )
        
        # 检查是否包含目标资产信息
        has_asset = self._contains_asset(output_str, target_asset)
        has_facility = self._contains_facility(output_str, target_facility)
        
        # 检查是否仅为 schema 探索
        if self._is_schema_only(output_str):
            return CaseResult(
                status="FAIL",
                reason="Output contains only schema/preview information without entity data",
                evidence={"output_preview": output_str[:200]}
            )
        
        # 判断成功
        if has_asset and has_facility:
            return CaseResult(
                status="SUCCESS",
                reason=f"Output contains target asset '{target_asset}' and facility '{target_facility}'",
                evidence={"asset": target_asset, "facility": target_facility}
            )
        elif has_asset:
            return CaseResult(
                status="SUCCESS",
                reason=f"Output contains target asset '{target_asset}' (facility not explicitly required)",
                evidence={"asset": target_asset}
            )
        else:
            return CaseResult(
                status="FAIL",
                reason=f"Output does not contain target asset '{target_asset}'",
                evidence={"output_preview": output_str[:300]}
            )
    
    def _check_generic(self, query: str, agent_output: Any) -> CaseResult:
        """通用校验（宽松模式）"""
        output_str = self._normalize_output(agent_output)
        
        # 检查是否有实质性内容
        if len(output_str) < 50:
            return CaseResult(
                status="FAIL",
                reason="Output too short to contain meaningful result",
                evidence=None
            )
        
        # 检查是否仅为错误信息
        if self._is_error_only(output_str):
            return CaseResult(
                status="FAIL",
                reason="Output contains only error messages",
                evidence={"output": output_str[:200]}
            )
        
        return CaseResult(
            status="SUCCESS",
            reason="Generic check passed",
            evidence=None
        )
    
    def _normalize_output(self, output: Any) -> str:
        """将输出标准化为字符串"""
        if isinstance(output, str):
            return output.lower()
        elif isinstance(output, dict):
            return str(output).lower()
        elif isinstance(output, list):
            return str(output).lower()
        else:
            return str(output).lower() if output else ""
    
    def _extract_asset_from_query(self, query: str) -> str:
        """从查询中提取目标资产名称"""
        # 匹配 "Chiller X" 模式
        match = re.search(r'(chiller\s*\d+)', query, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        # 匹配其他资产模式
        match = re.search(r'(pump\s*\d+|boiler\s*\d+|ahu\s*\d+)', query, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        return ""
    
    def _extract_facility_from_query(self, query: str) -> str:
        """从查询中提取设施名称"""
        # 匹配 "at the XXX facility" 模式
        match = re.search(r'(?:at\s+(?:the\s+)?)?(\w+)\s+facility', query, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        # 匹配 "@ XXX" 模式
        match = re.search(r'@\s*(\w+)', query, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        return ""
    
    def _contains_asset(self, output: str, asset: str) -> bool:
        """检查输出是否包含目标资产"""
        if not asset:
            return True  # 如果没有指定资产，视为通过
        
        # 标准化资产名称进行匹配
        asset_lower = asset.lower().replace(" ", "")
        output_normalized = output.replace(" ", "")
        
        return asset_lower in output_normalized
    
    def _contains_facility(self, output: str, facility: str) -> bool:
        """检查输出是否包含目标设施"""
        if not facility:
            return True  # 如果没有指定设施，视为通过
        
        return facility.lower() in output
    
    def _is_not_found_declaration(self, output: str) -> bool:
        """检查是否为 NOT_FOUND 声明"""
        not_found_patterns = [
            r"not\s+found",
            r"no\s+data",
            r"does\s+not\s+exist",
            r"无法找到",
            r"不存在",
            r"未找到",
        ]
        for pattern in not_found_patterns:
            if re.search(pattern, output, re.IGNORECASE):
                return True
        return False
    
    def _extract_checked_sources(self, output: str) -> List[str]:
        """提取已检查的数据源"""
        sources = []
        # 匹配文件名模式
        file_patterns = [
            r'checked?\s+(?:file|table|dataset)s?:\s*([a-z_,\s]+)',
            r'searched?\s+in\s+([a-z_,\s]+)',
            r'查询了?\s*([a-z_、，\s]+)',
        ]
        for pattern in file_patterns:
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                sources.extend([s.strip() for s in re.split(r'[,、，\s]+', match.group(1)) if s.strip()])
        
        return sources
    
    def _is_schema_only(self, output: str) -> bool:
        """检查是否仅包含 schema 探索信息"""
        schema_only_patterns = [
            r'^.*schema.*:.*$',
            r'^.*columns?.*:.*$',
            r'^.*preview.*:.*$',
            r'^.*sample.*rows?.*$',
        ]
        
        # 如果输出包含实际数据值（非 schema 描述），则不是 schema-only
        has_data_values = bool(re.search(r'"[a-z_]+"\s*:\s*"[^"]+"|\'[a-z_]+\'\s*:\s*\'[^\']+\'', output))
        
        if has_data_values:
            return False
        
        # 检查是否所有内容都是 schema 描述
        for pattern in schema_only_patterns:
            if re.search(pattern, output, re.MULTILINE | re.IGNORECASE):
                return True
        
        return False
    
    def _is_error_only(self, output: str) -> bool:
        """检查是否仅包含错误信息"""
        error_patterns = [
            r'^error:',
            r'^exception:',
            r'traceback',
            r'failed to',
        ]
        for pattern in error_patterns:
            if re.search(pattern, output, re.IGNORECASE):
                return True
        return False
