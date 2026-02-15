"""
DataToolkit - 数据工具集 (基于第一性原理重构)

按需求文档设计，提供:
1. 工具发现 (list_tools)
2. 数据发现与理解 (list_files, describe_file, preview_schema, count_rows, get_time_range)
3. 受控数据访问 (sample_rows, query_data)
4. 数据分析 (analyze_data)
5. 元数据管理 (tag_file, search_by_tag)

设计原则:
- 装饰器模式统一注册工具
- 所有工具方法在单一类中
- 严格限制返回数据量 (max 200 rows)
- 结构化返回值 (dict/list)
"""

from __future__ import annotations
import functools
import json
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field
import logging
import time

logger = logging.getLogger(__name__)


# =============================================================================
# 工具装饰器
# =============================================================================

def tool(description: str = ""):
    """
    工具装饰器 - 标记方法为可调用工具
    
    Usage:
        @tool("返回系统中已注册的数据文件列表")
        def list_files(self, tag: Optional[List[str]] = None) -> List[str]:
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        
        # 附加元信息
        wrapper._is_tool = True
        wrapper._tool_name = func.__name__
        wrapper._tool_description = description or func.__doc__ or ""
        return wrapper
    
    return decorator


# =============================================================================
# 文件注册表
# =============================================================================

@dataclass
class FileEntry:
    """数据文件注册项"""
    file_id: str          # 逻辑名 (e.g., "event", "failure_codes")
    path: Path            # 物理路径
    tags: List[str] = field(default_factory=list)
    format: str = "csv"   # csv, json, jsonl


class FileRegistry:
    """数据文件注册表"""
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self._files: Dict[str, FileEntry] = {}
        self._auto_register()
    
    def _auto_register(self):
        """自动注册目录下的数据文件"""
        if not self.data_dir.exists():
            return
        
        for f in self.data_dir.iterdir():
            if f.is_file() and f.suffix in ('.csv', '.json', '.jsonl'):
                # 从文件名生成 file_id (去掉扩展名)
                file_id = f.stem.lower().replace('-', '_').replace(' ', '_')
                self._files[file_id] = FileEntry(
                    file_id=file_id,
                    path=f,
                    format=f.suffix[1:]  # 去掉点号
                )
    
    def get(self, file_id: str) -> Optional[FileEntry]:
        return self._files.get(file_id)
    
    def list_all(self) -> List[str]:
        return list(self._files.keys())
    
    def list_by_tags(self, tags: List[str]) -> List[str]:
        result = []
        for fid, entry in self._files.items():
            if any(t in entry.tags for t in tags):
                result.append(fid)
        return result
    
    def add_tags(self, file_id: str, tags: List[str]) -> bool:
        entry = self.get(file_id)
        if entry:
            entry.tags.extend([t for t in tags if t not in entry.tags])
            return True
        return False


# =============================================================================
# DataToolkit 主类
# =============================================================================

class DataToolkit:
    """
    数据工具集
    
    提供统一的数据发现、访问、分析接口。
    所有工具通过 @tool 装饰器注册。
    
    Usage:
        toolkit = DataToolkit("/path/to/data")
        files = toolkit.list_files()
        schema = toolkit.describe_file("event")
    """
    
    MAX_ROWS = 200  # 最大返回行数
    
    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        self.registry = FileRegistry(self.data_dir)
    
    # =========================================================================
    # 1. 工具发现
    # =========================================================================
    
    @tool("返回当前系统允许调用的工具名称列表")
    def list_tools(self) -> List[str]:
        """列出所有可用工具"""
        tools = []
        for name in dir(self):
            attr = getattr(self, name)
            if callable(attr) and getattr(attr, '_is_tool', False):
                tools.append(attr._tool_name)
        return tools
    
    # =========================================================================
    # 2. 数据发现与理解
    # =========================================================================
    
    @tool("列出系统中已注册的数据文件（逻辑名）")
    def list_files(self, tag: Optional[List[str]] = None) -> List[str]:
        """列出数据文件"""
        if tag:
            return self.registry.list_by_tags(tag)
        return self.registry.list_all()
    
    @tool("返回数据文件的整体结构信息")
    def describe_file(self, file_id: str) -> Dict[str, Any]:
        """描述文件结构"""
        entry = self.registry.get(file_id)
        if not entry:
            return {"error": f"File '{file_id}' not found"}
        
        df = self._load_dataframe(entry, nrows=100)
        if df is None:
            return {"error": f"Failed to load file '{file_id}'"}
        
        # 识别时间列
        time_columns = []
        for col in df.columns:
            if 'time' in col.lower() or 'date' in col.lower():
                time_columns.append(col)
            elif df[col].dtype == 'datetime64[ns]':
                time_columns.append(col)
        
        # 行数数量级
        full_df = self._load_dataframe(entry)
        row_count = len(full_df) if full_df is not None else 0
        if row_count >= 1000000:
            row_count_level = "1M+"
        elif row_count >= 100000:
            row_count_level = "100K+"
        elif row_count >= 10000:
            row_count_level = "10K+"
        elif row_count >= 1000:
            row_count_level = "1K+"
        else:
            row_count_level = f"{row_count}"
        
        return {
            "file_id": file_id,
            "format": entry.format,
            "columns": list(df.columns),
            "types": {col: str(df[col].dtype) for col in df.columns},
            "row_count_level": row_count_level,
            "time_columns": time_columns,
            "tags": entry.tags
        }
    
    @tool("快速查看字段及其语义说明（不读数据）")
    def preview_schema(self, file_id: str) -> List[Dict[str, str]]:
        """预览 Schema"""
        entry = self.registry.get(file_id)
        if not entry:
            return [{"error": f"File '{file_id}' not found"}]
        
        df = self._load_dataframe(entry, nrows=5)
        if df is None:
            return [{"error": f"Failed to load file"}]
        
        schema = []
        for col in df.columns:
            col_type = str(df[col].dtype)
            # 简单启发式描述
            sample = df[col].dropna().iloc[0] if len(df[col].dropna()) > 0 else None
            desc = self._infer_column_description(col, col_type, sample)
            schema.append({
                "column": col,
                "type": col_type,
                "description": desc
            })
        
        return schema
    
    @tool("返回数据行数（或数量级）")
    def count_rows(self, file_id: str) -> Dict[str, Any]:
        """统计行数"""
        entry = self.registry.get(file_id)
        if not entry:
            return {"error": f"File '{file_id}' not found"}
        
        df = self._load_dataframe(entry)
        if df is None:
            return {"error": "Failed to load file"}
        
        return {"row_count": len(df)}
    
    @tool("获取数据的时间覆盖范围")
    def get_time_range(
        self,
        file_id: str,
        time_column: Optional[str] = None
    ) -> Dict[str, Any]:
        """获取时间范围"""
        entry = self.registry.get(file_id)
        if not entry:
            return {"error": f"File '{file_id}' not found"}
        
        df = self._load_dataframe(entry)
        if df is None:
            return {"error": "Failed to load file"}
        
        # 自动检测时间列
        if not time_column:
            for col in df.columns:
                if 'time' in col.lower() or 'date' in col.lower():
                    time_column = col
                    break
        
        if not time_column or time_column not in df.columns:
            return {"error": "No time column found or specified"}
        
        # 转换为时间类型
        try:
            times = pd.to_datetime(df[time_column], errors='coerce')
            times = times.dropna()
            if len(times) == 0:
                return {"error": "No valid time values"}
            
            return {
                "time_column": time_column,
                "start_time": str(times.min()),
                "end_time": str(times.max())
            }
        except Exception as e:
            return {"error": str(e)}
    
    # =========================================================================
    # 3. 受控数据访问
    # =========================================================================
    
    @tool("安全抽样少量数据行，用于理解字段语义")
    def sample_rows(
        self,
        file_id: str,
        columns: Optional[List[str]] = None,
        n: int = 10
    ) -> Dict[str, Any]:
        """抽样数据"""
        n = min(n, self.MAX_ROWS)
        
        entry = self.registry.get(file_id)
        if not entry:
            return {"error": f"File '{file_id}' not found"}
        
        df = self._load_dataframe(entry)
        if df is None:
            return {"error": "Failed to load file"}
        
        # 选择列
        if columns:
            valid_cols = [c for c in columns if c in df.columns]
            if not valid_cols:
                return {"error": f"No valid columns specified"}
            df = df[valid_cols]
        
        # 抽样
        if len(df) > n:
            df = df.sample(n=n, random_state=42)
        
        return {"rows": df.to_dict(orient='records')}
    
    @tool("受限条件查询，只读、小结果集")
    def query_data(
        self,
        file_id: str,
        columns: Optional[List[str]] = None,
        where: Optional[Dict[str, Any]] = None,
        limit: int = 50
    ) -> Dict[str, Any]:
        """条件查询"""
        limit = min(limit, self.MAX_ROWS)
        
        entry = self.registry.get(file_id)
        if not entry:
            return {"error": f"File '{file_id}' not found"}
        
        df = self._load_dataframe(entry)
        if df is None:
            return {"error": "Failed to load file"}
        
        # 应用 where 条件
        if where:
            for col, val in where.items():
                if col in df.columns:
                    if isinstance(val, dict):
                        # 支持 {"gt": 10, "lt": 100}
                        if "gt" in val:
                            df = df[df[col] > val["gt"]]
                        if "lt" in val:
                            df = df[df[col] < val["lt"]]
                        if "eq" in val:
                            df = df[df[col] == val["eq"]]
                        if "contains" in val:
                            df = df[df[col].astype(str).str.contains(val["contains"], na=False)]
                    else:
                        df = df[df[col] == val]
        
        # 选择列
        if columns:
            valid_cols = [c for c in columns if c in df.columns]
            if valid_cols:
                df = df[valid_cols]
        
        truncated = len(df) > limit
        df = df.head(limit)
        
        return {
            "rows": df.to_dict(orient='records'),
            "truncated": truncated
        }
    
    # =========================================================================
    # 4. 数据分析 (核心)
    # =========================================================================
    
    @tool("在 Tool 内执行统计与分析（不返回行级数据）")
    def analyze_data(
        self,
        file_id: str,
        operation: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        数据分析
        
        Supported operations:
        - describe: 描述性统计
        - value_counts: 值计数
        - groupby_agg: 分组聚合
        - timeseries_agg: 时序聚合
        - correlation: 相关性分析
        """
        params = params or {}
        
        entry = self.registry.get(file_id)
        if not entry:
            return {"error": f"File '{file_id}' not found"}
        
        df = self._load_dataframe(entry)
        if df is None:
            return {"error": "Failed to load file"}
        
        try:
            if operation == "describe":
                return self._op_describe(df, params)
            elif operation == "value_counts":
                return self._op_value_counts(df, params)
            elif operation == "groupby_agg":
                return self._op_groupby_agg(df, params)
            elif operation == "timeseries_agg":
                return self._op_timeseries_agg(df, params)
            elif operation == "correlation":
                return self._op_correlation(df, params)
            else:
                return {"error": f"Unknown operation '{operation}'"}
        except Exception as e:
            return {"error": str(e)}
    
    def _op_describe(self, df: pd.DataFrame, params: Dict) -> Dict:
        """描述性统计"""
        column = params.get("column")
        if column and column in df.columns:
            stats = df[column].describe().to_dict()
        else:
            stats = df.describe().to_dict()
        return {"result": stats}
    
    def _op_value_counts(self, df: pd.DataFrame, params: Dict) -> Dict:
        """值计数"""
        column = params.get("column")
        if not column or column not in df.columns:
            return {"error": "Column required for value_counts"}
        
        top_n = params.get("top_n", 20)
        counts = df[column].value_counts().head(top_n).to_dict()
        return {"result": counts}
    
    def _op_groupby_agg(self, df: pd.DataFrame, params: Dict) -> Dict:
        """分组聚合"""
        group_by = params.get("group_by")
        agg_column = params.get("agg_column")
        agg_func = params.get("agg_func", "count")
        
        if not group_by or group_by not in df.columns:
            return {"error": "group_by column required"}
        
        if agg_column and agg_column in df.columns:
            result = df.groupby(group_by)[agg_column].agg(agg_func).to_dict()
        else:
            result = df.groupby(group_by).size().to_dict()
        
        return {"result": result}
    
    def _op_timeseries_agg(self, df: pd.DataFrame, params: Dict) -> Dict:
        """时序聚合"""
        time_column = params.get("time_column")
        value_column = params.get("value_column")
        freq = params.get("freq", "1D")
        agg_func = params.get("agg_func", "mean")
        
        if not time_column or time_column not in df.columns:
            return {"error": "time_column required"}
        if not value_column or value_column not in df.columns:
            return {"error": "value_column required"}
        
        df = df.copy()
        df[time_column] = pd.to_datetime(df[time_column], errors='coerce')
        df = df.dropna(subset=[time_column])
        df = df.set_index(time_column)
        
        resampled = df[value_column].resample(freq).agg(agg_func)
        result = {str(k): v for k, v in resampled.to_dict().items()}
        
        return {"result": result}
    
    def _op_correlation(self, df: pd.DataFrame, params: Dict) -> Dict:
        """相关性分析"""
        columns = params.get("columns")
        
        if columns:
            valid_cols = [c for c in columns if c in df.columns]
            if len(valid_cols) < 2:
                return {"error": "At least 2 valid numeric columns required"}
            numeric_df = df[valid_cols].select_dtypes(include='number')
        else:
            numeric_df = df.select_dtypes(include='number')
        
        if numeric_df.shape[1] < 2:
            return {"error": "At least 2 numeric columns required"}
        
        corr = numeric_df.corr().to_dict()
        return {"result": corr}
    
    # =========================================================================
    # 5. 元数据管理
    # =========================================================================
    
    @tool("为数据文件写入标签（仅 metadata）")
    def tag_file(self, file_id: str, tags: List[str]) -> Dict[str, bool]:
        """添加标签"""
        success = self.registry.add_tags(file_id, tags)
        return {"ok": success}
    
    @tool("按标签查找数据文件")
    def search_by_tag(self, tags: List[str]) -> List[str]:
        """按标签搜索"""
        return self.registry.list_by_tags(tags)
    
    # =========================================================================
    # 6. IoT 资产元数据 (新增)
    # =========================================================================
    
    @tool("一步获取 IoT 资产元数据（设备+设施）")
    def get_asset_metadata(
        self,
        asset_name: str,
        facility: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        IoT 资产元数据统一查询
        
        Args:
            asset_name: 资产名称（如 "Chiller 3"）
            facility: 设施名称（如 "MAIN"，可选）
            
        Returns:
            包含 asset_name, facility, components, collections, available_tables 的结构化数据
        """
        result = {
            "asset_name": asset_name,
            "facility": facility,
            "components": [],
            "collections": [],
            "available_tables": [],
            "related_data": {}
        }
        
        # 1. 查询 component 表
        component_entry = self.registry.get("component")
        if component_entry:
            df = self._load_dataframe(component_entry)
            if df is not None:
                result["available_tables"].append("component")
                
                # 查找匹配的资产
                name_lower = asset_name.lower()
                for col in ["equipment", "equipment_name", "asset_name", "name"]:
                    if col in df.columns:
                        matches = df[df[col].astype(str).str.lower().str.contains(name_lower, na=False)]
                        if len(matches) > 0:
                            result["components"] = matches.head(10).to_dict('records')
                            break
        
        # 2. 查询 collection_component_mapping 表
        mapping_entry = self.registry.get("collection_component_mapping")
        if mapping_entry:
            df = self._load_dataframe(mapping_entry)
            if df is not None:
                result["available_tables"].append("collection_component_mapping")
                
                # 基于 component 查找关联的 collections
                if result["components"]:
                    # 尝试匹配 component 列
                    for comp in result["components"][:3]:
                        comp_name = str(comp.get("component", ""))
                        if comp_name:
                            for col in ["component", "component_name"]:
                                if col in df.columns:
                                    matches = df[df[col].astype(str).str.contains(comp_name, na=False, case=False)]
                                    if len(matches) > 0:
                                        for _, row in matches.head(5).iterrows():
                                            coll = row.get("collection", row.get("collection_name", ""))
                                            if coll and coll not in result["collections"]:
                                                result["collections"].append(coll)
                                        break
        
        # 3. 查询相关事件数据
        for file_id in ["event", "alert_events", "anomaly_events", "all_wo_with_code_component_events"]:
            entry = self.registry.get(file_id)
            if entry:
                df = self._load_dataframe(entry, nrows=500)
                if df is not None:
                    result["available_tables"].append(file_id)
                    
                    # 检查是否包含目标资产数据
                    name_lower = asset_name.lower()
                    for col in ["equipment_name", "asset_name", "equipment", "equipment_id"]:
                        if col in df.columns:
                            matches = df[df[col].astype(str).str.lower().str.contains(name_lower, na=False)]
                            if len(matches) > 0:
                                result["related_data"][file_id] = {
                                    "count": len(matches),
                                    "sample": matches.head(3).to_dict('records')
                                }
                                break
        
        # 4. 查询 meter 数据
        meter_entry = self.registry.get("meter")
        if meter_entry:
            df = self._load_dataframe(meter_entry, nrows=200)
            if df is not None:
                result["available_tables"].append("meter")
        
        # 判断是否找到有效数据
        if not result["components"] and not result["related_data"]:
            result["status"] = "NOT_FOUND"
            result["message"] = f"No data found for asset '{asset_name}' in checked tables: {result['available_tables']}"
        else:
            result["status"] = "SUCCESS"
        
        return result
    
    @tool("列出系统中所有可用的标签")
    def list_tags(self) -> List[str]:
        """列出所有标签"""
        all_tags = set()
        for file_id in self.registry.list_all():
            entry = self.registry.get(file_id)
            if entry and entry.tags:
                all_tags.update(entry.tags)
        return list(all_tags)
    
    @tool("获取指定文件的标签")
    def get_file_tags(self, file_id: str) -> Dict[str, Any]:
        """获取文件标签"""
        entry = self.registry.get(file_id)
        if not entry:
            return {"error": f"File '{file_id}' not found"}
        return {"file_id": file_id, "tags": entry.tags}
    
    # =========================================================================
    # 7. FMSA 故障模式查询 (新增)
    # =========================================================================
    
    @tool("获取指定资产的所有故障模式")
    def get_failure_modes(
        self,
        asset_name: str
    ) -> Dict[str, Any]:
        """
        获取资产的故障模式列表
        
        Args:
            asset_name: 资产名称（如 "Chiller 6"）
            
        Returns:
            包含 asset_name, equipment_type, failure_modes 的结构化数据
        """
        result = {
            "asset_name": asset_name,
            "equipment_type": "",
            "failure_modes": [],
            "status": "NOT_FOUND"
        }
        
        # 1. 从资产名称提取设备类型 (e.g., "Chiller 6" -> "Chiller")
        # 常见模式: "Device N" 或 "Device-N"
        import re
        match = re.match(r'^([A-Za-z]+)[\s\-_]*\d*$', asset_name.strip())
        if match:
            equipment_type = match.group(1).strip()
        else:
            # 尝试取第一个单词
            equipment_type = asset_name.split()[0] if asset_name else ""
        
        result["equipment_type"] = equipment_type
        
        # 2. 查询 failure_mode 表
        failure_mode_entry = self.registry.get("failure_mode")
        if failure_mode_entry:
            df = self._load_dataframe(failure_mode_entry)
            if df is not None and "Equipment" in df.columns:
                # 大小写不敏感匹配
                matches = df[df["Equipment"].astype(str).str.lower() == equipment_type.lower()]
                if len(matches) > 0:
                    result["failure_modes"] = matches.to_dict('records')
                    result["status"] = "SUCCESS"
        
        # 3. 如果找不到，尝试模糊匹配
        if result["status"] == "NOT_FOUND" and failure_mode_entry:
            df = self._load_dataframe(failure_mode_entry)
            if df is not None and "Equipment" in df.columns:
                matches = df[df["Equipment"].astype(str).str.lower().str.contains(equipment_type.lower(), na=False)]
                if len(matches) > 0:
                    result["failure_modes"] = matches.to_dict('records')
                    result["status"] = "SUCCESS"
                    result["match_type"] = "fuzzy"
        
        # 4. 附加来自 primary_failure_codes 的信息
        pfc_entry = self.registry.get("primary_failure_codes")
        if pfc_entry:
            df = self._load_dataframe(pfc_entry)
            if df is not None:
                result["available_failure_codes"] = df.head(20).to_dict('records')
        
        return result

    # =========================================================================
    # 内部辅助方法
    # =========================================================================
    
    def _load_dataframe(
        self,
        entry: FileEntry,
        nrows: Optional[int] = None
    ) -> Optional[pd.DataFrame]:
        """加载数据为 DataFrame"""
        try:
            if entry.format == "csv":
                return pd.read_csv(entry.path, nrows=nrows)
            elif entry.format == "json":
                return pd.read_json(entry.path)
            elif entry.format == "jsonl":
                return pd.read_json(entry.path, lines=True, nrows=nrows)
            else:
                return None
        except Exception:
            return None
    
    def _infer_column_description(
        self,
        col_name: str,
        col_type: str,
        sample: Any
    ) -> str:
        """推断列描述"""
        name_lower = col_name.lower()
        
        if 'id' in name_lower:
            return "标识符/主键"
        elif 'time' in name_lower or 'date' in name_lower:
            return "时间戳/日期"
        elif 'code' in name_lower:
            return "编码/代码"
        elif 'name' in name_lower:
            return "名称"
        elif 'status' in name_lower or 'state' in name_lower:
            return "状态值"
        elif 'count' in name_lower or 'num' in name_lower:
            return "计数/数量"
        elif 'value' in name_lower or 'amount' in name_lower:
            return "数值/金额"
        elif col_type.startswith('float') or col_type.startswith('int'):
            return "数值型字段"
        else:
            return "文本/分类字段"
    
    # =========================================================================
    # 统一调用接口
    # =========================================================================
    
    def call(self, tool_name: str, **kwargs) -> Any:
        """
        统一工具调用接口
        
        Args:
            tool_name: 工具名称
            **kwargs: 工具参数
            
        Returns:
            工具返回值
        """
        method = getattr(self, tool_name, None)
        if method and getattr(method, '_is_tool', False):
            return method(**kwargs)
        else:
            return {"error": f"Tool '{tool_name}' not found"}
    
    def get_tool_info(self) -> List[Dict[str, str]]:
        """获取所有工具的描述信息"""
        infos = []
        for name in dir(self):
            attr = getattr(self, name)
            if callable(attr) and getattr(attr, '_is_tool', False):
                infos.append({
                    "name": attr._tool_name,
                    "description": attr._tool_description
                })
        return infos


# =============================================================================
# 便捷函数
# =============================================================================

def create_toolkit(data_dir: str | Path = None) -> DataToolkit:
    """创建默认 Toolkit 实例"""
    if data_dir is None:
        data_dir = Path(__file__).parent / "sample_data"
    return DataToolkit(data_dir)


# =============================================================================
# 测试入口
# =============================================================================

if __name__ == "__main__":
    # 测试
    data_dir = Path(__file__).parent / "sample_data"
    toolkit = DataToolkit(data_dir)
    
    print("=" * 60)
    print("DataToolkit Test")
    print("=" * 60)
    
    print("\n[1] list_tools:")
    print(toolkit.list_tools())
    
    print("\n[2] list_files:")
    print(toolkit.list_files())
    
    print("\n[3] describe_file('failure_codes'):")
    print(json.dumps(toolkit.describe_file("failure_codes"), indent=2, ensure_ascii=False))
    
    print("\n[4] preview_schema('event'):")
    print(json.dumps(toolkit.preview_schema("event"), indent=2, ensure_ascii=False))
    
    print("\n[5] count_rows('alert_events'):")
    print(toolkit.count_rows("alert_events"))
    
    print("\n[6] sample_rows('failure_codes', n=3):")
    print(json.dumps(toolkit.sample_rows("failure_codes", n=3), indent=2, ensure_ascii=False))
    
    print("\n[7] analyze_data('failure_codes', 'describe'):")
    print(json.dumps(toolkit.analyze_data("failure_codes", "describe"), indent=2, ensure_ascii=False))
