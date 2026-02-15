# OpsBench 数据分析工具设计

## 1. 数据访问层 (Data Access Layer)
该层工具负责与文件系统的基本交互和数据检索。

### 工具 1: `list_opsbench_files`
- **功能**: 列出 OpsBench 环境中所有可用的数据文件，并显示其大小和最后修改日期。
- **参数**:
    - `directory_path` (字符串, 默认="/Users/richw/OpsBench"): 目标目录路径。
    - `filter_extension` (字符串, 可选): 按文件扩展名过滤 (例如 ".csv")。

### 工具 2: `get_dataset_schema`
- **功能**: 返回特定数据集的结构，包括列名和数据类型。
- **参数**:
    - `filename` (字符串): 要查看的文件名。

### 工具 3: `preview_dataset`
- **功能**: 返回数据集的前 N 行，以便快速预览内容。
- **参数**:
    - `filename` (字符串): 文件名。
    - `n_rows` (整数, 默认=5): 返回的行数。

### 工具 4: `search_dataset_value`
- **功能**: 在数据集的所有列中搜索特定的值或关键词。
- **参数**:
    - `filename` (字符串): 文件名。
    - `query` (字符串): 要搜索的值。
    - `case_sensitive` (布尔值, 默认=False): 是否区分大小写。

## 2. 数据预处理层 (Data Processing Layer)
用于清洗数据并为分析做准备的工具。

### 工具 5: `clean_missing_data`
- **功能**: 处理数据集中的缺失值 (NaN/Null)。
- **参数**:
    - `filename` (字符串): 输入文件。
    - `strategy` (字符串): 处理策略，可选 "drop_rows" (删除行), "fill_zero" (填0), "fill_mean" (填均值), "fill_forward" (前向填充)。
    - `output_filename` (字符串, 可选): 保存清洗后数据的文件名。

### 工具 6: `filter_dataset_rows`
- **功能**: 根据列条件过滤数据集。
- **参数**:
    - `filename` (字符串): 输入文件。
    - `column` (字符串): 用于过滤的列名。
    - `operator` (字符串): 操作符，可选 ">", "<", "==", "!=", "contains"。
    - `value` (字符串/数字): 用于比较的值。

### 工具 7: `join_datasets`
- **功能**: 基于公共键 (例如 `equipment_id`) 合并两个数据集。
- **参数**:
    - `left_file` (字符串): 左表文件。
    - `right_file` (字符串): 右表文件。
    - `on_column` (字符串): 公共列名。
    - `how` (字符串, 默认="inner"): 连接类型 ("inner", "left", "right", "outer")。

## 3. 数据特征分析层 (Data Analysis Layer)
用于统计分析和特定 OpsBench 业务逻辑的工具。

### 工具 8: `calculate_column_stats`
- **功能**: 计算数值列的描述性统计信息 (均值, 中位数, 标准差, 最小值, 最大值) 或分类列的计数。
- **参数**:
    - `filename` (字符串): 输入文件。
    - `column_name` (字符串): 要分析的列名。

### 工具 9: `analyze_failure_distribution`
- **功能**: 针对 `failure_codes.csv` 的专用工具，分析主要故障代码与次要故障代码的频率分布。
- **参数**:
    - `filename` (字符串, 默认="failure_codes.csv")。

### 工具 10: `analyze_time_series_trend`
- **功能**: 分析时间序列数据 (如传感器数据) 以发现趋势或重采样数据 (如计算每小时平均值)。
- **参数**:
    - `filename` (字符串): 包含时间序列的文件 (例如 chiller 数据)。
    - `time_column` (字符串): 时间戳列名。
    - `value_column` (字符串): 要计算平均值/总和的列。
    - `frequency` (字符串): 重采样频率 (例如 "1H", "1D")。

### 工具 11: `map_alert_to_failure`
- **功能**: 利用映射文件，解析从告警事件到其潜在故障定义的完整链路。
- **参数**:
    - `alert_rule_id` (字符串): `alert_events.csv` 中的规则 ID。
- **依赖**: 内部调用 `join_datasets` 来连接 `alert_rule.csv` -> `mapping.csv` -> `failure_codes.csv`。

## 4. 高级脚本分析层 (Advanced Analysis Layer)
使用动态代码生成进行复杂、即席分析的工具。

### 工具 12: `generate_analysis_script`
- **功能**: 使用 LLM 为标准工具无法覆盖的复杂分析查询生成独立的 Python 脚本。
- **参数**:
    - `goal` (字符串): 分析目标，例如 "分析冷水机温度与功率输入之间的相关性"。
    - `input_files` (字符串列表): 涉及的文件。

### 工具 13: `execute_analysis_script`
- **功能**: 在安全沙箱中执行 Python 脚本并捕获其输出 (stdout, 图表)。
- **参数**:
    - `script_path` (字符串): .py 文件路径。

## 5. 用户交互与报告层 (User Interaction & Reporting Layer)
聚合结果并呈现给用户的高级工具。

### 工具 14: `explain_dataset_relationships`
- **功能**: 分析多个文件的列名，推断并描述不同数据集之间的关联 (例如 "表A 通过 `equipment_id` 与表B 关联")。
- **参数**:
    - `dataset_names` (字符串列表): 要交叉引用的文件列表。

### 工具 15: `generate_opsbench_report`
- **功能**: 生成一份综合的 Markdown 报告，总结数据集情况，包括结构、基本统计数据和已识别的关系。
- **参数**:
    - `target_files` (字符串列表): 报告中要包含的数据集。
- **依赖**: 调用 `get_dataset_schema`, `calculate_column_stats`, 和 `explain_dataset_relationships`。
