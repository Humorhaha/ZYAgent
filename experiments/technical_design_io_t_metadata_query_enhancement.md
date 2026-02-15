# 技术设计文档（Technical Design Doc）

## 项目名称
AssetOpsBench Dual-System Agent  
IoT 元数据查询能力增强（Chiller / Facility 场景）

---

## 1. 设计背景

在当前 Dual-System Pipeline（System 1: ReAct Agent + System 2: MCTS Agent）中，Agent 已能够在真实数据集上完成端到端执行流程。但在 **IoT/Data Query** 场景（例如：

> Download the metadata for Chiller 3 at the MAIN facility

）中，系统存在“**逻辑完成但语义失败**”的问题：

- Agent 仅完成了工具探索（`preview_schema / sample_rows`）
- 未返回任何可验证的 **资产（Chiller 3）+ 设施（MAIN）元数据**
- 仍被框架判定为 *Success*

该问题在真实评测中会显著高估 Agent 能力，削弱 benchmark 的区分度。

---

## 2. 设计目标

### 2.1 核心目标

> **将“任务成功”从“搜索过程完成”升级为“语义目标达成”。**

即：
- 对 IoT/Data Query 类任务，必须返回与 Query 约束一致的结构化结果
- 否则明确失败或返回“不可达成功（Not Found）”

### 2.2 约束条件

- 不重构 Dual-System 架构
- 不引入外部索引系统（如 Elastic / Vector DB）
- 尽量复用现有工具与数据表

---

## 3. 系统整体架构（更新后）

```
User Query
   │
   ▼
System 1: ReAct Agent
   │  (工具规划 + 快速命中)
   ▼
[Optional Escalation]
   │
   ▼
System 2: MCTS Agent
   │  (搜索 + 去重 + reward-guided)
   ▼
Case Success Checker  ← 新增（强约束）
   │
   ├─ Success (with metadata)
   └─ Failure / Not Found (explainable)
```

---

## 4. 核心模块设计

### 4.1 Case Success Checker（新增）

#### 4.1.1 职责

对每个 Case 的 **最终输出结果** 进行语义校验，而非过程校验。

#### 4.1.2 接口设计

```python
class CaseSuccessChecker:
    def check(self, query: str, agent_output: Any) -> CaseResult:
        ...
```

#### 4.1.3 IoT/Data Query 判定规则

**输入：**
- 原始 Query（字符串）
- Agent 最终输出（结构化或自然语言）

**成功条件（满足其一）：**
1. 输出为结构化对象，且包含：
   - asset_name == "Chiller 3"（或等价 asset_id）
   - facility == "MAIN"
2. 输出明确声明：
   - 数据集中不存在该资产/设施组合
   - 并列出已检查的数据源

**失败条件：**
- 仅包含 schema / preview / sample 信息
- 未出现任何资产级或设施级实体

#### 4.1.4 返回结构

```python
@dataclass
class CaseResult:
    status: Literal["SUCCESS", "FAIL", "NOT_FOUND"]
    reason: str
    evidence: dict | None
```

---

### 4.2 IoT 资产元数据统一入口 Tool（推荐新增）

#### 4.2.1 设计动机

当前 Agent 需要通过多步 schema 探索自行“拼装”元数据，搜索成本高、错误率高。

#### 4.2.2 Tool 接口

```python
get_asset_metadata(asset_name: str, facility: str) -> dict
```

#### 4.2.3 内部实现逻辑（示意）

```python
# pseudo-code
components = query(component, where={"equipment": asset_name})
collections = join(collection_component_mapping, components)
events = query(event / meter / anomaly_events, where={"asset_name": asset_name})

return {
    "asset_name": asset_name,
    "facility": facility,
    "components": components,
    "collections": collections,
    "available_tables": [...],
}
```

#### 4.2.4 设计原则

- Tool 内复杂，Tool 外简单
- 对 Agent 暴露“一步命中”的能力

---

### 4.3 Tag 与数据发现能力增强

#### 4.3.1 新增工具

```python
list_tags() -> list[str]
get_file_tags(file_id: str) -> list[str]
```

#### 4.3.2 作用

- 消除 `search_by_tag` 的盲猜行为
- 降低无效搜索路径

---

### 4.4 MCTS 搜索策略改进

#### 4.4.1 工具调用去重

- 维护 `visited = {(tool_name, normalized_args)}`
- 重复调用直接降权或禁止展开

#### 4.4.2 Reward 设计（简化版）

| 信号 | Reward |
|----|----|
| Tool 返回包含目标 asset + facility | +1.0 |
| 返回无关资产（如 Chiller 6） | -0.5 |
| 重复工具调用 | -0.2 |
| Cache hit 后再次展开 | -0.1 |

#### 4.4.3 无关数据源黑名单

- 一旦确认 file 仅包含非目标资产
- 在当前 Case 内禁止再次探索

---

### 4.5 LLM API 异常恢复

#### 4.5.1 触发条件

- incomplete chunked read
- timeout / connection reset

#### 4.5.2 策略

1. 自动重试（≤ N 次）
2. 切换备用模型 / decoding
3. 进入 deterministic fallback（启发式路径）

---

## 5. 执行流程（IoT Case 示例）

```
Query: Chiller 3 @ MAIN
  ↓
System 1: get_asset_metadata
  ↓ (if fail)
System 2: MCTS search
  ↓
CaseSuccessChecker
  ↓
SUCCESS / NOT_FOUND / FAIL
```

---

## 6. 验收标准（Acceptance Criteria）

- 不允许“仅 schema 探索”判定为成功
- Case 7（Chiller 3 @ MAIN）：
  - 若数据存在 → 返回结构化元数据
  - 若不存在 → 明确 NOT_FOUND + 证据

---

## 7. 风险与权衡

| 风险 | 处理方式 |
|----|----|
| 数据集中缺少 facility 字段 | 允许 NOT_FOUND |
| Tool 复杂度上升 | 限定 IoT 专用 tool subset |
| MCTS 计算成本 | 去重 + 早停 |

---

## 8. 后续演进方向

- 各任务类型定义专属 Success Schema
- Schema-aware planning
- Trajectory replay + 评分

---

**文档状态**：Draft v1
