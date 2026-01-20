# LLM-MCTS Augmented Framework: SOTA Formulas & Engineering Review

在 LLM Agent 领域，将 MCTS (Monte Carlo Tree Search) 应用于推理任务已成为提升复杂问题解决能力的关键路径。目前的 SOTA 主要是 **LATS (Language Agent Tree Search)** 和 **RAP (Reasoning via Planning)**。

本文档基于 **SOTA 论文** 与 **工程落地** 的双重视角，拆解 MCTS 四个核心阶段的 **公式变体 (Formula Variants)**。

---

## Part 1: SOTA 公式变体 (Formula Variations)

### 1. 选择阶段 (Selection): 平衡的艺术

核心是 **UCT (Upper Confidence Bound for Trees)** 及其变体，用于平衡 "Exploitation" (利用高分路径) 和 "Exploration" (探索未知路径)。

#### **Variant A: Standard PUCT (AlphaZero / LATS Baseline)**
最通用的形式，LATS 直接沿用了此公式。

$$
a_t = \arg\max_{a} \left[ Q(s, a) + c_{puct} \cdot P(s, a) \cdot \frac{\sqrt{\sum_b N(s, b)}}{1 + N(s, a)} \right]
$$

*   **$Q(s,a)$**: 动作价值 (Value)，通常由 LLM 打分或 Rollout 下一阶段的胜率决定。
*   **$P(s,a)$**: 动作先验概率 (Prior)，即 LLM 生成该动作的 Logprobs (Normalized)。
*   **$N(s,a)$**: 访问次数。
*   **$c_{puct}$**: 探索常数 (通常取 1.0 ~ 4.0)。

#### **Variant B: Uncertainty-Aware UCT (Engineering Optimization)**
针对 LLM 生成的不确定性进行改进，SOTA 论文 (如 *TS-LLM*) 建议引入熵 (Entropy) 惩罚或奖励。

$$
a_t = \arg\max_{a} \left[ Q(s, a) + \lambda \cdot \mathcal{H}(P(\cdot|s)) + c_{puct} \frac{\sqrt{N(s)}}{1 + N(s, a)} \right]
$$

*   **$\mathcal{H}$**: 策略的熵。当 LLM 对当前状态很困惑 (高熵) 时，$\lambda$ 项强迫 Agent 增加探索力度。

---

### 2. 扩展阶段 (Expansion): 动作空间的生成

LLM 的动作空间是无限的词表，因此 Expand 策略决定了树的宽度。

#### **Variant A: Constrained Expansion (RAP / ToT)**
限制最大子节点数 $k$，直接采样 $k$ 个最可能的动作。

$$
\mathcal{A}_{expand} = \text{Top-k}_{a \in \mathcal{V}} P_{LLM}(a|s)
$$

*   **优点**: 简单直接，Token 消耗可控。
*   **缺点**: 容易漏掉 "低概率但高价值" 的创新解法 (The "Eureka" moment)。

#### **Variant B: Progressive Widening (AlphaGo / Stochastic Beam Search)**
随着父节点访问次数 $N(s)$ 的增加，动态解锁新的子节点。

$$
|\text{children}(s)| \leq C \cdot N(s)^\alpha
$$

*   **工程意义**: 初始只生成 1 个子节点 (Greedy)。只有当父节点被反复访问 (说明当前子节点不够好或值得深挖) 时，才花费 Token 生成第 2、第 3 个子节点。这是**降本增效**的核心。

---

### 3. 评估阶段 (Evaluation): 奖励的设计

这是 MCTS for LLM 最困难的部分。

#### **Variant A: Self-Correction Scoring (LATS SOTA)**
LATS 不仅打分，还让 LLM 生成 "反思"。

$$
r = \text{LLM}_{eval}(s, \text{rubric}) \in [0, 1]
$$
$$
Q(s, a) \leftarrow \frac{1}{n} \sum r_i + w \cdot \text{HasFeedback}(s)
$$

*   **特点**: **Dense Reward**。不仅看结果，还看中间步骤的 "Self-Refine" 质量。即使任务失败，如果 Agent 产生了有价值的错误反思，也给予微小奖励。

#### **Variant B: Process Reward Model (PRM / Math-Shepherd)**
专门训练一个轻量级 Verifier ($\theta$) 来打分，而不是用昂贵的 LLM。

$$
V(s) = \prod_{i=1}^{t} P_{\theta}(\text{step}_i \text{ is correct} | s_{0...i-1})
$$

*   **SOTA 趋势**: OpenAI 的 *Let's Verify Step by Step* 证明了 PRM (过程奖励) 远优于 ORM (结果奖励)。
*   **公式**: 价值 $V(s)$ 是路径上所有步骤正确率的乘积 (或对数和)。

---

### 4. 反向传播 (Backpropagation): 梯度的传递

#### **Variant A: Standard Mean Update (Baseline)**
$$
Q(s, a) \leftarrow Q(s, a) + \frac{r - Q(s, a)}{N(s, a)}
$$
*   仅传递数值，信息丢失严重。

#### **Variant B: Reflexion Feedback (LATS / ExpeL)**
不仅回传 $r$，还回传文本反馈 $m$。

$$
\mathcal{M}(s) \leftarrow \mathcal{M}(s) \cup \{ \text{feedback from failed leaf} \}
$$

*   **机制**: 当叶子节点失败时，它的 "Error Trace" 会被加入到父节点的 Prompt Context 中。
*   **效果**: 下次父节点 Expand 时，LLM 会 "看到" 兄弟节点的死因，从而避免重蹈覆辙。

---

## Part 2: 创新方案全景对比 (Configuration Menu)

针对工程落地的不同约束（成本 vs 精度），我们提供多种**创新方案 (Option A/B)** 的对比选择：

| 维度 | **创新方案 A (轻量/高效率)** | **创新方案 B (高性能/高质量)** | **选型建议 (Trade-offs)** |
| :--- | :--- | :--- | :--- |
| **选择机制**<br>(Selection) | **基于熵的动态探索 (Entropy UCT)**<br>公式引入 $\lambda \mathcal{H}$。<br>**原理**: 越困惑，越探索。 | **语义去重 (Semantic UCT)**<br>引入 $-\text{Sim}(e_i, e_j)$ 惩罚项。<br>**原理**: 拒绝“换汤不换药”的重复路径。 | **通用推理**选 A (无需额外模型)；<br>**代码/创作**选 B (重复生成率高)。 |
| **扩展策略**<br>(Expansion) | **渐进式宽度 (Progressive Widening)**<br>$k \propto N(s)^\alpha$。<br>**特点**: 按需扩容，好路径一条就够。 | **置信度感知扩展 (Confidence-aware)**<br>仅当 Top-1 Score < Threshold 时触发。<br>**特点**: 类似于“不满意才重做”。 | **预算有限**选 B (Token 最省)；<br>**算力充足**选 A (搜索更完备)。 |
| **评估方式**<br>(Evaluation) | **过程奖励模型 (PRM)**<br>$V = \prod P_{ver}(step)$。<br>**特点**: 速度快，线性复杂度 $O(N)$。 | **成对比较验证 (Pairwise Verifier)**<br>$\text{Elo}(A, B)$。<br>**特点**: 准确度极高，但复杂度为 $O(N \log N)$。 | **长链推理**选 A (否则太慢)；<br>**关键决策**选 B (精度优先)。 |
| **反馈机制**<br>(Backprop) | **错误摘要回传 (Feedback Summary)**<br>回传 $E_{msg}$。<br>**特点**: 占用 Context Window 小。 | **完整轨迹反思 (Full Trace Reflexion)**<br>回传 Full Trace。<br>**特点**: 信息量大，但 Context 消耗巨大。 | **复杂项目**选 A (长文本限制)；<br>**短逻辑题**选 B。 |

### 架构设计建议

#### 1. "Economical Scout" (经济型侦察兵)
*   **目标**: 极致性价比，适合在线服务。
*   **配置**: `Entropy UCT` + `Confidence-aware Expansion` + `PRM` + `Feedback Summary`

#### 2. "Deep Thinker" (深度思考者)
*   **目标**: 追求 SOTA 解决率，适合离线任务。
*   **配置**: `Semantic UCT` + `Progressive Widening` + `Pairwise Verifier` + `Full Trace Reflexion`