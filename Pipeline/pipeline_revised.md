# Pipeline Architecture (Revised)

> This document refines the original pipeline design, clarifies responsibilities, and adds **Neo4j graph storage** for MCTS paths (nodes + edges + summary/reflection).
> It also makes the “mock” areas explicit and proposes **batch persistence** to avoid slowing down MCTS.

---

## 1. System Overview

### 1.1 High-level diagram

```mermaid
graph TD
    subgraph "Pipeline Orchestrator"
        A[Task Input] --> B{Hot Start (once)}
        B -->|Wisdom L2| C[TTS Examples (Few-shot)]
        C --> D[System 1: ReActAgent]

        D -->|Success| E[Memory L1: Trajectory Samples]
        D -->|Failure| F[Failure Queue]

        F --> G{System 2 Enabled?}
        G -->|Yes| H[System 2 Search]
        G -->|No| I[Record Failure]

        H --> J{Search Result}
        J -->|Success| K[Promotion Check]
        J -->|Failure| I

        K -->|Meet Conditions| L[Generate / Update Wisdom L2]

        E --> M[Update TTS]
        M --> N[Compress TTS (Markov Window)]
    end

    subgraph "Persistence Layer"
        P1[JsonlBackend]:::store
        P2[Neo4jBackend]:::store
        P3[TTS Persistent Store]:::store
    end

    E --> P1
    E --> P2
    M --> P3
    H -->|MCTS path batch write| P2

    classDef store fill:#f6f6f6,stroke:#888,stroke-width:1px;
```

### 1.2 Key constraints

- **Hot Start is executed only once at task start.** System 1 is not allowed to actively retrieve mid-task.
- **System 2 (MCTS / Beam / None) triggers only on failure cases**, and should not block the hot path.
- **Graph persistence is not in the rollout hot loop**: write MCTS paths in batch.

---

## 2. Execution Flow

### Step 1: Hot Start (L2 Wisdom injection, once)

```python
wisdom_objs = memory.retrieve(task, k=wisdom_k)  # L2 retrieval
injected_wisdom = [w.text for w in wisdom_objs]
```

Constraint: only once at task start. No active retrieval during System 1.

---

### Step 2: TTS Examples (Few-shot)

```python
tts_examples = tts.retrieve(query=task, k=3)
context.examples = tts_examples
```

Purpose: improve prompt quality and reduce brittle reasoning.

---

### Step 3: System 1 Execution (ReAct)

```python
self.react_agent = ReActAgent(
    llm_provider=LLMAdapter(self.llm),
    max_steps=config.system1.max_steps,
)
result = self.react_agent.run(context)
```

---

### Step 4: Success / Failure branching

```python
failure = create_failure_case(trajectory, soft_threshold)

if not failure:
    # Success Path
    memory.put_samples([trajectory])  # L1 store
    tts.add(tts_trajectory)           # TTS update
else:
    # Failure Path
    failure_queue.enqueue(failure)
    # -> System 2
```

---

### Step 5: System 2 Search (MCTS / Beam / None)

```python
if strategy == "mcts":    s2 = MCTSStrategy()
elif strategy == "beam":  s2 = BeamSearchStrategy()
else:                     s2 = NoSearchStrategy()

trajectories = s2.search(failure_case, config)
```

**Important**: System 2 should return:
- (a) candidate trajectories (success or best-effort)
- (b) a **selected MCTS path** (root → leaf) and per-node `summary/reflection` if enabled
- (c) metadata for evaluation (rollouts, time, branching, etc.)

---

### Step 6: Promotion (L1 → L2 Wisdom)

```python
if memory.enable_promotion:
    new_wisdom = memory.promote(min_samples=5)
    # LLM generates concise Wisdom text (must be real, not mock)
```

---

## 3. Data Contracts (Recommended)

### 3.1 Trajectory object (L1 sample)
Minimum fields:
- `task_id`, `run_id`
- `steps[]`: structured (tool calls, observations, intermediate decisions)
- `final_answer`, `success`, `metrics`
- `created_at`, `model`, `prompt_hash`

### 3.2 FailureCase object
Minimum fields:
- `failure_type`, `evidence`, `trajectory_snapshot`
- `task_context`, `constraints`
- `created_at`

### 3.3 MCTS Path (Graph write payload)
Recommended payload:
- `treeId`, `taskId`, `runId`, `createdAt`
- `nodes[]`: each node carries `{nodeId, stateHash, depth, isTerminal, N,W,Q,P, summary, reflection, modelMeta...}`
- `edges[]`: each edge carries `{parentId, childId, actionId, actionText, prior, visitsOnEdge}`

---

## 4. Neo4j Graph Storage (MCTS Tree + Notes)

### 4.1 What Neo4j stores (NOT “classes”)
Neo4j stores:
- **Node instances** (entities): e.g. an MCTS node in a specific search tree
- **Relationships** (edges): e.g. parent → child transitions
- **Properties** (key-values): summary/reflection, N/W/Q, priors, etc.
- **Labels**: e.g. `:MCTSNode`, `:SearchTree`

### 4.2 Suggested graph model (MVP)

Nodes:
- `(:SearchTree {treeId, taskId, runId, createdAt, strategy})`
- `(:MCTSNode {treeId, nodeId, stateHash, depth, isTerminal, N, W, Q, P, summary, reflection, updatedAt})`

Relationships:
- `(t:SearchTree)-[:HAS_ROOT]->(root:MCTSNode)`
- `(parent:MCTSNode)-[:CHILD {treeId, actionId, actionText, prior, visitsOnEdge, updatedAt}]->(child:MCTSNode)`

### 4.3 Constraints & indexes (recommended)

```cypher
CREATE CONSTRAINT mcts_node_unique IF NOT EXISTS
FOR (n:MCTSNode)
REQUIRE (n.treeId, n.nodeId) IS UNIQUE;

CREATE INDEX mcts_statehash IF NOT EXISTS
FOR (n:MCTSNode) ON (n.treeId, n.stateHash);

CREATE FULLTEXT INDEX mcts_text IF NOT EXISTS
FOR (n:MCTSNode) ON EACH [n.summary, n.reflection];
```

### 4.4 Batch write strategy (critical for performance)

**Do not write per rollout step synchronously**.

Recommended:
- Accumulate N rollouts worth of `{nodes, edges}` in memory
- Every `batch_size` (e.g. 50–200 rollouts) OR every `flush_interval_ms` (e.g. 500–1500ms), do one write transaction with `UNWIND`
- Optionally only persist:
  - tree structure and final selected path, plus a snapshot of stats at the end; or
  - structure + periodic stat snapshots

---

## 5. Storage Backends & Consistency

### 5.1 Backends
- JsonlBackend: simplest and fastest for append-only logs
- Neo4jBackend: best for relationship queries, path replay, and semantic search over notes

### 5.2 Required: backend equivalence tests
Implement a test harness to ensure:
- `retrieve()` returns comparable results for the same input
- `promote()` produces equivalent candidate wisdom given the same L1 samples
- ordering differences are accounted for (tie-breakers must be deterministic)

---

## 6. Known Gaps / To-Do

### 🔴 Must implement (blocking)
1) **MCTSAdapter LLM calls are mock** → connect to real LLM for action expansion / next-step generation.
2) **Promotion LLM summarization is mock** → connect `llm.generate_wisdom()` or equivalent.
3) **JsonlBackend vs Neo4jBackend equivalence test** → required before switching storage in production.

### 🟡 Recommended optimizations
- Make `_agent_result_to_trajectory` parse agent internal states for rich steps.
- Persist TTS (avoid losing few-shot memory after restart).
- Build `evaluation/` to plot learning curves: success rate, search cost, wisdom hit rate.

---

## 7. Configuration Mapping

```yaml
system1:
  max_steps: 10
  use_wisdom: true
  wisdom_k: 3
  markov_window: 3

system2:
  enabled: true
  search_strategy: mcts  # mcts | beam | none
  graph_persist:
    enabled: true
    backend: neo4j
    batch_size: 100
    flush_interval_ms: 1000
    persist_stats: snapshot  # off | snapshot | periodic

memory:
  enabled: true
  backend: jsonl         # jsonl | neo4j
  enable_hot_start: true
  enable_promotion: true

tts:
  persist: true
  backend: file  # file | sqlite | object_store
```

---

## Appendix A. Cypher Demo: Write one MCTS path

```cypher
MERGE (t:SearchTree {treeId:$treeId})
ON CREATE SET t.taskId=$taskId, t.runId=$runId, t.createdAt=timestamp(), t.strategy=$strategy;

WITH t
UNWIND $nodes AS nd
MERGE (n:MCTSNode {treeId:$treeId, nodeId:nd.nodeId})
SET n.stateHash=nd.stateHash,
    n.depth=nd.depth,
    n.isTerminal=nd.isTerminal,
    n.N=nd.N, n.W=nd.W, n.Q=nd.Q, n.P=nd.P,
    n.summary=nd.summary,
    n.reflection=nd.reflection,
    n.updatedAt=timestamp();

WITH t
UNWIND $edges AS e
MATCH (p:MCTSNode {treeId:$treeId, nodeId:e.parentId})
MATCH (c:MCTSNode {treeId:$treeId, nodeId:e.childId})
MERGE (p)-[r:CHILD {treeId:$treeId, actionId:e.actionId, childId:e.childId}]->(c)
SET r.actionText=e.actionText,
    r.prior=e.prior,
    r.visitsOnEdge=e.visitsOnEdge,
    r.updatedAt=timestamp();
```

---

## Appendix B. Cypher Demo: Query best terminal path

```cypher
MATCH (t:SearchTree {treeId:$treeId})-[:HAS_ROOT]->(root)
MATCH p=(root)-[:CHILD*0..50]->(leaf:MCTSNode)
WHERE leaf.isTerminal = true
RETURN p
ORDER BY leaf.Q DESC
LIMIT 1;
```
