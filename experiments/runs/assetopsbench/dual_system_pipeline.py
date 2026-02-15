"""
Dual-System Pipeline - System 1 + System 2 完整评测流水线 (使用预设 Prompts)

流程:
1. 从 HuggingFace 加载评测样本
2. System 1 (ReAct Agent) 使用 MCTS_NEXT_STEP prompt
3. 如果 System 1 失败或低置信度 -> 升级到 System 2 (MCTS)
4. System 2 使用完整 MCTS prompts: MCTS_NEXT_STEP, MCTS_REFLECTION, MCTS_VALUE_EVALUATION
5. 记录结果并保存
"""

import sys
import json
import re
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

# 添加路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from experiments.runs.assetopsbench.eval_loader import sample_eval_cases, EvalCase
from experiments.runs.assetopsbench.toolkit import DataToolkit
from experiments.runs.assetopsbench.case_checker import CaseSuccessChecker, CaseResult

from core.trajectory import Trajectory, Step
from LLM.llm import create_llm, LLM
from LLM.prompts import (
    REACT_NEXT_STEP,
    MCTS_NEXT_STEP,
    MCTS_NEXT_STEP_WITH_REFLECTION,
    MCTS_REFLECTION,
    MCTS_VALUE_EVALUATION,
    MCTS_NEXT_STEPS_BATCH,
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logging.getLogger("httpx").setLevel(logging.WARNING)  # Silence HTTP requests
logger = logging.getLogger(__name__)


# =============================================================================
# 扩展 Prompts - 添加 DataToolkit 工具说明
# =============================================================================

TOOLKIT_TOOLS_DESCRIPTION = """
### 可用 DataToolkit 工具:
- **list_files(tag)**: 列出系统中已注册的数据文件
- **describe_file(file_id)**: 返回数据文件的整体结构信息 (columns, types, row_count)
- **preview_schema(file_id)**: 快速查看字段及其语义说明
- **count_rows(file_id)**: 返回数据行数
- **get_time_range(file_id, time_column)**: 获取数据的时间覆盖范围
- **sample_rows(file_id, columns, n)**: 安全抽样少量数据行 (n<=200)
- **query_data(file_id, columns, where, limit)**: 受限条件查询 (limit<=200)
- **analyze_data(file_id, operation, params)**: 执行统计分析
  - operation: describe, value_counts, groupby_agg, timeseries_agg, correlation
- **tag_file(file_id, tags)**: 为数据文件写入标签
- **search_by_tag(tags)**: 按标签查找数据文件
- **get_asset_metadata(asset_name, facility)**: 一步获取 IoT 资产元数据
- **list_tags()**: 列出系统中所有可用的标签
- **get_file_tags(file_id)**: 获取指定文件的标签
- **get_failure_modes(asset_name)**: 获取指定资产的所有故障模式（自动提取设备类型）
- **Finish(answer)**: 输出最终答案

### 参数格式示例:
- describe_file(file_id="event")
- sample_rows(file_id="failure_codes", n=5)
- analyze_data(file_id="event", operation="value_counts", params={"column": "event_type"})
- query_data(file_id="alert_events", where={"equipment_id": "CWC04009"}, limit=50)
"""



@dataclass
class PipelineResult:
    """流水线运行结果"""
    case_id: int
    query: str
    system_used: str = ""  # "system1", "system2", "both"
    success: bool = False
    final_answer: str = ""
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    trajectory_steps: int = 0
    duration_seconds: float = 0.0
    error: str = ""
    confidence: float = 0.0
    semantic_status: str = ""  # SUCCESS, FAIL, NOT_FOUND


class OpsBenchReActAgent:
    """
    System 1: ReAct Agent (使用 MCTS_NEXT_STEP prompt)
    """
    
    def __init__(
        self,
        llm: LLM,
        toolkit: DataToolkit,
        max_steps: int = 8,
        confidence_threshold: float = 0.6
    ):
        self.llm = llm
        self.toolkit = toolkit
        self.max_steps = max_steps
        self.confidence_threshold = confidence_threshold
        self.tool_calls = []
    
    def run(self, task: str, category: str = "") -> Tuple[Trajectory, float]:

        """
        执行 ReAct 循环 (使用预设 MCTS_NEXT_STEP prompt)
        """
        trajectory = Trajectory(task=task)
        self.tool_calls = []
        
        files_list = self.toolkit.list_files()
        
        history = ""
        for step_num in range(1, self.max_steps + 1):
            # 使用 ReAct Agent 专用 Prompt
            prompt = REACT_NEXT_STEP.format(
                task=f"{task}\n\n可用数据文件: {files_list[:10]}\n{TOOLKIT_TOOLS_DESCRIPTION}",
                history=history or "(无)"
            )
            
            response = self.llm.generate(prompt)
            
            # 解析 Thought 和 Action
            thought, action, action_args = self._parse_response(response)
            
            # 检查是否完成
            if action.lower() == "finish":
                trajectory.add_step(
                    thought=thought,
                    action="Finish",
                    observation=action_args
                )
                trajectory.mark_success()
                trajectory.metadata["final_answer"] = action_args
                confidence = self._evaluate_confidence(trajectory)
                return trajectory, confidence
            
            # 执行真实工具调用
            observation = self._execute_real_tool(action, action_args)
            
            # 记录步骤
            trajectory.add_step(
                thought=thought,
                action=f"{action}({action_args})",
                observation=observation
            )
            
            # 更新历史
            history += f"\nStep {step_num}: Action: {action}({action_args})\nObservation: {observation}...\n"
            
            print(f"\n[Step {step_num}]")
            print(f"  Thought: {thought}")
            print(f"  Action:  {action}({action_args})")
            logger.info(f"Step {step_num}: {action}({action_args}) -> {len(observation)} chars")
        
        # 超过最大步数
        trajectory.mark_failure()
        trajectory.metadata["failure_reason"] = "max_steps_exceeded"
        return trajectory, 0.3
    
    def _parse_response(self, response: str) -> Tuple[str, str, str]:
        """解析 LLM 响应，提取 Thought 和 Action"""
        thought = ""
        action = "Finish"
        action_args = ""
        
        # 提取 Thought (支持中英文)
        thought_match = re.search(r'(?:Thought|思路|下一步)[:：]\s*(.+?)(?=\n|Action|$)', response, re.DOTALL)
        if thought_match:
            thought = thought_match.group(1).strip()
        
        # 提取 Action: tool_name(args)
        action_match = re.search(r'Action[:：]\s*(\w+)\s*\(([^)]*)\)', response, re.DOTALL)
        if action_match:
            action = action_match.group(1)
            action_args = action_match.group(2).strip()
        elif "Finish" in response:
            action = "Finish"
            finish_match = re.search(r'Finish\s*[:\(]\s*(.+?)(?:\)|$)', response, re.DOTALL)
            if finish_match:
                action_args = finish_match.group(1).strip()
        
        return thought, action, action_args
    
    def _execute_real_tool(self, action: str, args_str: str) -> str:
        """执行真实工具调用"""
        available_tools = self.toolkit.list_tools()
        
        if action not in available_tools:
            return f"Error: Unknown tool '{action}'. Available: {available_tools}"
        
        try:
            kwargs = self._parse_tool_args(args_str)
            result = self.toolkit.call(action, **kwargs)
            
            self.tool_calls.append({
                "tool": action,
                "args": kwargs,
                "result_preview": str(result)
            })
            
            if isinstance(result, dict):
                return json.dumps(result, indent=2, ensure_ascii=False, default=str)

            elif isinstance(result, list):
                return json.dumps(result[:20], indent=2, ensure_ascii=False, default=str)
            else:
                return str(result)
        
        except Exception as e:
            return f"Error executing {action}: {e}"
    
    def _parse_tool_args(self, args_str: str) -> Dict[str, Any]:
        """解析工具参数字符串"""
        kwargs = {}
        
        if not args_str.strip():
            return kwargs
        
        # 解析关键字参数: file_id="event", operation="describe"
        kv_pattern = r'(\w+)\s*=\s*(?:"([^"]*)"|\'([^\']*)\'|(\{[^}]+\})|(\[[^\]]+\])|(\w+))'
        
        for match in re.finditer(kv_pattern, args_str):
            key = match.group(1)
            value = match.group(2) or match.group(3) or match.group(4) or match.group(5) or match.group(6)
            
            if value:
                if value.startswith('{') or value.startswith('['):
                    try:
                        value = json.loads(value.replace("'", '"'))
                    except:
                        pass
                elif value.isdigit():
                    value = int(value)
                
                kwargs[key] = value
        
        # 兜底: 单个位置参数
        if not kwargs and args_str:
            parts = args_str.split(',')
            if len(parts) == 1 and '=' not in parts[0]:
                kwargs['file_id'] = parts[0].strip().strip('"').strip("'")
        
        return kwargs
    
    def _evaluate_confidence(self, trajectory: Trajectory) -> float:
        """评估置信度"""
        tool_count = len(self.tool_calls)
        if tool_count >= 3:
            return 0.85
        elif tool_count >= 2:
            return 0.7
        elif tool_count >= 1:
            return 0.55
        return 0.3


class OpsBenchMCTSAdapter:
    """
    System 2: MCTS Adapter (使用完整 MCTS prompts)
    """
    
    def __init__(
        self,
        llm: LLM,
        toolkit: DataToolkit,
        iteration_limit: int = 5
    ):
        self.llm = llm
        self.toolkit = toolkit
        self.iteration_limit = iteration_limit
        self.tool_calls = []
    
    def search(self, task: str, category: str = "") -> Tuple[Trajectory, float]:
        """执行 MCTS 搜索 (使用预设 prompts)"""
        from MCTS.base import MCTS, MCTSTask
        
        mcts_task = self._create_mcts_task(task, category)
        
        mcts = MCTS(mcts_task)
        mcts.search()
        
        best_node, best_value = mcts.root.get_best_value()
        
        trajectory = Trajectory(task=task)
        if best_node and best_node.history_action:
            trajectory.add_step(
                thought="MCTS Search Result",
                action="MCTS",
                observation=best_node.history_action or "No solution found"
            )
            trajectory.mark_success()
            trajectory.metadata["final_answer"] = best_node.history_action
            self.tool_calls = mcts_task.tool_calls if hasattr(mcts_task, 'tool_calls') else []
            return trajectory, best_value
        
        trajectory.mark_failure()
        return trajectory, 0.3
    
    def _create_mcts_task(self, task: str, category: str):
        """创建 MCTS Task (使用预设 prompts)"""
        from MCTS.base import MCTSTask
        
        toolkit = self.toolkit
        llm = self.llm
        tool_calls_list = []
        files_list = toolkit.list_files()
        
        # 增强任务描述
        enhanced_task = f"{task}\n\n可用数据文件: {files_list[:10]}\n{TOOLKIT_TOOLS_DESCRIPTION}"
        
        class OpsBenchTask(MCTSTask):
            def __init__(self):
                super().__init__(
                    iteration_limit=5,
                    branch=2,
                    use_reflection='common',
                    roll_branch=1,        # Reduce branching
                    roll_forward_steps=0  # Disable rollout simulation
                )
                self.tool_calls = tool_calls_list
                self._tool_cache = {}  # Cache: (tool_name, args_key) -> result
            
            def _get_cache_key(self, tool_name: str, kwargs: Dict) -> str:
                """Generate cache key from tool name and arguments"""
                args_str = json.dumps(kwargs, sort_keys=True, ensure_ascii=False)
                return f"{tool_name}:{args_str}"
            
            def _execute_tool_cached(self, tool_name: str, kwargs: Dict) -> str:
                """Execute tool with caching to avoid redundant calls"""
                cache_key = self._get_cache_key(tool_name, kwargs)
                
                if cache_key in self._tool_cache:
                    logger.info(f"[Cache Hit] {tool_name}({kwargs})")
                    return self._tool_cache[cache_key]
                
                # Execute and cache
                result = toolkit.call(tool_name, **kwargs)
                tool_calls_list.append({
                    'tool': tool_name,
                    'args': kwargs,
                    'result_preview': str(result)[:200]
                })
                
                result_str = json.dumps(result, ensure_ascii=False, default=str)[:500] if isinstance(result, (dict, list)) else str(result)[:500]
                self._tool_cache[cache_key] = result_str
                return result_str
            
            def get_next_step(self, history: str, step_n: int) -> str:
                """使用 MCTS_NEXT_STEP prompt"""
                prompt = MCTS_NEXT_STEP.format(
                    task=enhanced_task,
                    history=history or "(无)"
                )
                response = llm.generate(prompt)
                
                # 解析并执行工具 (使用缓存)
                action_match = re.search(r'Action[:：]\s*(\w+)\s*\(([^)]*)\)', response)
                if action_match:
                    tool_name = action_match.group(1)
                    args_str = action_match.group(2)
                    
                    if tool_name in toolkit.list_tools():
                        try:
                            kwargs = self._parse_args(args_str)
                            result_str = self._execute_tool_cached(tool_name, kwargs)
                            return f"{response}\n\n[Tool Result]\n{result_str}"
                        except Exception as e:
                            return f"{response}\n\n[Tool Error] {e}"
                
                return response
            
            def _parse_args(self, args_str: str) -> Dict[str, Any]:
                """解析参数"""
                kwargs = {}
                kv_pattern = r'(\w+)\s*=\s*(?:"([^"]*)"|\'([^\']*)\'|(\{[^}]+\})|(\[[^\]]+\])|(\w+))'
                
                for match in re.finditer(kv_pattern, args_str):
                    key = match.group(1)
                    value = match.group(2) or match.group(3) or match.group(4) or match.group(5) or match.group(6)
                    if value:
                        if value.startswith('{') or value.startswith('['):
                            try:
                                value = json.loads(value.replace("'", '"'))
                            except:
                                pass
                        elif value.isdigit():
                            value = int(value)
                        kwargs[key] = value
                
                return kwargs
            
            def get_next_step_use_reflection(self, history: str, step_n: int, reflection: str) -> str:
                """使用 MCTS_NEXT_STEP_WITH_REFLECTION prompt"""
                prompt = MCTS_NEXT_STEP_WITH_REFLECTION.format(
                    task=enhanced_task,
                    history=history or "(无)",
                    reflection=reflection,
                    step_n=step_n
                )
                response = llm.generate(prompt)
                
                # 执行工具调用 (使用缓存)
                action_match = re.search(r'Action[:：]\s*(\w+)\s*\(([^)]*)\)', response)
                if action_match:
                    tool_name = action_match.group(1)
                    args_str = action_match.group(2)
                    if tool_name in toolkit.list_tools():
                        try:
                            kwargs = self._parse_args(args_str)
                            result_str = self._execute_tool_cached(tool_name, kwargs)
                            return f"{response}\n\n[Tool Result]\n{result_str}"
                        except Exception as e:
                            return f"{response}\n\n[Tool Error] {e}"
                
                return response
            
            def get_reflection(self, history: str, step_n: int) -> str:
                """使用 MCTS_REFLECTION prompt"""
                prompt = MCTS_REFLECTION.format(
                    task=enhanced_task,
                    history=history or "(无)"
                )
                response = llm.generate(prompt)
                
                # 检查是否通过
                if "<end>" in response.lower():
                    return "<end>"
                
                return response
            
            def get_simple_reflection(self, history: str, step_n: int) -> str:
                """复用 get_reflection"""
                return self.get_reflection(history, step_n)
            
            def get_step_value(self, history: str) -> float:
                """使用 MCTS_VALUE_EVALUATION prompt"""
                prompt = MCTS_VALUE_EVALUATION.format(
                    task=enhanced_task,
                    history=history or "(无)"
                )
                response = llm.generate(prompt)
                
                # 解析分数
                score_match = re.search(r'分数[:：]\s*(0?\.\d+|1\.0|1|0)', response)
                if score_match:
                    try:
                        return float(score_match.group(1))
                    except:
                        pass
                
                # 启发式评估
                if "Error" in history:
                    return 0.3
                if "[Tool Result]" in history:
                    return 0.7
                return 0.5
        
        return OpsBenchTask()


def run_dual_system_pipeline(
    n_samples: int = 5,
    seed: int = 142,
    confidence_threshold: float = 0.5,
    verbose: bool = True
) -> List[PipelineResult]:
    """
    运行 System 1 + System 2 双系统流水线 (使用预设 Prompts)
    """
    print("=" * 70)
    print("Dual-System Pipeline (Preset Prompts + Real Tool Execution)")
    print("=" * 70)
    
    # 初始化组件
    print("\n[1] Initializing LLM...")
    llm = create_llm()
    is_mock = getattr(llm, 'is_mock', True)
    model_name = getattr(llm.config, 'model', 'N/A') if hasattr(llm, 'config') else 'N/A'
    print(f"    Mode: {'Mock' if is_mock else 'Real'} | Model: {model_name}")
    
    print("\n[2] Initializing DataToolkit...")
    data_dir = Path(__file__).parent / "sample_data"
    toolkit = DataToolkit(data_dir)
    print(f"    Tools: {len(toolkit.list_tools())} | Files: {len(toolkit.list_files())}")
    
    print("\n[3] Loading evaluation samples...")
    # DEBUG: Manual Injection for "List all failure modes of asset Chiller 6"
    cases = [EvalCase(
        id=109,
        type="FMSA",
        text="List all failure modes of asset Chiller 6.",
        category="Knowledge Query",
        deterministic=True,
        characteristic_form=""
    )]
    logger.info(f"Running pipeline with {len(cases)} cases (Manual Injection)")
    for c in cases:
        print(f"    [{c.id}] {c.type}: {c.text[:50]}...")
    
    # 初始化 Agents
    system1 = OpsBenchReActAgent(llm, toolkit, max_steps=8)
    system2 = OpsBenchMCTSAdapter(llm, toolkit, iteration_limit=5)
    
    results = []
    
    print(f"\n[4] Running pipeline ({n_samples} cases)...")
    print("-" * 70)

    
    for i, case in enumerate(cases):
        start_time = datetime.now()
        result = PipelineResult(
            case_id=case.id,
            query=case.text
        )
        
        if verbose:
            print(f"\n[Case {i+1}/{n_samples}] ID={case.id} | {case.type}/{case.category}")
            print(f"    Query: {case.text}...")
        
        try:
            # ===== System 1 =====
            if verbose:
                print("    [System 1] Running ReAct Agent (MCTS_NEXT_STEP)...")
            
            traj1, confidence1 = system1.run(case.text, case.category)
            
            result.trajectory_steps = len(traj1.steps)
            result.tool_calls = system1.tool_calls.copy()
            result.confidence = confidence1
            
            if verbose:
                print(f"    [System 1] Steps: {len(traj1.steps)} | Confidence: {confidence1:.2f} | Tools: {len(system1.tool_calls)}")
                for tc in system1.tool_calls[:3]:
                    print(f"        -> {tc['tool']}({tc['args']})")
            
            # 判断是否需要升级到 System 2
            if traj1.result == "success" and confidence1 >= confidence_threshold:
                result.system_used = "system1"
                result.success = True
                result.final_answer = traj1.metadata.get("final_answer", "")
                if verbose:
                    print(f"    [Success] System 1 solved the task")
            else:
                # ===== System 2 =====
                if verbose:
                    status = "low confidence" if traj1.result == "success" else "failed"
                    print(f"    [Escalate] System 1 {status}, escalating to System 2 (MCTS)...")
                
                traj2, confidence2 = system2.search(case.text, case.category)
                
                result.system_used = "both"
                result.trajectory_steps += len(traj2.steps)
                result.tool_calls.extend(system2.tool_calls)
                result.confidence = max(confidence1, confidence2)
                
                if verbose:
                    print(f"    [System 2] MCTS Tools: {len(system2.tool_calls)}")
                
                if traj2.result == "success":
                    result.success = True
                    result.final_answer = traj2.metadata.get("final_answer", "")
                    if verbose:
                        print(f"    [Success] System 2 solved the task")
                else:
                    result.success = False
                    if verbose:
                        print(f"    [Failed] Both systems failed")
            
            # ===== Semantic Validation =====
            checker = CaseSuccessChecker()
            agent_output = result.final_answer or (traj1.steps[-1].observation if traj1.steps else "")
            case_result = checker.check(case.text, agent_output, case.category)
            
            result.semantic_status = case_result.status
            if case_result.status == "FAIL":
                result.success = False
                if verbose:
                    print(f"    [Semantic Check] FAIL - {case_result.reason}")
            elif verbose:
                print(f"    [Semantic Check] {case_result.status}")
        
        except Exception as e:
            import traceback
            result.error = str(e)
            result.success = False
            if verbose:
                print(f"    [Error] {e}")
                traceback.print_exc()
        
        result.duration_seconds = (datetime.now() - start_time).total_seconds()
        results.append(result)
    
    # 汇总
    print("\n" + "=" * 70)
    print("Pipeline Summary")
    print("=" * 70)
    
    s1_only = sum(1 for r in results if r.system_used == "system1")
    both = sum(1 for r in results if r.system_used == "both")
    success = sum(1 for r in results if r.success)
    total_tools = sum(len(r.tool_calls) for r in results)
    total_time = sum(r.duration_seconds for r in results)
    
    print(f"- Total cases: {len(results)}")
    print(f"- Success rate: {success}/{len(results)} ({100*success/len(results):.0f}%)")
    print(f"- System 1 only: {s1_only}")
    print(f"- Escalated to System 2: {both}")
    print(f"- Total tool calls: {total_tools}")
    print(f"- Total time: {total_time:.1f}s")
    
    return results


if __name__ == "__main__":
    results = run_dual_system_pipeline(n_samples=1, seed=42, verbose=True)
    
    # 保存结果
    output_path = Path(__file__).parent / "dual_system_results.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump([{
            'case_id': r.case_id,
            'query': r.query,
            'system_used': r.system_used,
            'success': r.success,
            'final_answer': r.final_answer if r.final_answer else "",
            'tool_calls': r.tool_calls,
            'confidence': r.confidence,
            'semantic_status': r.semantic_status,
            'duration': r.duration_seconds,
            'error': r.error
        } for r in results], f, indent=2, ensure_ascii=False)
    
    print(f"\n[Saved] {output_path}")
