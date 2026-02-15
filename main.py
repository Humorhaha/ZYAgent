"""
Main Entry Point - 实验主入口

提供命令行接口运行实验。

Usage:
    python main.py                           # 默认配置
    python main.py --config path/to/config.yaml
    python main.py --ablation no_system2     # 消融实验
"""

import argparse
from pathlib import Path

from config.schema import (
    ExperimentConfig,
    no_system2_config,
    no_mcts_config,
    no_memory_config,
    no_promotion_config,
    no_hot_start_config,
    no_markov_config,
)
from pipeline.orchestrator_new import PipelineOrchestrator


# 消融预设映射
ABLATION_PRESETS = {
    "no_system2": no_system2_config,
    "no_mcts": no_mcts_config,
    "no_memory": no_memory_config,
    "no_promotion": no_promotion_config,
    "no_hot_start": no_hot_start_config,
    "no_markov": no_markov_config,
}


def main():
    parser = argparse.ArgumentParser(description="ZYAgent Experiment Runner")
    parser.add_argument("--config", type=str, help="Path to config.yaml")
    parser.add_argument("--ablation", type=str, choices=ABLATION_PRESETS.keys(),
                        help="Run ablation experiment")
    parser.add_argument("--tasks", type=str, nargs="+", 
                        default=["Solve math problem: 2+2", "Write Python function to reverse string"],
                        help="Tasks to run")
    
    args = parser.parse_args()
    
    # 加载配置
    if args.config:
        config = ExperimentConfig.from_yaml(args.config)
    elif args.ablation:
        config = ABLATION_PRESETS[args.ablation]()
    else:
        config = ExperimentConfig()
    
    print(f"=== Experiment: {config.experiment_name} ===")
    print(f"Seed: {config.seed}")
    print(f"System2: {config.system2.enabled} ({config.system2.search_strategy})")
    print(f"Memory: {config.memory.enabled} ({config.memory.backend})")
    print()
    
    # 运行实验
    orchestrator = PipelineOrchestrator(config)
    
    try:
        metrics = orchestrator.run_batch(args.tasks)
    finally:
        orchestrator.close()
    
    return metrics


if __name__ == "__main__":
    main()
