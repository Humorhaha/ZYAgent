"""
MCTS Demo - 演示变体组合使用 (工程化版本)

本示例展示如何:
1. 使用 Config 统一管理参数
2. 组合多个 MCTS 变体
3. 运行搜索并观察结果
"""

import sys
import os
import random
from typing import List

# 确保可以导入本地模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mcts import Environment, Node, MCTS, VanillaMCTS
from variants.selection import PUCTMCTS
from variants.expansion import ProgressiveWideningMCTS
from variants.backprop import ReflexionMCTS
from config import MCTSConfig, EconomicalConfig, BalancedConfig, DeepThinkerConfig


# =============================================================================
# 1. 环境定义
# =============================================================================

class NumberGuessingEnv(Environment[int, int]):
    """数字猜谜游戏环境
    
    规则:
        - 目标是找到一个隐藏的目标数字 (target)
        - 状态是当前猜测的数字
        - 动作是调整值 (-10, -5, -1, +1, +5, +10)
        - 奖励基于与目标的距离
    """
    
    def __init__(self, target: int = 50, min_val: int = 0, max_val: int = 100):
        self.target = target
        self.min_val = min_val
        self.max_val = max_val
        self.actions = [-10, -5, -1, +1, +5, +10]
    
    def get_actions(self, state: int) -> List[int]:
        """获取合法动作 (不能越界)"""
        valid = []
        for a in self.actions:
            new_state = state + a
            if self.min_val <= new_state <= self.max_val:
                valid.append(a)
        return valid if valid else [0]
    
    def step(self, state: int, action: int) -> int:
        """执行动作"""
        new_state = state + action
        return max(self.min_val, min(self.max_val, new_state))
    
    def is_terminal(self, state: int) -> bool:
        """是否猜中目标"""
        return state == self.target
    
    def get_reward(self, state: int) -> float:
        """奖励: 越接近目标越高"""
        distance = abs(state - self.target)
        max_distance = self.max_val - self.min_val
        return 1.0 - (distance / max_distance)
    
    def get_prior(self, state: int, action: int) -> float:
        """先验: 根据动作方向给予先验概率"""
        new_state = state + action
        if (new_state > state and state < self.target) or \
           (new_state < state and state > self.target):
            return 0.3
        elif new_state == state:
            return 0.1
        return 0.2


# =============================================================================
# 2. 组合变体 (使用 Config)
# =============================================================================

class ConfigurableMCTS(PUCTMCTS, ProgressiveWideningMCTS, ReflexionMCTS):
    """可配置的组合 MCTS
    
    通过 MCTSConfig 统一管理所有参数。
    """
    
    def __init__(self, env: Environment, config: MCTSConfig):
        """
        Args:
            env: 环境实例
            config: 配置对象
        """
        # 存储配置引用
        self.config = config
        
        # 调用基类初始化
        MCTS.__init__(self, env, c_param=config.c_puct, max_depth=config.max_depth)
        
        # Selection 参数
        self.c_puct = config.c_puct
        
        # Expansion 参数
        self.widening_constant = config.widening_constant
        self.widening_alpha = config.widening_alpha
        
        # Backprop 参数
        self.failure_threshold = config.failure_threshold
        self.max_feedback_per_node = config.max_feedback_per_node
    
    def search_with_config(self, initial_state) -> int:
        """使用配置中的 num_simulations 进行搜索"""
        return self.search(initial_state, num_simulations=self.config.num_simulations)


# =============================================================================
# 3. Demo Runner
# =============================================================================

class MCTSDemo:
    """MCTS 演示运行器"""
    
    def __init__(self, env: Environment, config: MCTSConfig, name: str = "Demo"):
        self.env = env
        self.config = config
        self.name = name
        self.mcts = ConfigurableMCTS(env, config)
    
    def run_single_search(self, initial_state: int) -> dict:
        """运行单次搜索"""
        best_action = self.mcts.search_with_config(initial_state)
        stats = self.mcts.get_stats()
        
        return {
            "initial_state": initial_state,
            "best_action": best_action,
            "stats": stats,
        }
    
    def run_full_game(self, initial_state: int, max_steps: int = 20) -> dict:
        """运行完整游戏"""
        state = initial_state
        steps = 0
        trajectory = []
        
        while not self.env.is_terminal(state) and steps < max_steps:
            action = self.mcts.search_with_config(state)
            new_state = self.env.step(state, action)
            
            trajectory.append({
                "step": steps + 1,
                "state": state,
                "action": action,
                "new_state": new_state,
            })
            
            state = new_state
            steps += 1
            self.mcts.root = None  # 重置根节点
        
        return {
            "success": self.env.is_terminal(state),
            "final_state": state,
            "total_steps": steps,
            "trajectory": trajectory,
        }
    
    def print_result(self, result: dict, show_trajectory: bool = True) -> None:
        """打印结果"""
        print(f"\n{'=' * 60}")
        print(f"Demo: {self.name}")
        print(f"Config: {self.config.__class__.__name__}")
        print(f"{'=' * 60}")
        
        if "trajectory" in result:
            # 完整游戏结果
            status = "✅ Success" if result["success"] else "❌ Failed"
            print(f"Result: {status}")
            print(f"Total Steps: {result['total_steps']}")
            print(f"Final State: {result['final_state']}")
            
            if show_trajectory:
                print(f"\nTrajectory:")
                for step in result["trajectory"]:
                    print(f"  Step {step['step']}: {step['state']} + ({step['action']:+d}) = {step['new_state']}")
        else:
            # 单次搜索结果
            print(f"Initial State: {result['initial_state']}")
            print(f"Best Action: {result['best_action']:+d}")
            print(f"Stats: {result['stats']}")


# =============================================================================
# 4. Main
# =============================================================================

def main():
    """主函数"""
    print("=" * 60)
    print("MCTS Demo with Configuration")
    print("=" * 60)
    
    # 定义目标
    target = random.randint(10, 90)
    print(f"\nTarget (hidden): {target}")
    
    # 创建环境
    env = NumberGuessingEnv(target=target)
    
    # 测试不同配置
    configs = [
        ("Economical", EconomicalConfig()),
        ("Balanced", BalancedConfig()),
        ("DeepThinker", DeepThinkerConfig()),
    ]
    
    for name, config in configs:
        demo = MCTSDemo(env, config, name=name)
        result = demo.run_full_game(initial_state=50)
        demo.print_result(result, show_trajectory=False)
    
    # 详细展示 Balanced 配置
    print("\n" + "=" * 60)
    print("Detailed Run with Balanced Config")
    print("=" * 60)
    
    demo = MCTSDemo(env, BalancedConfig(), name="Balanced (Detailed)")
    result = demo.run_full_game(initial_state=50)
    demo.print_result(result, show_trajectory=True)


if __name__ == "__main__":
    main()
