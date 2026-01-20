"""
MCTS Demo - 演示变体组合使用

本示例展示如何:
1. 实现一个简单的环境 (数字猜谜游戏)
2. 组合多个 MCTS 变体 (PUCT + ProgressiveWidening + Reflexion)
3. 运行搜索并观察结果
"""

import sys
import os
import random
import math

# 确保可以导入本地模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mcts import Environment, Node, MCTS
from variants.selection import PUCTMCTS
from variants.expansion import ProgressiveWideningMCTS
from variants.backprop import ReflexionMCTS


# =============================================================================
# 1. 定义一个简单的环境: 数字猜谜游戏
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
    
    def get_actions(self, state: int) -> list:
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
        # 归一化到 [0, 1]
        reward = 1.0 - (distance / max_distance)
        return reward
    
    def get_prior(self, state: int, action: int) -> float:
        """先验: 根据动作方向给予先验概率"""
        new_state = state + action
        
        # 如果动作朝着目标方向，给更高先验
        if (new_state > state and state < self.target) or \
           (new_state < state and state > self.target):
            return 0.3
        elif new_state == state:
            return 0.1
        else:
            return 0.2


# =============================================================================
# 2. 组合多个变体: Deep Thinker 配置
# =============================================================================

class DeepThinkerMCTS(PUCTMCTS, ProgressiveWideningMCTS, ReflexionMCTS):
    """组合变体: PUCT + 渐进式扩展 + 反思机制
    
    配置说明:
        - Selection: PUCT 公式，利用先验引导搜索
        - Expansion: 渐进式宽度，节省计算资源
        - Backprop: Reflexion，回传失败原因避免重蹈覆辙
    """
    
    def __init__(
        self, 
        env: Environment, 
        c_puct: float = 2.0,
        widening_constant: float = 1.0,
        widening_alpha: float = 0.5,
        failure_threshold: float = 0.3,
        max_feedback_per_node: int = 3,
        max_depth: int = None,
    ):
        # 直接调用基类 MCTS 的 __init__
        MCTS.__init__(self, env, c_param=c_puct, max_depth=max_depth)
        
        # 手动设置各变体的属性
        # PUCT
        self.c_puct = c_puct
        
        # Progressive Widening
        self.widening_constant = widening_constant
        self.widening_alpha = widening_alpha
        
        # Reflexion
        self.failure_threshold = failure_threshold
        self.max_feedback_per_node = max_feedback_per_node


# =============================================================================
# 3. 运行演示
# =============================================================================

def demo_vanilla_mcts():
    """演示: Vanilla MCTS"""
    print("=" * 60)
    print("Demo 1: Vanilla MCTS (Random Rollout)")
    print("=" * 60)
    
    from mcts import VanillaMCTS
    
    env = NumberGuessingEnv(target=42)
    mcts = VanillaMCTS(env, c_param=1.414, max_depth=20)
    
    initial_state = 0
    best_action = mcts.search(initial_state, num_simulations=50)
    
    print(f"Target: {env.target}")
    print(f"Initial state: {initial_state}")
    print(f"Best action: {best_action:+d}")
    print(f"Stats: {mcts.get_stats()}")
    print()


def demo_puct_mcts():
    """演示: PUCT 变体"""
    print("=" * 60)
    print("Demo 2: PUCT MCTS (Prior-guided)")
    print("=" * 60)
    
    env = NumberGuessingEnv(target=75)
    mcts = PUCTMCTS(env, c_puct=2.0, max_depth=20)
    
    initial_state = 50
    best_action = mcts.search(initial_state, num_simulations=50)
    
    print(f"Target: {env.target}")
    print(f"Initial state: {initial_state}")
    print(f"Best action: {best_action:+d}")
    print(f"Stats: {mcts.get_stats()}")
    print()


def demo_progressive_widening():
    """演示: 渐进式扩展变体"""
    print("=" * 60)
    print("Demo 3: Progressive Widening MCTS")
    print("=" * 60)
    
    env = NumberGuessingEnv(target=30)
    mcts = ProgressiveWideningMCTS(
        env, 
        widening_constant=1.0,
        widening_alpha=0.5,
        max_depth=20
    )
    
    initial_state = 50
    best_action = mcts.search(initial_state, num_simulations=50)
    
    print(f"Target: {env.target}")
    print(f"Initial state: {initial_state}")
    print(f"Best action: {best_action:+d}")
    print(f"Stats: {mcts.get_stats()}")
    print()


def demo_deep_thinker():
    """演示: 组合变体 (Deep Thinker)"""
    print("=" * 60)
    print("Demo 4: Deep Thinker MCTS (Combined Variants)")
    print("=" * 60)
    
    env = NumberGuessingEnv(target=88)
    mcts = DeepThinkerMCTS(env, max_depth=20)
    
    initial_state = 10
    best_action = mcts.search(initial_state, num_simulations=100)
    
    print(f"Target: {env.target}")
    print(f"Initial state: {initial_state}")
    print(f"Best action: {best_action:+d}")
    print(f"Stats: {mcts.get_stats()}")
    
    # 打印搜索树
    print("\nSearch Tree (depth=2):")
    mcts.print_tree(max_depth=2)
    print()


def demo_full_game():
    """演示: 完整游戏流程"""
    print("=" * 60)
    print("Demo 5: Full Game with MCTS Agent")
    print("=" * 60)
    
    target = random.randint(1, 99)
    env = NumberGuessingEnv(target=target)
    mcts = DeepThinkerMCTS(env, max_depth=15)
    
    state = 50  # 从中间开始
    steps = 0
    max_steps = 20
    
    print(f"Target (hidden): {target}")
    print(f"Starting at: {state}")
    print("-" * 40)
    
    while not env.is_terminal(state) and steps < max_steps:
        # 使用 MCTS 选择动作
        action = mcts.search(state, num_simulations=30)
        new_state = env.step(state, action)
        
        print(f"Step {steps + 1}: {state} + ({action:+d}) = {new_state}")
        
        state = new_state
        steps += 1
        
        # 重置 MCTS 根节点用于下一步搜索
        mcts.root = None
    
    print("-" * 40)
    if env.is_terminal(state):
        print(f"✅ Found target {target} in {steps} steps!")
    else:
        print(f"❌ Failed to find target. Final state: {state}, distance: {abs(state - target)}")


if __name__ == "__main__":
    demo_vanilla_mcts()
    demo_puct_mcts()
    demo_progressive_widening()
    demo_deep_thinker()
    demo_full_game()
    
    print("=" * 60)
    print("All demos completed!")
    print("=" * 60)
