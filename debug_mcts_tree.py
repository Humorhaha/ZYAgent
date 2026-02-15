
import sys
import os

# Ensure we can import modules from current directory
sys.path.append(os.getcwd())

from MCTS.base import MCTS
from system2.mcts import MCTSAdapter, MCTSTask
from LLM.llm import create_llm
from core.failure import FailureCase
from core.trajectory import Trajectory
from config import System2Config

class DebugMCTSTask(MCTSTask):
    """Wrapper to connect MCTS implementation with our Adapter"""
    def __init__(self, adapter):
        super().__init__(
            iteration_limit=3,    # Run 3 iterations
            branch=2,             # Branch factor 2
            use_reflection='common'
        )
        self.adapter = adapter

    def get_next_step(self, history, step_n):
        return self.adapter.get_next_step(history, step_n)

    def get_next_step_use_reflection(self, history, step_n, reflection):
        return self.adapter.get_next_step_use_reflection(history, step_n, reflection)

    def get_reflection(self, history, step_n):
        return self.adapter.get_reflection(history, step_n)

    def get_simple_reflection(self, history, step_n):
        return self.adapter.get_reflection(history, step_n) # alias

    def get_step_value(self, history):
        return self.adapter.get_step_value(history)

def run_tree_demo():
    print(">>> Initializing LLM & MCTS...")
    llm = create_llm()
    
    # Mock Trajectory & FailureCase
    traj = Trajectory(id="mock_traj_001", task="Diagnose intermittent failure of Sensor T-204")
    failure_case = FailureCase(
        task="Diagnose intermittent failure of Sensor T-204",
        trajectory=traj
    )
    
    adapter = MCTSAdapter(failure_case, System2Config(), llm=llm)
    task = DebugMCTSTask(adapter)
    mcts = MCTS(task)
    
    print(">>> Starting MCTS Search (3 Iterations)...")
    mcts.search()
    
    print("\n>>> Final Tree Structure:")
    mcts.print_tree(max_depth=4)

if __name__ == "__main__":
    run_tree_demo()
