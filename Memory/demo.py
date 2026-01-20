"""
Memory 模块演示脚本

演示 TTS + HCC 混合架构的使用方式。
"""

from memory import AgentMemory, MemoryConfig, TrajectoryCategory
from tts import Trajectory, TrajectoryStep
from hcc import EventType


def demo_basic_usage():
    """基础使用演示"""
    print("=" * 60)
    print("Demo 1: Basic Usage")
    print("=" * 60)
    
    # 1. 初始化 Memory
    memory = AgentMemory()
    print(f"Initialized: {memory}")
    
    # 2. 手动添加一个示例轨迹
    example = Trajectory(
        trajectory_id="ml_example_1",
        category=TrajectoryCategory.DATA_SCIENCE,
        task="How to train a ResNet model for image classification?",
        steps=[
            TrajectoryStep(
                step_id=1,
                thought="First, I need to load and preprocess the dataset",
                action="LoadDataset",
                action_input="imagenet_subset",
                observation="Dataset loaded: 10000 images, 10 classes"
            ),
            TrajectoryStep(
                step_id=2,
                thought="Now I should define the model architecture",
                action="DefineModel",
                action_input="ResNet50(pretrained=True)",
                observation="Model created with 25M parameters"
            ),
            TrajectoryStep(
                step_id=3,
                thought="Time to train the model",
                action="Train",
                action_input="epochs=10, lr=1e-4",
                observation="Training complete. Accuracy: 92%"
            ),
        ],
        final_answer="Use pretrained ResNet50 with lr=1e-4 for 10 epochs."
    )
    memory.add_example(example)
    print(f"Added example: {example.trajectory_id}")
    
    # 3. 开始一个新任务
    print("\n--- Starting new task ---")
    initial_context = memory.start_task(
        task_description="Build an image classifier for plant diseases",
        user_instruction="Maximize F1 score"
    )
    print(f"Initial context:\n{initial_context[:200]}...")
    
    # 4. 模拟任务执行过程
    print("\n--- Recording events ---")
    memory.record_thought_action(
        thought="I should first explore the dataset structure",
        action="ExploreData",
        observation="Found 5000 images with 4 disease classes"
    )
    
    memory.record_thought_action(
        thought="The dataset is imbalanced, I need to handle this",
        action="AnalyzeDistribution",
        observation="Class distribution: [2000, 1500, 1000, 500]"
    )
    
    memory.record_thought_action(
        thought="I'll use data augmentation to balance classes",
        action="ApplyAugmentation",
        observation="Applied RandomRotation, RandomFlip, ColorJitter"
    )
    
    print(f"Recorded {memory._step_count} steps")
    
    # 5. 构建完整 Prompt
    print("\n--- Building prompt ---")
    prompt = memory.build_prompt(
        task="What model architecture should I use?",
        system_prompt="You are a machine learning expert.",
    )
    print(f"Prompt length: {len(prompt)} chars")
    print(f"\n{prompt[:500]}...")
    
    # 6. 查看统计
    print("\n--- Statistics ---")
    stats = memory.get_stats()
    print(f"TTS: {stats['tts']}")
    print(f"HCC: {stats['hcc']}")


def demo_tts_retrieval():
    """TTS 检索演示"""
    print("\n" + "=" * 60)
    print("Demo 2: TTS Retrieval")
    print("=" * 60)
    
    memory = AgentMemory()
    
    # 添加多个不同类别的示例
    examples = [
        Trajectory(
            trajectory_id="code_1",
            category=TrajectoryCategory.CODE_GENERATION,
            task="Write a Python function to calculate factorial",
            steps=[
                TrajectoryStep(1, "I need a recursive approach", "WriteCode", "def factorial(n):", "Code written")
            ],
            final_answer="def factorial(n): return 1 if n <= 1 else n * factorial(n-1)"
        ),
        Trajectory(
            trajectory_id="reasoning_1",
            category=TrajectoryCategory.REASONING,
            task="If all A are B, and all B are C, are all A also C?",
            steps=[
                TrajectoryStep(1, "This is a syllogism problem", "Analyze", "transitive property", "Yes, by transitivity")
            ],
            final_answer="Yes, all A are C (transitive property)"
        ),
        Trajectory(
            trajectory_id="ml_2",
            category=TrajectoryCategory.DATA_SCIENCE,
            task="How to handle missing values in a dataset?",
            steps=[
                TrajectoryStep(1, "First check the percentage of missing values", "Analyze", "df.isnull().sum()", "Found 5% missing")
            ],
            final_answer="Use mean imputation for numerical, mode for categorical"
        ),
    ]
    
    for ex in examples:
        memory.add_example(ex)
    
    print(f"Loaded {len(memory.tts)} examples")
    
    # 按类别检索
    print("\n--- Retrieve by category ---")
    ds_examples = memory.get_examples(category=TrajectoryCategory.DATA_SCIENCE)
    print(f"Data Science examples: {len(ds_examples)}")
    
    # 按查询检索
    print("\n--- Retrieve by query ---")
    results = memory.get_examples(query="How to write Python code?", k=2)
    for r in results:
        print(f"  - [{r.category.value}] {r.task[:50]}...")


def demo_hcc_promotion():
    """HCC 阶段提升演示"""
    print("\n" + "=" * 60)
    print("Demo 3: HCC Phase Promotion")
    print("=" * 60)
    
    # 配置较小的 auto_promote_steps 以便演示
    config = MemoryConfig(hcc_auto_promote_steps=5)
    memory = AgentMemory(config=config)
    
    memory.start_task("Complex multi-phase task")
    
    print("--- Simulating task execution ---")
    for i in range(12):
        memory.record_event(f"Step {i}: Doing something important", EventType.AGENT)
        # 注意: 每 5 步会自动触发 phase promotion
    
    print(f"\nFinal stats: {memory.get_stats()['hcc']}")
    
    # 结束任务
    memory.finish_task("Task completed successfully")
    print("Task finished, wisdom saved to L3")


def demo_prompt_structure():
    """Prompt 结构演示"""
    print("\n" + "=" * 60)
    print("Demo 4: Prompt Structure (Sandwich)")
    print("=" * 60)
    
    memory = AgentMemory()
    
    # 添加示例
    memory.add_example(Trajectory(
        trajectory_id="demo",
        category=TrajectoryCategory.GENERAL,
        task="Sample question",
        steps=[TrajectoryStep(1, "Think", "Act", "input", "output")],
        final_answer="Sample answer"
    ))
    
    # 记录一些上下文
    memory.start_task("Demo task")
    memory.record_event("Previous exploration result", EventType.ENVIRONMENT)
    
    # 构建 Prompt
    prompt = memory.build_prompt(
        task="What should I do next?",
        system_prompt="You are a helpful assistant.",
    )
    
    print("Generated Prompt Structure:")
    print("-" * 40)
    print(prompt)
    print("-" * 40)


if __name__ == "__main__":
    demo_basic_usage()
    demo_tts_retrieval()
    demo_hcc_promotion()
    demo_prompt_structure()
    
    print("\n" + "=" * 60)
    print("All demos completed!")
    print("=" * 60)
