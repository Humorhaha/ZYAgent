
system_prompt = """You are a helpful and precise assistant for answering questions.
"""



ROOT_TASK_DECOMPOSITION_PROMPT = """
{system_prompt}
Here’s the rule of the environment:
{rule}
{init_obs}
Now you need to find the answer for the following question using the
actions I provide.
Here is the description:
{goal}
Now, start the task. Please firstly generate a list of general subtasks to
accomplish the task. And generate the thought process for the task. 
Your answer should be in the following json format:
{{   
    "thought": "your thought process for the task",
    "sub_tasks": ["subtask1", "subtask2", ...]
}}

"""



DOWNWARD_PROMPT = """
Your current task: {goal}
We wish you to generate a list of subtasks for the current task and the thought process for the task.
Your answer should be in the following json format:
{{   
    "thought": "your thought process for the task",
    "sub_tasks": ["subtask1", "subtask2", ...]
}}
"""



LEAF_BACKTRACK_PROMPT = """
You have completed the task: {done_task_name}
Here is the result:
{obs}
Now, you return to the parent task:
Your current task: {previous_stage_task_name}
Your previous think: {previous_stage_think}
Your remaining subtasks:
{remaining_subtask_str}
We wish you to refine your list of subtasks based on the latest observation
to achieve your goal.
If there are no remaining subtasks, check if the goal is achieved.
If yes, return an empty list; otherwise, return the required subtasks.
Do not generate subtasks beyond the current task.
Also, return the updated thought process for the task.
Your answer should be in the following json format:
{{   
    "thought": "your updated thought process for the task",
    "sub_tasks": ["subtask1", "subtask2", ...]
}}
"""




NON_LEAF_COMPLETION_PROMPT = """
Fires when a non-leaf node has no subtasks left and needs to decide if it’s fully done or generate
more.
You have successfully completed the task: {done_task_name}
Now, you return to the previous stage.
Your current task: {previous_stage_task_name}
Your previous think: {previous_stage_think}
There are no remaining subtasks. Determine if the task is complete.
If it is, set subtasks to an empty list; if not, generate necessary
subtasks.
"""




LEAF_COMPLETION_PROMPT = """
You have completed the task: {done_task_name}
Here is the result:
{obs}
Now, you return to the previous stage.
Your current task: {previous_stage_task_name}
Your previous think: {previous_stage_think}
There are no remaining subtasks. Determine if the task is complete.
If it is, set subtasks to an empty list; if not, generate necessary
subtasks."""




LEAF_FAILURE_PROMPT = """
You fail to complete the task: {fail_task_name}
Because the action is not among the valid options.
{obs}
Now, you return to the previous stage.
Your current task: {previous_stage_task_name}
Your previous think: {previous_stage_think}
Your remaining subtasks:
{remaining_subtask_str}
We wish you to modify your subtasks to fix the error.
"""




NON_LEAF_BACKTRACK_PROMPT = """
You have completed the task: {done_task_name}
The result shows in the previous context.
Now, you return to the parent task:
Your current task: {previous_stage_task_name}
Your previous think: {previous_stage_think}
Your remaining subtasks:
{remaining_subtask_str}
We wish you to refine your list of subtasks based on the latest observation
to achieve your goal.
Do not generate subtasks beyond the current task.
Also, return the updated thought process for the task.
Your answer should be in the following json format:
{{   
    "thought": "your updated thought process for the task",
    "sub_tasks": ["subtask1", "subtask2", ...]
}}
"""