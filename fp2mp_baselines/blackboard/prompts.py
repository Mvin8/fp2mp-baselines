GENERATOR_PROMPT = """
Create 1 to 3 expert roles that will help solve the task through a shared blackboard.
The roles should be specialized and complementary.
Do not create controller, planner, critic, cleaner, or arbiter roles.
---
Task:
{question}
"""


CONTROLLER_PROMPT = """
Your task is to assign other agents to collaborate on solving this task.
Agents exchange information through a shared blackboard.
All current blackboard messages will be provided to you as separate messages below.
Based on them, choose suitable agents from the list so they can leave new notes on the blackboard.
---
Available agents:
{workers}
---
Task:
{question}
"""


WORKER_PROMPT = """
Your personal ID is {id}.
You are {role_name}, collaborating with other agents to solve the task.
There is a shared blackboard that each of you can read and write notes to.
All current blackboard messages will be provided to you as separate messages below.
---
Working rules:
- Do not duplicate information that is already on the blackboard.
- Do not ask other agents for information.
- Only perform your part of the work.
---
Role description:
{role_description}
---
Task:
{question}
---
"""


EXPERT_PROMPT = """
You are an expert: {role_name}.
Your area of expertise: {role_description}

Analyze the original task and all messages on the blackboard.
Add one substantive note that moves the solution forward within your area of expertise.
You may disagree with previous messages if you explain why.
"""


PLANNER_PROMPT = """
Create a plan for solving the original task based on the current contents of the shared blackboard.
Describe the task in your own words, then provide a step-by-step plan for solving it.
If a plan already exists on the blackboard or the task is simple enough for a direct solution,
state that decomposition is unnecessary and that you are waiting for additional information.
Do not solve the task. Provide only the plan.
"""


CRITIC_PROMPT = """
Analyze the notes on the shared blackboard and identify incorrect, useless, or redundant entries.
If such entries exist, describe them and explain why they should be corrected or removed.
If there are no such entries, state that there are no obvious problems and that you are waiting for additional information.
"""


CLEANER_PROMPT = """
Analyze the notes on the shared blackboard and identify useless or redundant entries.
If such entries exist, list their IDs. If there are no useless notes, return an empty list.
"""


DECIDER_PROMPT = """
Analyze the current state of the shared blackboard and decide whether there is enough information
to produce the final answer.
If the information is sufficient, indicate that the work is complete.
If more information from other agents is needed, indicate that the process should continue.
Do not solve the task.
"""


SUMMARIZER_PROMPT = """
Produce the final answer to the task using the contents of the blackboard.
The answer must be final and must not mention the blackboard, the agents, or the discussion process.
---
Task:
{question}
"""


__all__ = [
    "CLEANER_PROMPT",
    "CONTROLLER_PROMPT",
    "CRITIC_PROMPT",
    "DECIDER_PROMPT",
    "EXPERT_PROMPT",
    "GENERATOR_PROMPT",
    "PLANNER_PROMPT",
    "SUMMARIZER_PROMPT",
    "WORKER_PROMPT",
]
