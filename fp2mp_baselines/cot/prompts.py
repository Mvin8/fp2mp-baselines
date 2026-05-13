COT_REASONING_HUMAN_PROMPT = """Produce reasoning for solving the task.
Highlight assumptions, constraints, key factors, the solution plan, and verification steps.

Task:
{input}"""


COT_FINAL_HUMAN_PROMPT = """User task:
{input}

Reasoning:
{reasoning_summary}

Produce the final answer for the user."""


__all__ = ["COT_FINAL_HUMAN_PROMPT", "COT_REASONING_HUMAN_PROMPT"]
