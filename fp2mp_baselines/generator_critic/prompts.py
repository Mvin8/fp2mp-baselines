GENERATOR_DRAFT_HUMAN_PROMPT = """Generate a draft answer for the user's task.

Task:
{input}"""


CRITIC_HUMAN_PROMPT = """Review the draft answer for the user's task.
Identify weaknesses, omissions, contradictions, and opportunities for improvement.

Task:
{input}

Draft answer:
{draft}"""


GENERATOR_FINAL_HUMAN_PROMPT = """Produce the final answer for the user, taking the critique into account.

Task:
{input}

Draft answer:
{draft}

Critique:
{critique}"""


__all__ = [
    "CRITIC_HUMAN_PROMPT",
    "GENERATOR_DRAFT_HUMAN_PROMPT",
    "GENERATOR_FINAL_HUMAN_PROMPT",
]
