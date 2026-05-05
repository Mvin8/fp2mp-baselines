COT_REASONING_HUMAN_PROMPT = """Сформируй reasoning для решения задачи.
Выдели допущения, ограничения, ключевые факторы, план решения и проверки.

Задача:
{input}"""


COT_FINAL_HUMAN_PROMPT = """Задача пользователя:
{input}

Reasoning:
{reasoning_summary}

Сформируй финальный ответ пользователю."""


__all__ = ["COT_FINAL_HUMAN_PROMPT", "COT_REASONING_HUMAN_PROMPT"]
