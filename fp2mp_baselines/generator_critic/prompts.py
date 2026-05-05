GENERATOR_DRAFT_HUMAN_PROMPT = """Сгенерируй черновой ответ на задачу пользователя.

Задача:
{input}"""


CRITIC_HUMAN_PROMPT = """Проверь черновой ответ на задачу пользователя.
Найди слабые места, пропуски, противоречия и возможности улучшения.

Задача:
{input}

Черновой ответ:
{draft}"""


GENERATOR_FINAL_HUMAN_PROMPT = """Сформируй финальный ответ пользователю, учитывая критику.

Задача:
{input}

Черновой ответ:
{draft}

Критика:
{critique}"""


__all__ = [
    "CRITIC_HUMAN_PROMPT",
    "GENERATOR_DRAFT_HUMAN_PROMPT",
    "GENERATOR_FINAL_HUMAN_PROMPT",
]
