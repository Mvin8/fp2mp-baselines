from collections import Counter
from dataclasses import dataclass

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage

from .graph_utils import build_message_log, message_content_to_text


VOTE_TIE_BREAKER_PROMPT = """Choose the final answer based on the votes of several independent agents.

Rules:
- group semantically equivalent answers as one option;
- choose the option supported by the majority of agents;
- if there is no clear majority, choose the most complete and well-justified answer;
- return only the final answer, without explaining the voting procedure.

Task:
{input}

Agent responses:
{responses}"""


@dataclass(frozen=True)
class VoteResult:
    answer: str
    counts: dict[str, int]
    log: list[BaseMessage]


def normalize_vote(text: str) -> str:
    return " ".join(text.strip().lower().split())


def format_agent_responses(responses: list[str]) -> str:
    return "\n\n".join(f"Agent {index + 1}:\n{response}" for index, response in enumerate(responses))


def aggregate_majority_vote(
    *,
    input_text: str,
    responses: list[str],
    llm: BaseChatModel,
) -> VoteResult:
    if not responses:
        return VoteResult(answer="", counts={}, log=[])

    normalized = [normalize_vote(response) for response in responses]
    counts = Counter(normalized)
    first_by_normalized: dict[str, str] = {}
    for key, response in zip(normalized, responses):
        first_by_normalized.setdefault(key, response)
    max_count = max(counts.values())
    winners = [answer for answer, count in counts.items() if count == max_count]

    if len(winners) == 1:
        winner = winners[0]
        return VoteResult(answer=first_by_normalized[winner], counts=dict(counts), log=[])

    messages = [
        HumanMessage(
            content=VOTE_TIE_BREAKER_PROMPT.format(
                input=input_text,
                responses=format_agent_responses(responses),
            )
        )
    ]
    response = llm.invoke(messages)
    return VoteResult(
        answer=message_content_to_text(response.content),
        counts=dict(counts),
        log=build_message_log(messages, response),
    )


__all__ = [
    "VoteResult",
    "aggregate_majority_vote",
    "format_agent_responses",
    "normalize_vote",
]
