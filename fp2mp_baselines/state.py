from typing import Any, TypedDict


class AgentLogEntry(TypedDict, total=False):
    """Structured trace entry for one agent action."""

    agent: str
    event: str
    messages: list[dict[str, Any]]
    response: dict[str, Any]
    usage_metadata: dict[str, Any]
    response_metadata: dict[str, Any]
    tool_calls: list[dict[str, Any]]
    invalid_tool_calls: list[dict[str, Any]]


class TextToTextState(TypedDict):
    """LangGraph state for text input, text output, and agent trace."""

    input: str
    output: str
    log: list[AgentLogEntry]


class CotState(TextToTextState, total=False):
    """State for CoT-style baselines with an explicit public reasoning summary."""

    reasoning_summary: str


class ReactState(TextToTextState, total=False):
    """State for ReAct baselines."""

    messages: list[dict[str, Any]]


class GeneratorCriticState(TextToTextState, total=False):
    """State for Generator-Critic baselines."""

    draft: str
    critique: str


class BlackboardState(TextToTextState, total=False):
    """State for Blackboard baselines."""

    board: str
    notes: list[dict[str, Any]]
    is_final: bool


__all__ = [
    "AgentLogEntry",
    "BlackboardState",
    "CotState",
    "GeneratorCriticState",
    "ReactState",
    "TextToTextState",
]
