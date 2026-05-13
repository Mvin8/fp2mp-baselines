from typing import Any, TypedDict

from langchain_core.messages import BaseMessage


class BaseState(TypedDict):
    input: str
    output: str
    log: list[BaseMessage]


class CotState(BaseState, total=False):
    """State for CoT-style baselines with an explicit public reasoning summary."""

    reasoning_summary: str


class ReactState(BaseState, total=False):
    """State for ReAct baselines."""

    messages: list[dict[str, Any]]


class GeneratorCriticState(BaseState, total=False):
    """State for Generator-Critic baselines."""

    draft: str
    critique: str


class MajorVoteState(BaseState, total=False):
    """State for Majority Voting baselines."""

    agent_responses: list[str]
    vote_counts: dict[str, int]


class DebateState(BaseState, total=False):
    """State for Multi-Agent Debate baselines."""

    agent_responses: list[str]
    rounds: list[list[str]]
    vote_counts: dict[str, int]


class BlackboardState(BaseState, total=False):
    """State for Blackboard baselines."""

    board: str
    notes: list[dict[str, Any]]
    is_final: bool


__all__ = [
    "BaseState",
    "BlackboardState",
    "CotState",
    "DebateState",
    "GeneratorCriticState",
    "MajorVoteState",
    "ReactState",
]
