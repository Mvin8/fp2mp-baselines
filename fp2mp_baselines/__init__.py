"""Baselines for text-to-text FP2MP planning tasks."""

try:
    from importlib.metadata import version

    __version__ = version("fp2mp-baselines")
except Exception:  # pragma: no cover - package may be used before installation
    __version__ = "0.0.0"

from .blackboard import BlackboardBaseline, build_blackboard_graph
from .cot import CotBaseline, build_cot_graph
from .generator_critic import GeneratorCriticBaseline, build_generator_critic_graph
from .react import ReactBaseline, build_react_graph, ddgs_tool
from .single_agent import SingleAgentBaseline, build_single_agent_graph
from .state import AgentLogEntry, BlackboardState, CotState, GeneratorCriticState, ReactState, TextToTextState


__all__ = [
    "AgentLogEntry",
    "BlackboardBaseline",
    "BlackboardState",
    "CotBaseline",
    "CotState",
    "GeneratorCriticBaseline",
    "GeneratorCriticState",
    "ReactBaseline",
    "ReactState",
    "SingleAgentBaseline",
    "TextToTextState",
    "build_blackboard_graph",
    "build_cot_graph",
    "build_generator_critic_graph",
    "build_react_graph",
    "build_single_agent_graph",
    "ddgs_tool",
]
