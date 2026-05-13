from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from ..state import DebateState
from .graph import build_debate_graph


class DebateBaseline:
    """Wrapper around the compiled Multi-Agent Debate LangGraph baseline."""

    def __init__(
        self,
        llm: BaseChatModel,
        *,
        num_agents: int = 5,
        debate_rounds: int = 5,
        vote_llm: BaseChatModel | None = None,
    ) -> None:
        self.graph = build_debate_graph(
            llm=llm,
            num_agents=num_agents,
            debate_rounds=debate_rounds,
            vote_llm=vote_llm,
        )

    def invoke_state(self, text: str, config: dict[str, Any] | None = None) -> DebateState:
        return self.graph.invoke(
            {"input": text, "output": "", "log": [], "agent_responses": [], "rounds": [], "vote_counts": {}},
            config=config,
        )

    def stream(self, text: str, config: dict[str, Any] | None = None):
        return self.graph.stream(
            {"input": text, "output": "", "log": [], "agent_responses": [], "rounds": [], "vote_counts": {}},
            config=config,
        )


__all__ = ["DebateBaseline"]
