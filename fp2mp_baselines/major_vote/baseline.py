from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from ..state import MajorVoteState
from .graph import build_major_vote_graph


class MajorVoteBaseline:
    """Wrapper around the compiled Majority Voting LangGraph baseline."""

    def __init__(
        self,
        llm: BaseChatModel,
        *,
        num_agents: int = 5,
        vote_llm: BaseChatModel | None = None,
    ) -> None:
        self.graph = build_major_vote_graph(
            llm=llm,
            num_agents=num_agents,
            vote_llm=vote_llm,
        )

    def invoke_state(self, text: str, config: dict[str, Any] | None = None) -> MajorVoteState:
        return self.graph.invoke(
            {"input": text, "output": "", "log": [], "agent_responses": [], "vote_counts": {}},
            config=config,
        )

    def stream(self, text: str, config: dict[str, Any] | None = None):
        return self.graph.stream(
            {"input": text, "output": "", "log": [], "agent_responses": [], "vote_counts": {}},
            config=config,
        )


__all__ = ["MajorVoteBaseline"]
