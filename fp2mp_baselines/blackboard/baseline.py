from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from ..state import BlackboardState
from .graph import build_blackboard_graph


class BlackboardBaseline:
    """Wrapper around the compiled Blackboard LangGraph baseline."""

    def __init__(
        self,
        llm: BaseChatModel,
        *,
        iterations: int = 3,
    ) -> None:
        self.graph = build_blackboard_graph(
            llm=llm,
            iterations=iterations,
        )

    def invoke(self, text: str, config: dict[str, Any] | None = None) -> str:
        result = self.invoke_state(text, config=config)
        return result["output"]

    def invoke_state(self, text: str, config: dict[str, Any] | None = None) -> BlackboardState:
        return self.graph.invoke(
            {"input": text, "output": "", "log": [], "board": "", "notes": [], "is_final": False},
            config=config,
        )

    def stream(self, text: str, config: dict[str, Any] | None = None):
        return self.graph.stream(
            {"input": text, "output": "", "log": [], "board": "", "notes": [], "is_final": False},
            config=config,
        )


__all__ = ["BlackboardBaseline"]
