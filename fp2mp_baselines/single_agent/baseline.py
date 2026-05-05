from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from ..state import TextToTextState
from .graph import build_single_agent_graph


class SingleAgentBaseline:
    """Small wrapper around the compiled single-agent LangGraph baseline."""

    def __init__(
        self,
        llm: BaseChatModel,
    ) -> None:
        self.graph = build_single_agent_graph(llm=llm)

    def invoke(self, text: str, config: dict[str, Any] | None = None) -> str:
        result = self.invoke_state(text, config=config)
        return result["output"]

    def invoke_state(self, text: str, config: dict[str, Any] | None = None) -> TextToTextState:
        return self.graph.invoke({"input": text, "output": "", "log": []}, config=config)

    def stream(self, text: str, config: dict[str, Any] | None = None):
        return self.graph.stream({"input": text, "output": "", "log": []}, config=config)


__all__ = ["SingleAgentBaseline"]
