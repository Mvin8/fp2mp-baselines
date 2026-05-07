from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool

from ..state import ReactState
from .graph import build_react_graph
from .tools import ddgs_tool


class ReactBaseline:
    """Wrapper around the compiled ReAct LangGraph baseline."""

    def __init__(
        self,
        llm: BaseChatModel,
        *,
        tools: list[BaseTool] | None = None,
    ) -> None:
        self.graph = build_react_graph(
            llm=llm,
            tools=tools or [ddgs_tool],
        )

    def invoke_state(self, text: str, config: dict[str, Any] | None = None) -> ReactState:
        return self.graph.invoke({"input": text, "output": "", "log": [], "messages": []}, config=config)

    def stream(self, text: str, config: dict[str, Any] | None = None):
        return self.graph.stream({"input": text, "output": "", "log": [], "messages": []}, config=config)


__all__ = ["ReactBaseline"]
