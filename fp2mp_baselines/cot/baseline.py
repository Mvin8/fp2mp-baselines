from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from ..state import CotState
from .graph import build_cot_graph


class CotBaseline:
    """Wrapper around the compiled CoT-style LangGraph baseline."""

    def __init__(
        self,
        llm: BaseChatModel,
    ) -> None:
        self.graph = build_cot_graph(llm=llm)

    def invoke_state(self, text: str, config: dict[str, Any] | None = None) -> CotState:
        return self.graph.invoke({"input": text, "output": "", "log": [], "reasoning_summary": ""}, config=config)

    def stream(self, text: str, config: dict[str, Any] | None = None):
        return self.graph.stream({"input": text, "output": "", "log": [], "reasoning_summary": ""}, config=config)


__all__ = ["CotBaseline"]
