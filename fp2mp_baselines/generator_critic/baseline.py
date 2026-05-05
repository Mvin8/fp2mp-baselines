from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from ..state import GeneratorCriticState
from .graph import build_generator_critic_graph


class GeneratorCriticBaseline:
    """Wrapper around the compiled Generator-Critic LangGraph baseline."""

    def __init__(
        self,
        generator_llm: BaseChatModel,
        *,
        critic_llm: BaseChatModel | None = None,
    ) -> None:
        self.graph = build_generator_critic_graph(
            generator_llm=generator_llm,
            critic_llm=critic_llm,
        )

    def invoke(self, text: str, config: dict[str, Any] | None = None) -> str:
        result = self.invoke_state(text, config=config)
        return result["output"]

    def invoke_state(self, text: str, config: dict[str, Any] | None = None) -> GeneratorCriticState:
        return self.graph.invoke({"input": text, "output": "", "log": [], "draft": "", "critique": ""}, config=config)

    def stream(self, text: str, config: dict[str, Any] | None = None):
        return self.graph.stream({"input": text, "output": "", "log": [], "draft": "", "critique": ""}, config=config)


__all__ = ["GeneratorCriticBaseline"]
