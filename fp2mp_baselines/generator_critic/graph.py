from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from ..graph_utils import build_llm_log_entry, message_content_to_text
from ..state import GeneratorCriticState
from .prompts import CRITIC_HUMAN_PROMPT, GENERATOR_DRAFT_HUMAN_PROMPT, GENERATOR_FINAL_HUMAN_PROMPT


GENERATOR_DRAFT_NODE = "generator_draft"
CRITIC_NODE = "critic"
GENERATOR_FINAL_NODE = "generator_final"


def build_generator_critic_graph(
    generator_llm: BaseChatModel,
    *,
    critic_llm: BaseChatModel | None = None,
) -> CompiledStateGraph:
    """Build a Generator-Critic LangGraph baseline."""

    critic_model = critic_llm or generator_llm

    def generator_draft_node(state: GeneratorCriticState) -> GeneratorCriticState:
        messages = [
            HumanMessage(content=GENERATOR_DRAFT_HUMAN_PROMPT.format(input=state["input"])),
        ]
        response = generator_llm.invoke(messages)
        draft = message_content_to_text(response.content)
        return {
            "input": state["input"],
            "output": state.get("output", ""),
            "draft": draft,
            "critique": state.get("critique", ""),
            "log": [
                *state.get("log", []),
                build_llm_log_entry(GENERATOR_DRAFT_NODE, messages, response),
            ],
        }

    def critic_node(state: GeneratorCriticState) -> GeneratorCriticState:
        messages = [
            HumanMessage(
                content=CRITIC_HUMAN_PROMPT.format(
                    input=state["input"],
                    draft=state["draft"],
                )
            ),
        ]
        response = critic_model.invoke(messages)
        critique = message_content_to_text(response.content)
        return {
            "input": state["input"],
            "output": state.get("output", ""),
            "draft": state["draft"],
            "critique": critique,
            "log": [
                *state.get("log", []),
                build_llm_log_entry(CRITIC_NODE, messages, response),
            ],
        }

    def generator_final_node(state: GeneratorCriticState) -> GeneratorCriticState:
        messages = [
            HumanMessage(
                content=GENERATOR_FINAL_HUMAN_PROMPT.format(
                    input=state["input"],
                    draft=state["draft"],
                    critique=state["critique"],
                )
            ),
        ]
        response = generator_llm.invoke(messages)
        return {
            "input": state["input"],
            "output": message_content_to_text(response.content),
            "draft": state["draft"],
            "critique": state["critique"],
            "log": [
                *state.get("log", []),
                build_llm_log_entry(GENERATOR_FINAL_NODE, messages, response),
            ],
        }

    graph = StateGraph(GeneratorCriticState)
    graph.add_node(GENERATOR_DRAFT_NODE, generator_draft_node)
    graph.add_node(CRITIC_NODE, critic_node)
    graph.add_node(GENERATOR_FINAL_NODE, generator_final_node)
    graph.add_edge(START, GENERATOR_DRAFT_NODE)
    graph.add_edge(GENERATOR_DRAFT_NODE, CRITIC_NODE)
    graph.add_edge(CRITIC_NODE, GENERATOR_FINAL_NODE)
    graph.add_edge(GENERATOR_FINAL_NODE, END)
    return graph.compile()


__all__ = ["build_generator_critic_graph"]
