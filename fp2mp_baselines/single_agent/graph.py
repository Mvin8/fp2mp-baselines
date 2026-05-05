from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from ..graph_utils import build_llm_log_entry, message_content_to_text
from ..state import TextToTextState


SINGLE_AGENT_NODE = "single_agent"


def build_single_agent_graph(
    llm: BaseChatModel,
) -> CompiledStateGraph:
    """Build a LangGraph StateGraph baseline with one LLM node.

    Input state:
        {"input": "<user task>", "output": "", "log": []}

    Output state:
        {"input": "<user task>", "output": "<model answer>", "log": [...]}
    """

    def single_agent_node(state: TextToTextState) -> TextToTextState:
        messages = [HumanMessage(content=state["input"])]
        response = llm.invoke(messages)
        return {
            "input": state["input"],
            "output": message_content_to_text(response.content),
            "log": [
                *state.get("log", []),
                build_llm_log_entry(SINGLE_AGENT_NODE, messages, response),
            ],
        }

    graph = StateGraph(TextToTextState)
    graph.add_node(SINGLE_AGENT_NODE, single_agent_node)
    graph.add_edge(START, SINGLE_AGENT_NODE)
    graph.add_edge(SINGLE_AGENT_NODE, END)
    return graph.compile()


__all__ = ["build_single_agent_graph"]
