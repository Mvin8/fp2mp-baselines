from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from ..graph_utils import build_llm_log_entry, message_content_to_text
from ..state import CotState
from .prompts import COT_FINAL_HUMAN_PROMPT, COT_REASONING_HUMAN_PROMPT


COT_REASONING_NODE = "cot_reasoning"
COT_FINAL_NODE = "cot_final_answer"


def build_cot_graph(
    llm: BaseChatModel,
) -> CompiledStateGraph:
    """Build a two-step CoT-style LangGraph baseline.

    The graph stores reasoning in state, then uses it to produce the final text
    output. Agent messages and provider metadata are written to state["log"].
    """

    def reasoning_node(state: CotState) -> CotState:
        messages = [
            HumanMessage(
                content=COT_REASONING_HUMAN_PROMPT.format(input=state["input"])
            ),
        ]
        response = llm.invoke(messages)
        reasoning_summary = message_content_to_text(response.content)
        return {
            "input": state["input"],
            "output": state.get("output", ""),
            "reasoning_summary": reasoning_summary,
            "log": [
                *state.get("log", []),
                build_llm_log_entry(COT_REASONING_NODE, messages, response),
            ],
        }

    def final_answer_node(state: CotState) -> CotState:
        messages = [
            HumanMessage(
                content=COT_FINAL_HUMAN_PROMPT.format(
                    input=state["input"],
                    reasoning_summary=state["reasoning_summary"],
                )
            ),
        ]
        response = llm.invoke(messages)
        return {
            "input": state["input"],
            "output": message_content_to_text(response.content),
            "reasoning_summary": state["reasoning_summary"],
            "log": [
                *state.get("log", []),
                build_llm_log_entry(COT_FINAL_NODE, messages, response),
            ],
        }

    graph = StateGraph(CotState)
    graph.add_node(COT_REASONING_NODE, reasoning_node)
    graph.add_node(COT_FINAL_NODE, final_answer_node)
    graph.add_edge(START, COT_REASONING_NODE)
    graph.add_edge(COT_REASONING_NODE, COT_FINAL_NODE)
    graph.add_edge(COT_FINAL_NODE, END)
    return graph.compile()


__all__ = ["build_cot_graph"]
