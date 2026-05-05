from typing import Any

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from ..graph_utils import message_content_to_text, message_to_log_dict
from ..state import AgentLogEntry, ReactState
from .tools import ddgs_tool


REACT_NODE = "react_agent"


def _extract_final_output(messages: list[Any]) -> str:
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return message_content_to_text(message.content)
    if messages:
        return message_content_to_text(getattr(messages[-1], "content", messages[-1]))
    return ""


def _build_react_log_entry(messages: list[Any]) -> AgentLogEntry:
    tool_calls: list[dict[str, Any]] = []
    for message in messages:
        message_tool_calls = getattr(message, "tool_calls", None)
        if message_tool_calls:
            tool_calls.extend(message_tool_calls)

    entry: AgentLogEntry = {
        "agent": REACT_NODE,
        "event": "react_trace",
        "messages": [message_to_log_dict(message) for message in messages],
    }
    if tool_calls:
        entry["tool_calls"] = tool_calls
    return entry


def build_react_graph(
    llm: BaseChatModel,
    *,
    tools: list[BaseTool] | None = None,
) -> CompiledStateGraph:
    """Build a ReAct LangGraph baseline with DuckDuckGo search by default."""

    react_agent = create_agent(
        model=llm,
        tools=tools or [ddgs_tool],
    )

    def react_node(state: ReactState) -> ReactState:
        result = react_agent.invoke({"messages": [HumanMessage(content=state["input"])]})
        messages = result["messages"]
        return {
            "input": state["input"],
            "output": _extract_final_output(messages),
            "messages": [message_to_log_dict(message) for message in messages],
            "log": [
                *state.get("log", []),
                _build_react_log_entry(messages),
            ],
        }

    graph = StateGraph(ReactState)
    graph.add_node(REACT_NODE, react_node)
    graph.add_edge(START, REACT_NODE)
    graph.add_edge(REACT_NODE, END)
    return graph.compile()


__all__ = ["build_react_graph"]
