from typing import Any

from langchain_core.messages import BaseMessage, message_to_dict

from .state import AgentLogEntry


def message_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(parts)

    return str(content)


def message_to_log_dict(message: BaseMessage) -> dict[str, Any]:
    data = message_to_dict(message)
    data["content"] = message.content
    return data


def build_llm_log_entry(agent: str, messages: list[BaseMessage], response: BaseMessage) -> AgentLogEntry:
    entry: AgentLogEntry = {
        "agent": agent,
        "event": "llm_call",
        "messages": [message_to_log_dict(message) for message in messages],
        "response": message_to_log_dict(response),
    }

    usage_metadata = getattr(response, "usage_metadata", None)
    if usage_metadata:
        entry["usage_metadata"] = usage_metadata

    response_metadata = getattr(response, "response_metadata", None)
    if response_metadata:
        entry["response_metadata"] = response_metadata

    tool_calls = getattr(response, "tool_calls", None)
    if tool_calls:
        entry["tool_calls"] = tool_calls

    invalid_tool_calls = getattr(response, "invalid_tool_calls", None)
    if invalid_tool_calls:
        entry["invalid_tool_calls"] = invalid_tool_calls

    return entry


__all__ = ["build_llm_log_entry", "message_content_to_text", "message_to_log_dict"]
