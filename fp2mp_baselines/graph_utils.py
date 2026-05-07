from typing import Any

from langchain_core.messages import BaseMessage, message_to_dict


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


def build_message_log(messages: list[BaseMessage], response: Any) -> list[BaseMessage]:
    if isinstance(response, BaseMessage):
        return [*messages, response]
    return messages


__all__ = ["build_message_log", "message_content_to_text", "message_to_log_dict"]
