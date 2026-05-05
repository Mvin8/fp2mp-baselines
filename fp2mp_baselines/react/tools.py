from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchResults


@tool
def ddgs_tool(query: str) -> str:
    """
    Поиск по интернету в DuckDuckGo.
    Если возвращает ошибку, попробуйте позже или воспользуйтесь другим инструментом.
    """
    try:
        search_tool = DuckDuckGoSearchResults(num_results=4)
        return search_tool.invoke(query)
    except Exception as error:
        return str(error)


__all__ = ["ddgs_tool"]
