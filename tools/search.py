"""
tools/search.py

A simple web search tool using DuckDuckGo (no API key needed).
Swap for Tavily, Brave, or SerpAPI in production.
"""

from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun

_search = DuckDuckGoSearchRun()


@tool
def search_tool(query: str) -> str:
    """
    Search the web for current information. Use for facts, news,
    documentation, or anything requiring up-to-date knowledge.
    Input should be a clear search query string.
    """
    try:
        return _search.run(query)
    except Exception as e:
        return f"Search failed: {e}"
