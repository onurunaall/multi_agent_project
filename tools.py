"""
tools.py
Contains all the tool functions for the multi-agent workflow.
"""

from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper


@tool("browse_duckduckgo")
def duckduckgo_search(query: str) -> str:
    """
    Searches the web using DuckDuckGo for the given query.
    Returns the search result as a string.
    """
    return DuckDuckGoSearchRun().run(query)


@tool("respond_final")
def respond_final(message: str) -> str:
    """
    This tool finalizes the answer to the user.
    The 'message' should be the final textual response.
    """
    return message


@tool("tool_wikipedia")
def wiki_lookup(keywords: str) -> str:
    """
    Searches for given keywords on Wikipedia and returns a summary.
    """
    return WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper()).run(keywords)
