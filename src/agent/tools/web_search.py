# Web search tool

from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchResults

search = DuckDuckGoSearchResults()

@tool
def web_search(query: str) -> str:
    """
    This tool is used to perform web search based on the user's queries
    using DuckDuckGoSearchResults.
    """
    return search.run(query)