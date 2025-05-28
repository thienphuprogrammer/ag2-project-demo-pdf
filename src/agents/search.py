"""Web search agent implementation for retrieving external knowledge."""

from typing import List, Dict, Any, Optional

from autogen import AssistantAgent

def create_search_agent(
    config_list: List[Dict[str, Any]], 
    # tools: Optional[List[Dict[str, Any]]] = None,
    name: str = "web_search_agent",
    human_input_mode: str = "NEVER"
) -> AssistantAgent:
    """Create a web search agent for retrieving external knowledge.
    
    This agent specializes in performing web searches to answer questions that require
    external or up-to-date information.
    
    Args:
        config_list: List of LLM configurations.
        tools: Optional list of tools available to the agent. Should include search capabilities.
        name: Name of the agent. Defaults to "web_search_agent".
        human_input_mode: Human input mode. Defaults to "NEVER".
        
    Returns:
        An AssistantAgent configured for web search.
    """
    return AssistantAgent(
        name=name,
        system_message="""You are a web search specialist focused on retrieving accurate and up-to-date information.
        When asked a question:
        1. Analyze what information is needed from external sources
        2. Perform web searches to find relevant, current information
        3. Synthesize the search results into a comprehensive answer
        4. Provide source attribution for the information you retrieve
        5. Be objective and present multiple perspectives when appropriate
        
        Focus on factual information and reliable sources. If search results are inconclusive,
        acknowledge limitations in your findings.""",
        # tools=tools or [],
        llm_config={"config_list": config_list},
        human_input_mode=human_input_mode
    )
