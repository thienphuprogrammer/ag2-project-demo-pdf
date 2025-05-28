"""General purpose agent implementation."""

from typing import List, Dict, Any, Optional

from autogen import ConversableAgent

def create_general_agent(
    config_list: List[Dict[str, Any]], 
    name: str = "general_assistant",
    human_input_mode: str = "NEVER",
    system_message: Optional[str] = None
) -> ConversableAgent:
    """Create a general purpose assistant agent.
    
    This agent is designed to handle general questions and delegate to specialized agents when necessary.
    
    Args:
        config_list: List of LLM configurations.
        name: Name of the agent. Defaults to "general_assistant".
        human_input_mode: Human input mode. Defaults to "NEVER".
        system_message: System message for the agent. Defaults to None.
        
    Returns:
        A configured ConversableAgent instance.
    """
    return ConversableAgent(
        name=name,
        llm_config={"config_list": config_list},
        system_message=system_message or """You handle general questions and tasks. 
        Your primary role is to:
        1. Answer direct questions within your knowledge
        2. Delegate to specialized agents when appropriate:
           - Search agent for web information
           - RAG agent for document-specific queries
           - Multimodal agent for image/table processing
        3. Synthesize information from other agents into a coherent response
        
        Be helpful, concise, and accurate. If you don't know, say so.""",
        human_input_mode=human_input_mode
    )
