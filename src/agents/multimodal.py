"""Multimodal agent implementation for image and table processing."""

from typing import List, Dict, Any, Optional

from autogen.agentchat.contrib.multimodal_conversable_agent import MultimodalConversableAgent

def create_multimodal_agent(
    config_list: List[Dict[str, Any]],
    name: str = "image2table_convertor",
    human_input_mode: str = "NEVER",
    max_tokens: int = 300,
    max_consecutive_auto_reply: int = 1
) -> MultimodalConversableAgent:
    """Create a multimodal agent for processing images and converting tables.
    
    This agent specializes in processing image content, particularly extracting and
    converting tables from images into structured markdown format.
    
    Args:
        config_list: List of LLM configurations.
        name: Name of the agent. Defaults to "image2table_convertor".
        human_input_mode: Human input mode. Defaults to "NEVER".
        max_tokens: Maximum tokens for generation. Defaults to 300.
        max_consecutive_auto_reply: Maximum consecutive auto replies. Defaults to 1.
        
    Returns:
        A MultimodalConversableAgent configured for image processing.
    """
    return MultimodalConversableAgent(
        name=name,
        system_message="""You are a multimodal specialist focused on extracting and processing visual information.
        Your primary responsibilities are:
        1. Extract structured data from table images and convert to well-formatted markdown tables
        2. Maintain the precise structure and content of tables during conversion
        3. Process charts, graphs, and diagrams to extract key information
        4. Describe relevant visual elements accurately
        
        When processing tables:
        - Preserve all columns and rows completely
        - Maintain proper alignment and formatting
        - Handle merged cells appropriately
        - Include headers and any notes/footnotes
        
        Your goal is to make visual information accessible and structured in text format.""",
        llm_config={"config_list": config_list, "max_tokens": max_tokens},
        max_consecutive_auto_reply=max_consecutive_auto_reply,
        human_input_mode=human_input_mode
    )
