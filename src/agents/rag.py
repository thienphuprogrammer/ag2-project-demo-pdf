"""RAG agent implementation for document retrieval and answering."""

from typing import List, Dict, Any

from autogen import ConversableAgent

def create_rag_agent(
    config_list: List[Dict[str, Any]],
    name: str = "rag_agent",
    human_input_mode: str = "NEVER",
    max_consecutive_auto_reply: int = 5
) -> ConversableAgent:
    """Create a RAG (Retrieval-Augmented Generation) agent with Neo4j graph capabilities.
    
    This agent specializes in retrieving information from documents stored in Neo4j
    and answering document-specific questions.
    
    Args:
        name: Name of the agent. Defaults to "rag_agent".
        human_input_mode: Human input mode. Defaults to "NEVER".
        
    Returns:
        A ConversableAgent with Neo4j graph RAG capabilities.
    """
    # Create the base agent
    agent = ConversableAgent(
        name=name,
        human_input_mode=human_input_mode,
        system_message="""You are a document retrieval expert specializing in extracting relevant information from documents.
        When asked a question about document content:
        1. Query the Neo4j knowledge graph to find relevant information
        2. Provide accurate answers based strictly on information from the document
        3. If the answer is not in the document, clearly state so rather than making up information
        4. Always cite the specific sections of the document you're referencing
        
        Keep your responses concise and focused on the document content.""",
        max_consecutive_auto_reply=max_consecutive_auto_reply,
        llm_config={"config_list": config_list}
    )
    
    return agent