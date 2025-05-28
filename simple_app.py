"""Simplified Multi-Agent Application without Neo4j Integration

This is a simplified version of the application that doesn't require Neo4j.
It demonstrates the core multi-agent functionality without the RAG component.
"""

import os
import logging
import argparse
from typing import List, Dict, Any

import nest_asyncio
from autogen import UserProxyAgent, AssistantAgent, GroupChat, GroupChatManager

from src.config import load_config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_parser() -> argparse.ArgumentParser:
    """Setup command-line argument parser.
    
    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(description="Simplified Multi-Agent System")
    parser.add_argument(
        "--initial-message", 
        type=str, 
        default="Hello! How can you help me today?",
        help="Initial message to start the chat with."
    )
    parser.add_argument(
        "--auto-mode", 
        action="store_true",
        help="Run in automatic mode without requiring user input"
    )
    parser.add_argument(
        "--max-rounds", 
        type=int, 
        default=5,
        help="Maximum number of conversation rounds"
    )
    return parser

def create_agents(config_list: List[Dict[str, Any]], auto_mode: bool = False):
    """Create the agents for the group chat.
    
    Args:
        config_list: LLM configuration list.
        auto_mode: Whether to run in automatic mode without user input.
        
    Returns:
        Tuple of (user_proxy, list of all agents)
    """
    # Set human input mode based on auto_mode
    human_input_mode = "NEVER" if auto_mode else "ALWAYS"
    
    # Create the user proxy agent
    user_proxy = UserProxyAgent(
        name="user_proxy",
        system_message="""User entry point for the multi-agent system. 
        You coordinate between the user and the specialized agents.""",
        human_input_mode=human_input_mode,
        max_consecutive_auto_reply=5 if auto_mode else 0,
    )
    
    # Create assistant agents with distinct roles
    general = AssistantAgent(
        name="general_assistant",
        system_message="""You are a helpful assistant that provides general information and coordinates with other agents.
        You can help with a wide range of tasks and questions. Be concise and to the point.""",
        llm_config={"config_list": config_list}
    )
    
    research = AssistantAgent(
        name="research_specialist",
        system_message="""You are a research specialist who helps find information and provide detailed analysis.
        You excel at breaking down complex topics and providing thorough explanations.
        When asked a question, provide well-researched and accurate information.""",
        llm_config={"config_list": config_list}
    )
    
    creative = AssistantAgent(
        name="creative_agent",
        system_message="""You are a creative agent who helps with brainstorming, content creation, and thinking outside the box.
        You provide innovative ideas and approaches to problems.
        Be imaginative but practical in your suggestions.""",
        llm_config={"config_list": config_list}
    )
    
    # Return all agents
    return user_proxy, [user_proxy, general, research, creative]

def run_conversation():
    """Run a test conversation with the agents."""
    try:
        # Apply nest_asyncio to allow nested event loops
        nest_asyncio.apply()
        
        # Load configuration
        logger.info("Loading configuration")
        config_list = load_config()
        
        # Create agents
        logger.info("Creating agents")
        user_proxy, agents = create_agents(config_list, auto_mode=True)
        
        # Create group chat
        logger.info("Setting up group chat")
        group_chat = GroupChat(
            agents=agents,
            messages=[],
            max_round=3  # Limit to 3 rounds
        )
        
        # Create manager
        manager = GroupChatManager(
            groupchat=group_chat,
            llm_config={"config_list": config_list}
        )
        
        # Test conversation
        test_messages = [
            "What can you tell me about NVIDIA's latest financial performance?",
            "What are some innovative AI applications?",
            "Thank you, that's all for now."
        ]
        
        # Start the conversation
        logger.info("Starting test conversation")
        user_proxy.initiate_chat(
            manager,
            message=test_messages[0],
        )
        
        # Continue with the rest of the test messages
        for msg in test_messages[1:]:
            user_proxy.send(msg, manager)
        
        return 0
        
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        print(f"Error: {str(e)}")
        return 1

def main():
    """Main application entry point."""
    # Setup command-line argument parser
    parser = setup_parser()
    args = parser.parse_args()
    
    if args.auto_mode:
        # Run the test conversation in auto-mode
        return run_conversation()
    else:
        try:
            # Apply nest_asyncio to allow nested event loops
            nest_asyncio.apply()
            
            # Load configuration
            logger.info("Loading configuration")
            config_list = load_config()
            
            # Create agents
            logger.info("Creating agents")
            user_proxy, agents = create_agents(config_list, args.auto_mode)
            
            # Create group chat
            logger.info("Setting up group chat")
            group_chat = GroupChat(
                agents=agents,
                messages=[],
                max_round=args.max_rounds
            )
            
            # Create manager
            manager = GroupChatManager(
                groupchat=group_chat,
                llm_config={"config_list": config_list}
            )
            
            # Start the conversation
            logger.info("Starting conversation")
            user_proxy.initiate_chat(
                manager,
                message=args.initial_message,
            )
            
        except Exception as e:
            logger.error(f"Application error: {str(e)}")
            print(f"Error: {str(e)}")
            return 1
    
    return 0

if __name__ == "__main__":
    main()
