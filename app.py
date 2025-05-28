"""
Nvidia Financial Analysis Multi-Agent Application

This application uses AutoGen to create a multi-agent system that analyzes
Nvidia's financial filings. It demonstrates the use of PDF parsing, Neo4j graph-based RAG,
web search, and multimodal processing in a coordinated agent system.
"""

import os
import logging
import argparse
import traceback
from typing import List, Optional, Dict, Any

from dotenv import load_dotenv

load_dotenv()

import nest_asyncio
from autogen import UserProxyAgent, GroupChat, GroupChatManager

from src.config import load_config, DEFAULT_PDF_PATH, DEFAULT_OUTPUT_JSON, DEFAULT_IMAGE_DIR
from src.utils import parse_pdf, ensure_directory_exists
from src.neo4j_client import init_query_engine
from src.agents.rag import create_rag_agent
from src.agents.general import create_general_agent
from src.agents.search import create_search_agent
from src.agents.multimodal import create_multimodal_agent
from autogen.agentchat.contrib.graph_rag.neo4j_graph_rag_capability import Neo4jGraphCapability

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def setup_parser() -> argparse.ArgumentParser:
    """Setup command-line argument parser.
    
    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(description="Nvidia Financial Analysis Multi-Agent System")
    parser.add_argument(
        "--pdf-path", 
        type=str, 
        default=DEFAULT_PDF_PATH,
        help=f"Path to the PDF file for analysis. Default: {DEFAULT_PDF_PATH}"
    )
    parser.add_argument(
        "--parse-pdf", 
        action="store_true",
        help="Whether to parse the PDF file. Only needs to be done once."
    )
    parser.add_argument(
        "--output-json", 
        type=str, 
        default=DEFAULT_OUTPUT_JSON,
        help=f"Path to store the parsed JSON. Default: {DEFAULT_OUTPUT_JSON}"
    )
    parser.add_argument(
        "--image-dir", 
        type=str, 
        default=DEFAULT_IMAGE_DIR,
        help=f"Directory to store extracted images. Default: {DEFAULT_IMAGE_DIR}"
    )
    parser.add_argument(
        "--initial-message", 
        type=str, 
        default="What are the key financial highlights in Nvidia's 2024 filing?",
        help="Initial message to start the chat with."
    )
    
    return parser


def setup_tools() -> List[Dict[str, Any]]:
    """Setup tools for the agents.
    
    In a more complete implementation, this would include actual tools
    such as web search functions, calculators, etc.
    
    Returns:
        List of tool configurations.
    """
    # This is a placeholder for actual tool definitions
    # In a real implementation, you would define tools here
    return []


def parse_document(pdf_path: str, output_json: str, image_dir: str, force_parse: bool = False, skip_if_error: bool = True) -> bool:
    """Parse the PDF document if needed.
    
    Args:
        pdf_path: Path to the PDF file.
        output_json: Path to store the parsed JSON.
        image_dir: Directory to store extracted images.
        force_parse: Whether to force parsing even if output exists.
        skip_if_error: Whether to continue execution if parsing fails.
        
    Returns:
        True if parsing succeeded or was skipped, False if it failed.
    """
    # Check if parsing is needed
    if not force_parse and os.path.exists(output_json):
        logger.info(f"Parsed JSON already exists at {output_json}. Skipping parsing." + 
                  " Use --parse-pdf to force re-parsing.")
        return True
    
    try:
        # Ensure directories exist
        ensure_directory_exists(os.path.dirname(output_json))
        ensure_directory_exists(image_dir)
        
        # Parse the PDF
        logger.info(f"Parsing PDF: {pdf_path}")
        parse_pdf(
            file_path=pdf_path,
            output_json_path=output_json,
            image_output_dir=image_dir
        )
        logger.info("PDF parsing completed successfully")
        return True
    except FileNotFoundError as e:
        logger.error(f"PDF file not found: {str(e)}")
        if not skip_if_error:
            raise
        return False
    except Exception as e:
        # Check for Tesseract error
        if "tesseract is not installed" in str(e).lower():
            logger.error("Tesseract OCR is not installed. This is required for PDF parsing.")
            logger.error("You can install it with: sudo apt-get install tesseract-ocr")
            logger.warning("Skipping PDF parsing. Using existing parsed data if available.")
            
            # Create an empty JSON file if needed to allow the application to continue
            if not os.path.exists(output_json):
                logger.warning(f"Creating empty parsed file at {output_json} to continue execution")
                with open(output_json, 'w') as f:
                    f.write('[]')
                    
            if not skip_if_error:
                raise
            return False
        else:
            logger.error(f"Error parsing PDF: {str(e)}")
            if not skip_if_error:
                raise
            return False


def setup_agents(config_list: List[Dict[str, Any]], query_engine) -> tuple:
    """Set up all agents for the group chat.
    
    Args:
        config_list: LLM configuration list.
        query_engine: Neo4j query engine for RAG.
        
    Returns:
        Tuple of (user_proxy, list of all agents)
    """
    # Get tools for agents
    tools = setup_tools()
    
    try:
        # Create the user proxy agent
        user_proxy = UserProxyAgent(
            name="user_proxy",
            system_message="User entry point for the multi-agent financial analysis system",
            human_input_mode="ALWAYS",
        )
        
        # Create specialized agents
        # rag_agent, graph_rag_capability = create_rag_agent(query_engine)
        graph_rag_capability = Neo4jGraphCapability(query_engine)
        # Add GraphRAG capability to user_proxy
        graph_rag_capability.add_to_agent(user_proxy)
        
        general = create_general_agent(config_list, tools)
        search = create_search_agent(config_list, tools)
        image2table = create_multimodal_agent(config_list)
        final_agent = create_general_agent(
            config_list, 
            tools, 
            name="final_summarizer",
            human_input_mode="NEVER"
        )
        
        # Return all agents
        return user_proxy, [
            user_proxy,
            general,
            search,
            rag_agent,
            image2table,
            final_agent,
        ]
    except Exception as e:
        logger.error(f"Error setting up agents: {str(e)}")
        logger.error(traceback.format_exc())
        raise


def setup_group_chat(agents: List, config_list: List[Dict[str, Any]]) -> GroupChatManager:
    """Set up the group chat with all agents.
    
    Args:
        agents: List of all agent instances.
        config_list: LLM configuration list.
        
    Returns:
        Configured GroupChatManager instance.
    """
    try:
        # Create the group chat with all agents
        groupchat = GroupChat(
            agents=agents,
            messages=[],
            speaker_selection_method="round_robin",
        )
        
        # Create the chat manager
        return GroupChatManager(groupchat=groupchat, llm_config={"config_list": config_list})
    except Exception as e:
        logger.error(f"Error setting up group chat: {str(e)}")
        raise


def setup_parser() -> argparse.ArgumentParser:
    """Setup command-line argument parser.
    
    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(description="Nvidia Financial Analysis Multi-Agent System")
    parser.add_argument(
        "--pdf-path", 
        type=str, 
        default=DEFAULT_PDF_PATH,
        help=f"Path to the PDF file for analysis. Default: {DEFAULT_PDF_PATH}"
    )
    parser.add_argument(
        "--parse-pdf", 
        action="store_true",
        help="Whether to parse the PDF file. Only needs to be done once."
    )
    parser.add_argument(
        "--skip-pdf-parsing", 
        action="store_true",
        help="Skip PDF parsing entirely, even if no parsed data exists."
    )
    parser.add_argument(
        "--output-json", 
        type=str, 
        default=DEFAULT_OUTPUT_JSON,
        help=f"Path to store the parsed JSON. Default: {DEFAULT_OUTPUT_JSON}"
    )
    parser.add_argument(
        "--image-dir", 
        type=str, 
        default=DEFAULT_IMAGE_DIR,
        help=f"Directory to store extracted images. Default: {DEFAULT_IMAGE_DIR}"
    )
    parser.add_argument(
        "--initial-message", 
        type=str, 
        default="What are the key financial highlights in Nvidia's 2024 filing?",
        help="Initial message to start the chat with."
    )
    parser.add_argument(
        "--create-empty-json", 
        action="store_true",
        help="Create an empty JSON file for testing without PDF parsing."
    )
    
    return parser


def main():
    """Main application entry point."""
    # Setup command-line argument parser
    parser = setup_parser()
    args = parser.parse_args()
    
    try:
        # Apply nest_asyncio to allow nested event loops
        nest_asyncio.apply()
        
        # Load configuration
        logger.info("Loading configuration")
        config_list = load_config()
        
        # Create empty JSON for testing if requested
        if args.create_empty_json and not os.path.exists(args.output_json):
            logger.info(f"Creating empty JSON file at {args.output_json} for testing")
            ensure_directory_exists(os.path.dirname(args.output_json))
            with open(args.output_json, 'w') as f:
                f.write('[]')
            
        # Parse document if needed and not explicitly skipped
        if not args.skip_pdf_parsing and not args.create_empty_json:
            logger.info("Parsing PDF")
            parsing_succeeded = parse_document(
                args.pdf_path, 
                args.output_json, 
                args.image_dir, 
                args.parse_pdf
            )
            
            if not parsing_succeeded and not os.path.exists(args.output_json):
                logger.error("PDF parsing failed and no existing parsed data found.")
                print("Error: PDF parsing failed. Please install Tesseract OCR or use --create-empty-json for testing.")
                print("You can install Tesseract OCR with: sudo apt-get install tesseract-ocr")
                return
        elif args.skip_pdf_parsing:
            logger.info("Skipping PDF parsing as requested")
            if not os.path.exists(args.output_json):
                logger.warning(f"No parsed data found at {args.output_json}. Creating empty file.")
                ensure_directory_exists(os.path.dirname(args.output_json))
                with open(args.output_json, 'w') as f:
                    f.write('[]')
        
        # Check if parsed data exists
        if not os.path.exists(args.output_json):
            logger.error(f"Parsed data not found at {args.output_json}")
            print(f"Error: No parsed data found at {args.output_json}. Use --create-empty-json for testing.")
            return
        
        # Initialize Neo4j query engine
        logger.info("Initializing Neo4j query engine")
        query_engine = init_query_engine(config_list, args.output_json)
        
        # Setup all agents
        logger.info("Setting up agents")
        user_proxy, all_agents = setup_agents(config_list, query_engine)
        
        # Setup group chat
        logger.info("Setting up group chat")
        manager = setup_group_chat(all_agents, config_list)
        
        # Start the conversation
        logger.info("Starting conversation")
        user_proxy.initiate_chat(manager, message=args.initial_message)
        
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        print(f"Error: {str(e)}")


# if __name__ == "__main__":
#     main()
