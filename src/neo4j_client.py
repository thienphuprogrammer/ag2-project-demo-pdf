"""Neo4j client for graph-based RAG capabilities."""

import os
import logging
from typing import List, Dict, Any, Optional

from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

from dotenv import load_dotenv
load_dotenv()

from autogen.agentchat.contrib.graph_rag.document import Document, DocumentType
from autogen.agentchat.contrib.graph_rag.neo4j_graph_query_engine import Neo4jGraphQueryEngine

# Configure logging
logger = logging.getLogger(__name__)

# Neo4j configuration with environment variable support
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "2ViKyTzQ_kQO9njXaUvYyHaavDec8iqBQ9h8iv5NzYs")
NEO4J_HOST = os.getenv("NEO4J_HOST", "neo4j+s://02645b0d.databases.neo4j.io")
NEO4J_PORT = os.getenv("NEO4J_PORT", "7687")  # Default Neo4j port
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")
DEFAULT_EMBEDDING_MODEL = os.getenv("DEFAULT_EMBEDDING_MODEL", "text-embedding-3-small")

def init_query_engine(
    config_list: List[Dict[str, Any]] = None, 
    json_path: str = None,
) -> Neo4jGraphQueryEngine:
    """Initialize the Neo4j graph query engine for RAG.
    
    Args:
        config_list: Configuration list for the LLM.
        json_path: Path to the JSON file containing documents to be ingested.
        embedding_model: Optional embedding model name. Defaults to DEFAULT_EMBEDDING_MODEL.
        
    Returns:
        Initialized Neo4jGraphQueryEngine instance.
        
    Raises:
        RuntimeError: If there's an error connecting to Neo4j or initializing the database.
    """
    if not config_list:
        raise ValueError("LLM configuration is required")
        
    try:
        logger.info(f"Initializing Neo4j query engine with host: {NEO4J_HOST}")
        
        # Create the query engine with configuration
        # Extract configuration info
        if isinstance(config_list, list) and config_list:
            llm_config = config_list[0]
        else:
            llm_config = config_list
        
        # Create a simplified OpenAI configuration
        try:
            # If we have a list, use the first config
            if isinstance(config_list, list) and config_list:
                llm_config = config_list[0]
            else:
                llm_config = config_list  # Use as is
                
            # Ensure we have model information
            if not isinstance(llm_config, dict) or 'model' not in llm_config:
                raise ValueError("Invalid LLM configuration: must be a dictionary with 'model' key")
                
            model_name = llm_config.get('model')
            logger.info(f"Using model: {model_name}")
            
            # Set API key in environment if available
            if 'api_key' in llm_config and not os.environ.get("OPENAI_API_KEY"):
                os.environ["OPENAI_API_KEY"] = llm_config["api_key"]
                
            # Simple dictionary of configuration
            # Neo4j expects a simpler structure - we don't need the OpenAI class
            simple_config = {
                "model": model_name,
                "api_key": llm_config.get("api_key", os.environ.get("OPENAI_API_KEY")),
                "temperature": 0.1  # Lower temperature for more factual responses
            }
            
            # Add base_url if present
            if "base_url" in llm_config:
                simple_config["base_url"] = llm_config["base_url"]
                
            logger.info("Successfully created OpenAI configuration")
        except Exception as e:
            logger.error(f"Failed to create OpenAI configuration: {str(e)}")
            raise RuntimeError(f"LLM initialization error: {str(e)}") from e
        
        # Create the LLM instance
        llm = OpenAI(
            model_name=model_name,
            api_key=simple_config.get("api_key", os.environ.get("OPENAI_API_KEY")),
            temperature=simple_config.get("temperature", 0.1)
        )

        # Create the embedding instance
        embedding = OpenAIEmbedding(
            model="text-embedding-3-small",  # Use the specific model name directly
            api_key=simple_config.get("api_key", os.environ.get("OPENAI_API_KEY"))
        )

        # Create the Neo4j query engine
        engine = Neo4jGraphQueryEngine(
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD,
            host=NEO4J_HOST,
            port=int(NEO4J_PORT),  # Ensure port is an integer
            database=NEO4J_DATABASE,
            llm=llm,
            embedding=embedding
        )
        
        # Connect to the database
        try:
            engine.connect_db()
            logger.info("Successfully connected to Neo4j database")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j database: {str(e)}")
            raise RuntimeError(f"Neo4j connection error: {str(e)}") from e

        if not json_path:
            return engine

        # Initialize the database with the provided document
        try:
            # Load and validate the JSON document first
            import json
            with open(json_path, 'r', encoding='utf-8') as f:
                doc_content = json.load(f)
                
            # Validate that the document has valid content
            if not isinstance(doc_content, list) or not doc_content:
                raise ValueError(f"Document {json_path} is empty or not a valid JSON array")
                
            # Log the first few entries to help with debugging
            logger.info(f"First entry in document: {doc_content[0] if doc_content else 'Empty document'}")
            
            # Validate and transform entries to the expected format
            transformed_entries = []
            for entry in doc_content:
                if not isinstance(entry, dict):
                    continue
                    
                # Transform the entry to include required fields
                transformed = {
                    'text': entry.get('text', ''),  # Add empty text if not present
                    'metadata': {
                        'labels': ['Document'],  # Default label
                        'element_id': entry.get('element_id'),
                        'original_metadata': entry.get('metadata', {})
                    }
                }
                transformed_entries.append(transformed)
            
            if not transformed_entries:
                raise ValueError(f"No valid entries could be transformed in document {json_path}")
                
            # Save the transformed entries back to the file
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(transformed_entries, f, indent=2)
                
            logger.info(f"Transformed and saved {len(transformed_entries)} entries in {json_path}")
            
            # Create document instance with the JSON path
            doc = [Document(doctype=DocumentType.JSON, path_or_url=json_path)]
            
            # Initialize the database with the validated document
            engine.init_db(input_doc=doc)
            logger.info(f"Successfully initialized Neo4j database with document: {json_path}")
            
        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON in document {json_path}: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg) from e
        except FileNotFoundError as e:
            error_msg = f"Document not found: {json_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg) from e
        except Exception as e:
            logger.error(f"Failed to initialize Neo4j database: {str(e)}")
            raise RuntimeError(f"Neo4j database initialization error: {str(e)}") from e
            
        return engine
    except Exception as e:
        logger.error(f"Unexpected error initializing Neo4j query engine: {str(e)}")
        raise
