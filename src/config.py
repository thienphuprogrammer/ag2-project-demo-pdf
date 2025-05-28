"""Configuration management for the application."""

import os
import logging
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
from autogen import config_list_from_json

# Configure logging
logger = logging.getLogger(__name__)

# Load environment variables from .env file if present
load_dotenv()

# Configuration paths
CONFIG_PATH = os.getenv("CONFIG_PATH", "configs/oai_config_list.json")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-4o")

# Input/output paths
DEFAULT_PDF_PATH = os.getenv("DEFAULT_PDF_PATH", "input_files/supplementary.pdf")
DEFAULT_OUTPUT_JSON = os.getenv("DEFAULT_OUTPUT_JSON", "parsed/parsed_elements.json")
DEFAULT_IMAGE_DIR = os.getenv("DEFAULT_IMAGE_DIR", "parsed/pdf_images")

def load_config(model_filter: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load configuration from the config file with optional model filtering.
    
    Args:
        model_filter: Optional model name to filter by. Defaults to DEFAULT_MODEL.
        
    Returns:
        List of configuration dictionaries for AutoGen.
    """
    # Determine the filter to use
    if model_filter:
        filter_dict = {"model": [model_filter]}
    else:
        # If no specific filter is provided, load all models
        # This ensures we get at least one configuration
        filter_dict = None
    
    # Load config from the specified path
    try:
        config_list = config_list_from_json(CONFIG_PATH, filter_dict=filter_dict)
        
        # Check if we got any configurations
        if not config_list:
            logger.warning(f"No configurations found with filter {filter_dict}. Loading all configurations.")
            config_list = config_list_from_json(CONFIG_PATH)
            
            # If still no configurations, raise an error
            if not config_list:
                raise ValueError(f"No valid configurations found in {CONFIG_PATH}")
        
        # Set the API key to environment if not already set
        if not os.environ.get("OPENAI_API_KEY") and config_list and "api_key" in config_list[0]:
            os.environ["OPENAI_API_KEY"] = config_list[0]["api_key"]
            
        # Filter out incompatible parameters
        cleaned_configs = []
        for config in config_list:
            # Create a safe copy of the configuration
            clean_config = {}
            # Only include essential parameters
            if "model" in config:
                clean_config["model"] = config["model"]
            if "api_key" in config:
                clean_config["api_key"] = config["api_key"]
            if "base_url" in config:
                clean_config["base_url"] = config["base_url"]
            # Optional parameters that are sometimes supported
            if "temperature" in config:
                clean_config["temperature"] = config["temperature"]
            # Add tags if present (important for autogen)
            if "tags" in config:
                clean_config["tags"] = config["tags"]
            
            cleaned_configs.append(clean_config)
        
        # Log found models
        models = [cfg.get("model", "unknown") for cfg in cleaned_configs]
        logger.info(f"Loaded {len(cleaned_configs)} configurations for models: {', '.join(models)}")
            
        return cleaned_configs
    except Exception as e:
        raise RuntimeError(f"Error loading configuration from {CONFIG_PATH}: {e}") from e
