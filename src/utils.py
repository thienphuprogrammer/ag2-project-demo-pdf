"""Utility functions for the application."""

import os
import logging
from typing import List, Optional, Any
from pathlib import Path

from unstructured.partition.pdf import partition_pdf
from unstructured.staging.base import elements_to_json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def ensure_directory_exists(directory_path: str) -> None:
    """Ensure the specified directory exists.
    
    Args:
        directory_path: Path to the directory to create if it doesn't exist.
    """
    path = Path(directory_path)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory_path}")

def parse_pdf(
    file_path: str, 
    output_json_path: str, 
    image_output_dir: str,
    languages: List[str] = ["eng"],
    strategy: str = "hi_res"
) -> List[Any]:
    """Parse a PDF file and extract its content including tables and images.
    
    Args:
        file_path: Path to the PDF file.
        output_json_path: Path where the parsed JSON will be saved.
        image_output_dir: Directory where extracted images will be saved.
        languages: List of languages to process. Defaults to ["eng"].
        strategy: PDF parsing strategy. Defaults to "hi_res".
        
    Returns:
        List of extracted elements.
        
    Raises:
        FileNotFoundError: If the PDF file doesn't exist.
        RuntimeError: If PDF parsing fails.
    """
    import shutil
    
    # Check if file exists and is accessible
    if not os.path.isfile(file_path):
        error_msg = f"PDF file not found: {file_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    # Verify file is not empty
    if os.path.getsize(file_path) == 0:
        error_msg = f"PDF file is empty: {file_path}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    
    # Ensure output directories exist
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    os.makedirs(image_output_dir, exist_ok=True)
    
    # Try different strategies if the primary one fails
    strategies_to_try = [strategy, "fast", "ocr_only", "auto"]
    last_error = None
    
    for current_strategy in strategies_to_try:
        try:
            logger.info(f"Attempting to parse PDF with strategy: {current_strategy}")
            
            elements = partition_pdf(
                filename=file_path,
                strategy=current_strategy,
                languages=languages,
                extract_image_block_output_dir=image_output_dir,
                extract_image_block_types=["Image", "Table"],
                extract_forms=False,
                form_extraction_skip_tables=False,
            )
            
            # If we get here, parsing was successful
            elements_to_json(elements=elements, filename=output_json_path, encoding="utf-8")
            logger.info(
                f"Successfully parsed PDF with strategy '{current_strategy}'. "
                f"Extracted {len(elements)} elements to {output_json_path}"
            )
            
            return elements
            
        except Exception as e:
            last_error = e
            logger.warning(
                f"Failed to parse PDF with strategy '{current_strategy}': {str(e)}"
            )
            # If we're on the last strategy, don't continue
            if current_strategy == strategies_to_try[-1]:
                break
            # Otherwise, try the next strategy
            continue
    
    # If we get here, all strategies failed
    error_msg = (
        f"Failed to parse PDF after trying {len(strategies_to_try)} strategies. "
        f"Last error: {str(last_error)}"
    )
    logger.error(error_msg)
    raise RuntimeError(error_msg) from last_error
