"""Metadata handling utilities."""

from pathlib import Path
from typing import Dict, List, Optional
import json
from datetime import datetime


def save_metadata(metadata: Dict, output_path: Path) -> None:
    """
    Save metadata to JSON file.
    
    Args:
        metadata: Metadata dictionary to save
        output_path: Path to output JSON file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def load_metadata(metadata_path: Path) -> Optional[Dict]:
    """
    Load metadata from JSON file.
    
    Args:
        metadata_path: Path to metadata JSON file
    
    Returns:
        Loaded metadata dictionary, or None if loading failed
    """
    if not metadata_path.exists():
        return None
    
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading metadata from {metadata_path}: {e}")
        return None


def save_combined_results(results: List[Dict], output_path: Path) -> None:
    """
    Save combined results from multiple micrographs.
    
    Args:
        results: List of result dictionaries
        output_path: Path to output JSON file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

