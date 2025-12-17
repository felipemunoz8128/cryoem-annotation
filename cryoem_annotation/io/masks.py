"""Mask saving and loading utilities."""

from pathlib import Path
from typing import Optional
import numpy as np
import cv2


def save_mask_binary(mask: np.ndarray, output_path: Path) -> None:
    """
    Save binary mask as PNG image.
    
    Args:
        mask: Binary mask array (boolean or 0/1)
        output_path: Path to output PNG file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Convert to uint8 (0-255)
    mask_uint8 = (mask.astype(np.uint8) * 255)
    cv2.imwrite(str(output_path), mask_uint8)


def load_mask_binary(mask_path: Path) -> Optional[np.ndarray]:
    """
    Load binary mask from PNG file.
    
    Args:
        mask_path: Path to mask PNG file
    
    Returns:
        Binary mask as boolean array, or None if loading failed
    """
    if not mask_path.exists():
        return None
    
    try:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            return (mask > 127).astype(bool)
        return None
    except Exception as e:
        print(f"Error loading mask from {mask_path}: {e}")
        return None

