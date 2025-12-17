"""Image processing utilities for normalization and enhancement."""

from typing import Tuple
import numpy as np


def normalize_image(img: np.ndarray, percentile: Tuple[float, float] = (1, 99)) -> np.ndarray:
    """
    Normalize image to 0-255 uint8 using percentile-based scaling.
    
    More robust to outliers than min-max normalization.
    
    Args:
        img: Input image (any dtype)
        percentile: Percentiles for scaling (default: (1, 99))
    
    Returns:
        Normalized image (uint8, 0-255)
    """
    img_float = img.astype(np.float32)
    
    # Get percentile values (more robust to outliers)
    p_low, p_high = np.percentile(img_float, percentile)
    
    if p_high > p_low:
        # Scale from percentile range to [0, 1], then clip to handle outliers
        img_float = np.clip((img_float - p_low) / (p_high - p_low), 0, 1)
    else:
        img_float = np.zeros_like(img_float)
    
    return (img_float * 255).astype(np.uint8)

