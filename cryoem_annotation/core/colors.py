"""Color generation utilities for visualization."""

from typing import List
import numpy as np
from matplotlib.colors import hsv_to_rgb


def generate_label_colors(num_colors: int = 10) -> List[np.ndarray]:
    """
    Generate distinct colors for labels using HSV color space with golden ratio.

    Uses vectorized operations for better performance when generating
    many colors at once.

    Args:
        num_colors: Number of colors to generate (default 10 for labels 1-10)

    Returns:
        List of RGB color arrays (each with shape (3,))
    """
    if num_colors <= 0:
        return []

    # Generate all indices at once
    indices = np.arange(num_colors)

    # Vectorized HSV calculation
    hues = (indices * 0.618034) % 1.0  # Golden ratio for good distribution
    saturations = 0.7 + (indices % 3) * 0.1  # Vary saturation slightly
    values = np.full(num_colors, 0.9)

    # Stack into HSV array (shape: num_colors x 3)
    hsv = np.stack([hues, saturations, values], axis=1)

    # Convert all colors at once (vectorized)
    rgb = hsv_to_rgb(hsv)

    # Return as list for API compatibility
    return [rgb[i] for i in range(num_colors)]

