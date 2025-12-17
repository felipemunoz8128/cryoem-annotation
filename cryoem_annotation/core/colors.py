"""Color generation utilities for visualization."""

from typing import List
import numpy as np
from matplotlib.colors import hsv_to_rgb


def generate_label_colors(num_colors: int = 10) -> List[np.ndarray]:
    """
    Generate distinct colors for labels using HSV color space with golden ratio.
    
    Args:
        num_colors: Number of colors to generate (default 10 for labels 1-10)
    
    Returns:
        List of RGB color arrays (each with shape (3,))
    """
    colors = []
    for i in range(num_colors):
        # Generate distinct colors using HSV color space
        hue = (i * 0.618034) % 1.0  # Golden ratio for good distribution
        saturation = 0.7 + (i % 3) * 0.1  # Vary saturation slightly
        value = 0.9
        rgb = hsv_to_rgb([[hue, saturation, value]])[0]
        colors.append(rgb)
    return colors

