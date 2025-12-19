"""Image loading utilities for micrographs."""

from pathlib import Path
from typing import List, Optional
import numpy as np
import cv2

# Try to import mrcfile for reading MRC files
try:
    import mrcfile
    MRC_AVAILABLE = True
except ImportError:
    MRC_AVAILABLE = False

# Supported image extensions
IMAGE_EXTENSIONS = {'.mrc', '.tif', '.tiff', '.png', '.jpg', '.jpeg'}


def load_micrograph(file_path: Path) -> Optional[np.ndarray]:
    """
    Load micrograph from MRC or image file.
    
    Args:
        file_path: Path to the micrograph file
    
    Returns:
        Loaded image as numpy array, or None if loading failed
    """
    if not file_path.exists():
        print(f"File not found: {file_path}")
        return None
    
    # Try MRC file first
    if file_path.suffix.lower() == '.mrc' and MRC_AVAILABLE:
        try:
            with mrcfile.open(file_path, mode='r') as mrc:
                data = mrc.data
                # Handle 3D volumes
                if len(data.shape) == 3:
                    data = data[0]  # Take first slice
                return data
        except Exception as e:
            print(f"Error reading MRC file: {e}")
            return None
    
    # Try regular image file
    try:
        img = cv2.imread(str(file_path))
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return img
    except Exception as e:
        print(f"Error reading image file: {e}")
        return None
    
    return None


def get_image_files(folder: Path, extensions: Optional[set] = None) -> List[Path]:
    """
    Get all image files from folder, excluding hidden files.

    Uses a single directory pass with case-insensitive extension matching
    for better performance than multiple glob calls.

    Args:
        folder: Directory to search for image files
        extensions: Set of file extensions to search for (default: IMAGE_EXTENSIONS)

    Returns:
        Sorted list of image file paths
    """
    if extensions is None:
        extensions = IMAGE_EXTENSIONS

    # Single pass with case-insensitive matching (more efficient than multiple globs)
    files = []
    for item in folder.iterdir():
        # Skip hidden files and directories
        if item.name.startswith('.'):
            continue
        # Check if it's a file with a matching extension
        if item.is_file() and item.suffix.lower() in extensions:
            files.append(item)

    return sorted(files)

