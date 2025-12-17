"""
Cryo-EM Annotation Tool

Interactive annotation and analysis tool for cryo-electron microscopy micrographs
using Segment Anything Model (SAM).
"""

__version__ = "0.1.0"
__author__ = "Cryo-EM Annotation Tool Contributors"

from cryoem_annotation.core.sam_model import load_sam_model
from cryoem_annotation.core.image_loader import load_micrograph
from cryoem_annotation.core.image_processing import normalize_image
from cryoem_annotation.core.colors import generate_label_colors

__all__ = [
    "load_sam_model",
    "load_micrograph",
    "normalize_image",
    "generate_label_colors",
]

