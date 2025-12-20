"""
Cryo-EM Annotation Tool

Interactive annotation and analysis tool for cryo-electron microscopy micrographs
using Segment Anything Model (SAM).
"""

__version__ = "0.1.0"
__author__ = "Cryo-EM Annotation Tool Contributors"

__all__ = [
    "load_sam_model",
    "load_micrograph",
    "normalize_image",
    "generate_label_colors",
]


def __getattr__(name):
    """Lazy import for heavy modules to speed up CLI startup."""
    if name == "load_sam_model":
        from cryoem_annotation.core.sam_model import load_sam_model
        return load_sam_model
    elif name == "load_micrograph":
        from cryoem_annotation.core.image_loader import load_micrograph
        return load_micrograph
    elif name == "normalize_image":
        from cryoem_annotation.core.image_processing import normalize_image
        return normalize_image
    elif name == "generate_label_colors":
        from cryoem_annotation.core.colors import generate_label_colors
        return generate_label_colors
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

