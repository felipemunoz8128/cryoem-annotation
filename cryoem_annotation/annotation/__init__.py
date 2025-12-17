"""Annotation tools for interactive segmentation."""

from cryoem_annotation.annotation.click_collector import RealTimeClickCollector
from cryoem_annotation.annotation.annotator import annotate_micrographs

__all__ = [
    "RealTimeClickCollector",
    "annotate_micrographs",
]

