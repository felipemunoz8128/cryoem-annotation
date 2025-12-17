#!/usr/bin/env python3
"""
Example workflow: Annotate → Label → Extract

This script demonstrates how to use the cryoem-annotation package
programmatically to annotate micrographs, label segmentations, and extract results.
"""

from pathlib import Path
from cryoem_annotation.annotation.annotator import annotate_micrographs
from cryoem_annotation.labeling.labeler import label_segmentations
from cryoem_annotation.extraction.extractor import extract_results


def main():
    """Run complete annotation workflow."""
    
    # Configuration
    micrograph_folder = Path("data/micrographs")
    checkpoint_path = Path("sam_vit_b_01ec64.pth")
    output_folder = Path("results/annotations")
    
    print("=" * 60)
    print("Cryo-EM Annotation Workflow")
    print("=" * 60)
    
    # Step 1: Annotate micrographs
    print("\nStep 1: Annotating micrographs...")
    print("-" * 60)
    annotate_micrographs(
        micrograph_folder=micrograph_folder,
        checkpoint_path=checkpoint_path,
        output_folder=output_folder,
        model_type="vit_b",
    )
    
    # Step 2: Label segmentations
    print("\nStep 2: Labeling segmentations...")
    print("-" * 60)
    label_segmentations(
        results_folder=output_folder,
        micrograph_folder=micrograph_folder,
    )
    
    # Step 3: Extract results
    print("\nStep 3: Extracting results...")
    print("-" * 60)
    extract_results(
        results_folder=output_folder,
        output_path=output_folder / "final_results.csv",
        output_format="csv",
    )
    
    print("\n" + "=" * 60)
    print("Workflow complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

