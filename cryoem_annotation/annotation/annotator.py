"""Main annotation workflow."""

from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from cryoem_annotation.core.sam_model import SAMModel
from cryoem_annotation.core.image_loader import load_micrograph_with_pixel_size, get_image_files
from cryoem_annotation.core.image_processing import normalize_image
from cryoem_annotation.core.colors import generate_label_colors
from cryoem_annotation.annotation.click_collector import RealTimeClickCollector, create_bounded_overlay
from cryoem_annotation.io.metadata import save_metadata, save_combined_results
from cryoem_annotation.io.masks import save_mask_binary


def annotate_micrographs(
    micrograph_folder: Path,
    checkpoint_path: Path,
    output_folder: Path,
    model_type: str = "vit_b",
    device: Optional[str] = None,
    pixel_size_nm: Optional[float] = None,
) -> None:
    """
    Annotate micrographs using SAM segmentation.

    Args:
        micrograph_folder: Path to folder containing micrographs
        checkpoint_path: Path to SAM checkpoint file
        output_folder: Path to output folder for results
        model_type: SAM model type ("vit_b", "vit_l", or "vit_h")
        device: Device to use ("cuda", "cpu", or None for auto-detect)
        pixel_size_nm: Pixel size in nm/pixel (overrides MRC header if provided)
    """
    print("=" * 60)
    print("Interactive Micrograph Annotation Tool - REAL-TIME VERSION")
    print("=" * 60)
    print(f"Micrograph folder: {micrograph_folder}")
    print(f"Model type: {model_type}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output folder: {output_folder}")
    print("=" * 60)
    
    # Load SAM model (can take 10-30 seconds)
    print(f"\nLoading SAM model: {model_type} from {checkpoint_path}")
    print("  (This may take a moment...)")
    sam_model = SAMModel(model_type, checkpoint_path, device)
    predictor = sam_model.get_predictor()
    print("  Model loaded successfully.")
    
    # Get all image files
    image_files = get_image_files(micrograph_folder)
    
    if len(image_files) == 0:
        print(f"No image files found in {micrograph_folder}")
        return
    
    print(f"Found {len(image_files)} micrograph(s)\n")
    print("=" * 60)
    
    # Create output folder
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Results storage
    all_results = []
    
    # Process each file with progress bar
    for idx, file_path in enumerate(tqdm(image_files, desc="Micrographs", unit="file")):
        print(f"\n[{idx+1}/{len(image_files)}] Processing: {file_path.name}")
        print("-" * 60)

        # Load micrograph with pixel size from MRC header
        micrograph, mrc_pixel_size = load_micrograph_with_pixel_size(file_path)
        if micrograph is None:
            print(f"  [ERROR] Failed to load {file_path.name}, skipping...")
            continue

        # Use CLI override if provided, otherwise use MRC header value
        effective_pixel_size = pixel_size_nm if pixel_size_nm is not None else mrc_pixel_size
        if effective_pixel_size is not None:
            print(f"  Pixel size: {effective_pixel_size:.4f} nm/pixel")
        
        # Normalize for display
        micrograph_display = normalize_image(micrograph)
        
        # Convert to RGB for SAM
        if len(micrograph_display.shape) == 2:
            micrograph_rgb = cv2.cvtColor(micrograph_display, cv2.COLOR_GRAY2RGB)
        else:
            micrograph_rgb = micrograph_display
        
        # Set image in predictor (only once per image)
        predictor.set_image(micrograph_rgb)
        
        # Collect clicks with real-time segmentation
        collector = RealTimeClickCollector(
            micrograph_display,
            micrograph_rgb,
            predictor,
            title=file_path.name
        )
        
        clicks, segmentations = collector.collect_clicks()
        
        if len(clicks) == 0:
            print("  No clicks recorded, skipping...")
            continue
        
        print(f"  [OK] Recorded {len(clicks)} segmentation(s)")
        
        # Create output directory for this image
        image_output_dir = output_folder / file_path.stem
        image_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process segmentations for saving
        all_masks = []
        seg_data_list = []
        
        for seg_data in segmentations:
            mask = seg_data['mask']
            all_masks.append(mask)
            
            # Prepare data for JSON (without the mask array)
            seg_json = {
                'click_index': seg_data['click_index'],
                'click_coords': seg_data['click_coords'],
                'mask_score': seg_data['mask_score'],
                'mask_area': seg_data['mask_area'],
            }
            seg_data_list.append(seg_json)
            
            # Save raw mask as binary
            click_idx = seg_data['click_index']
            mask_binary_filename = image_output_dir / f"mask_{click_idx:03d}_binary.png"
            save_mask_binary(mask, mask_binary_filename)
        
        # Save results for this image
        result = {
            'filename': file_path.name,
            'filepath': str(file_path),
            'image_shape': list(micrograph_rgb.shape[:2]),
            'num_clicks': len(clicks),
            'timestamp': datetime.now().isoformat(),
            'pixel_size_nm': effective_pixel_size,
            'segmentations': seg_data_list,
        }
        
        all_results.append(result)
        
        # Save metadata
        save_metadata(result, image_output_dir / "metadata.json")
        
        # Save visualization
        _save_overview_image(
            micrograph_display,
            clicks,
            all_masks,
            segmentations,
            file_path.name,
            image_output_dir / "overview.png"
        )
        
        print(f"  [OK] Saved results to {image_output_dir}")
        
        # Clear VRAM if using CUDA
        if device == "cuda" or (device is None and torch.cuda.is_available()):
            torch.cuda.empty_cache()
    
    # Save combined results
    combined_output = output_folder / "all_annotations.json"
    save_combined_results(all_results, combined_output)
    
    print("\n" + "=" * 60)
    print(f"[OK] Annotation complete!")
    print(f"  Processed {len(all_results)} micrograph(s)")
    print(f"  Total segmentations: {sum(r['num_clicks'] for r in all_results)}")
    print(f"  Results saved to: {output_folder}")
    print(f"  Combined results: {combined_output}")
    print("=" * 60)


def _save_overview_image(
    micrograph_display: np.ndarray,
    clicks: List,
    all_masks: List[np.ndarray],
    segmentations: List[Dict],
    filename: str,
    output_path: Path,
) -> None:
    """Save overview visualization image with clicks and segmentations combined."""
    fig, ax = plt.subplots(figsize=(12, 12))

    # Display base image
    ax.imshow(micrograph_display, cmap='gray')

    # Generate colors for segmentations
    colors = generate_label_colors(len(all_masks))

    # Overlay segmentation masks with click markers
    for i, (mask, seg_data) in enumerate(zip(all_masks, segmentations)):
        x, y = seg_data['click_coords']

        # Overlay mask using bounded approach (90%+ memory reduction)
        color = colors[i % len(colors)]
        overlay, extent = create_bounded_overlay(mask, [*color[:3], 0.4])
        if overlay is not None:
            ax.imshow(overlay, extent=extent)

        # Mark click point with red cross
        ax.plot(x, y, 'r+', markersize=15, markeredgewidth=2)

        # Add numbered label
        ax.text(x + 8, y - 8, f"{i+1}",
                color='white', fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor=color, edgecolor='white',
                         alpha=0.85, linewidth=1))

    ax.set_title(f"{filename}\n{len(segmentations)} segmentation(s)", fontsize=14)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

