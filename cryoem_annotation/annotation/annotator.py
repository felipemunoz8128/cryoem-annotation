"""Main annotation workflow."""

from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch

from cryoem_annotation.core.sam_model import SAMModel
from cryoem_annotation.core.image_loader import load_micrograph_with_pixel_size
from cryoem_annotation.core.image_processing import normalize_image
from cryoem_annotation.core.colors import generate_label_colors
from cryoem_annotation.core.grid_dataset import GridDataset, MicrographItem
from cryoem_annotation.core.project import get_all_completion_states, set_completion_state, find_project_file
from cryoem_annotation.annotation.click_collector import RealTimeClickCollector, create_bounded_overlay
from cryoem_annotation.io.metadata import load_metadata, save_metadata, save_combined_results
from cryoem_annotation.io.masks import load_mask_binary, save_mask_binary
from cryoem_annotation.navigation import NavigationWindow


def annotate_micrographs(
    dataset: GridDataset,
    checkpoint_path: Path,
    output_folder: Path,
    model_type: str = "vit_b",
    device: Optional[str] = None,
) -> None:
    """
    Annotate micrographs using SAM segmentation.

    Args:
        dataset: GridDataset containing micrographs to annotate
        checkpoint_path: Path to SAM checkpoint file
        output_folder: Path to output folder for results
        model_type: SAM model type ("vit_b", "vit_l", or "vit_h")
        device: Device to use ("cuda", "cpu", or None for auto-detect)
    """
    print("=" * 60)
    print("Interactive Micrograph Annotation Tool - REAL-TIME VERSION")
    print("=" * 60)
    print(f"Input: {dataset.root_path}")
    if dataset.is_multi_grid:
        print(f"Mode: Multi-grid ({len(dataset.grid_names)} grids)")
        for grid_name in dataset.grid_names:
            count = dataset.get_micrograph_count(grid_name)
            print(f"  {grid_name}: {count} files")
    else:
        print("Mode: Single folder")
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

    # Get all micrograph items
    items = dataset.get_micrographs()

    if len(items) == 0:
        print(f"No image files found in {dataset.root_path}")
        return

    print(f"Found {len(items)} micrograph(s)\n")
    print("=" * 60)

    # Create output folder
    output_folder.mkdir(parents=True, exist_ok=True)

    # Results storage
    all_results = []
    completed_indices = set()

    # Navigation state
    nav_state = {'index': 0, 'action': None, 'target': None}

    def on_navigate(action: str, target_index: Optional[int]) -> None:
        """Callback for navigation events."""
        nav_state['action'] = action
        nav_state['target'] = target_index

    # Create navigation window with grid-aware mode
    nav_window = NavigationWindow(
        items,
        on_navigate,
        title="Annotation",
        is_multi_grid=dataset.is_multi_grid,
    )

    # Find project file for tracking completion state
    project_file = find_project_file(output_folder)

    # Initialize completion states from project file and existing metadata
    pre_completed_count = 0
    if project_file:
        states = get_all_completion_states(project_file, "annotation")
    else:
        states = {}

    for idx, item in enumerate(items):
        key = (f"{item.grid_name}/{item.micrograph_name}"
               if item.grid_name is not None else item.micrograph_name)

        # Check project state first
        if states.get(key) == "completed":
            completed_indices.add(idx)
            nav_window.mark_completed(idx)
            pre_completed_count += 1
        else:
            # Backward compatibility: check for existing metadata.json
            existing_dir = (output_folder / item.grid_name / item.micrograph_name
                           if item.grid_name is not None
                           else output_folder / item.micrograph_name)
            metadata_file = existing_dir / "metadata.json"
            if metadata_file.exists():
                completed_indices.add(idx)
                nav_window.mark_completed(idx)
                pre_completed_count += 1

    if pre_completed_count > 0:
        print(f"  [OK] Found {pre_completed_count} previously completed micrograph(s)")

    # Current collector and image data (for saving on navigation)
    current_collector = None
    current_item: Optional[MicrographItem] = None
    current_micrograph_display = None
    current_micrograph_rgb = None
    current_pixel_size = None

    def _get_output_dir(item: MicrographItem) -> Path:
        """Get output directory for a micrograph item."""
        if item.grid_name is not None:
            # Multi-grid: output_folder/Grid1/micrograph_name/
            return output_folder / item.grid_name / item.micrograph_name
        else:
            # Single folder: output_folder/micrograph_name/
            return output_folder / item.micrograph_name

    def _get_micrograph_key(item: MicrographItem) -> str:
        """Get the key used for tracking completion state."""
        if item.grid_name is not None:
            return f"{item.grid_name}/{item.micrograph_name}"
        return item.micrograph_name

    def save_current_segmentations() -> None:
        """Save segmentations for the current file if any exist."""
        nonlocal current_collector, current_item, current_micrograph_display
        nonlocal current_micrograph_rgb, current_pixel_size

        if current_collector is None or current_item is None:
            return

        clicks = current_collector.clicks
        segmentations = current_collector.segmentations

        if len(clicks) == 0:
            return

        print(f"  [OK] Saving {len(clicks)} segmentation(s) for {current_item.display_name}")

        # Create output directory for this image
        image_output_dir = _get_output_dir(current_item)
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
            'filename': current_item.file_path.name,
            'filepath': str(current_item.file_path),
            'micrograph_name': current_item.micrograph_name,
            'grid_name': current_item.grid_name,
            'image_shape': list(current_micrograph_rgb.shape[:2]),
            'num_clicks': len(clicks),
            'timestamp': datetime.now().isoformat(),
            'pixel_size_nm': current_pixel_size,
            'segmentations': seg_data_list,
        }

        all_results.append(result)

        # Save metadata
        save_metadata(result, image_output_dir / "metadata.json")

        # Save visualization
        _save_overview_image(
            current_micrograph_display,
            clicks,
            all_masks,
            segmentations,
            current_item.display_name,
            image_output_dir / "overview.png"
        )

        print(f"  [OK] Saved results to {image_output_dir}")

        # Mark as completed in navigation
        completed_indices.add(nav_state['index'])
        nav_window.mark_completed(nav_state['index'])

    def load_file(idx: int) -> bool:
        """Load a file and set up the collector. Returns True if successful."""
        nonlocal current_collector, current_item, current_micrograph_display
        nonlocal current_micrograph_rgb, current_pixel_size

        item = items[idx]
        file_path = item.file_path

        # Show grid context in progress message
        if dataset.is_multi_grid:
            print(f"\n[{idx+1}/{len(items)}] Processing: {item.display_name}")
        else:
            print(f"\n[{idx+1}/{len(items)}] Processing: {file_path.name}")
        print("-" * 60)

        # Load micrograph with pixel size from MRC header
        micrograph, pixel_size = load_micrograph_with_pixel_size(file_path)
        if micrograph is None:
            print(f"  [ERROR] Failed to load {item.display_name}")
            return False

        if pixel_size is not None:
            print(f"  Pixel size (from MRC header): {pixel_size:.4f} nm/pixel")

        # Normalize for display
        micrograph_display = normalize_image(micrograph)

        # Convert to RGB for SAM
        if len(micrograph_display.shape) == 2:
            micrograph_rgb = cv2.cvtColor(micrograph_display, cv2.COLOR_GRAY2RGB)
        else:
            micrograph_rgb = micrograph_display

        # Set image in predictor (only once per image)
        predictor.set_image(micrograph_rgb)

        # Store current state
        current_item = item
        current_micrograph_display = micrograph_display
        current_micrograph_rgb = micrograph_rgb
        current_pixel_size = pixel_size

        # Use display_name for title (shows grid context)
        title = item.display_name

        # Create or update collector
        if current_collector is None:
            current_collector = RealTimeClickCollector(
                micrograph_display,
                micrograph_rgb,
                predictor,
                title=title,
                navigation_callback=on_navigate
            )
            if not current_collector.setup_figure():
                print("  [ERROR] Could not create figure window")
                return False
        else:
            # Update existing collector with new image
            current_collector.update_image(micrograph_display, micrograph_rgb, title)

        # Check for existing annotations and load them
        existing_dir = _get_output_dir(item)
        existing_metadata_file = existing_dir / "metadata.json"
        if existing_metadata_file.exists():
            existing_metadata = load_metadata(existing_metadata_file)
            if existing_metadata and 'segmentations' in existing_metadata:
                existing_segs = existing_metadata['segmentations']
                # Load masks for each segmentation
                segs_with_masks = []
                for seg in existing_segs:
                    click_idx = seg['click_index']
                    mask_file = existing_dir / f"mask_{click_idx:03d}_binary.png"
                    mask = load_mask_binary(mask_file)
                    if mask is not None:
                        seg_copy = dict(seg)
                        seg_copy['mask'] = mask
                        segs_with_masks.append(seg_copy)
                if segs_with_masks:
                    current_collector.load_existing_segmentations(segs_with_masks)
                    # Mark as already completed
                    completed_indices.add(idx)
                    nav_window.mark_completed(idx)

        print("\n  Figure window opened. Click on objects in the image.")
        print("  Instructions:")
        print("    - Left-click: Segment an object (mask appears immediately)")
        print("    - Arrow keys or Right-click: Navigate between files")
        print("    - Press 'd' or 'u': Undo last segmentation")
        print("    - Escape: Finish session")
        print()

        return True

    # Main event-driven loop
    try:
        # Load first file
        while nav_state['index'] < len(items):
            if not load_file(nav_state['index']):
                # Skip to next file if load failed
                nav_state['index'] += 1
                nav_window.set_current(nav_state['index'])
                continue
            break

        # Event loop - process until quit
        while nav_state['action'] != 'quit' and nav_state['index'] < len(items):
            # Process matplotlib and tkinter events
            plt.pause(0.05)
            nav_window.root.update()

            # Handle navigation action if any
            if nav_state['action'] is not None:
                action = nav_state['action']
                target = nav_state['target']
                nav_state['action'] = None
                nav_state['target'] = None

                if action == 'quit':
                    # Save current work before quitting
                    save_current_segmentations()
                    # Mark as done even if no segmentations
                    completed_indices.add(nav_state['index'])
                    nav_window.mark_completed(nav_state['index'])
                    # Update completion state in project file
                    if project_file and current_item:
                        set_completion_state(project_file, "annotation",
                                           _get_micrograph_key(current_item), "completed")
                    break

                # Save current segmentations before navigating
                save_current_segmentations()
                # Mark as done even if no segmentations
                completed_indices.add(nav_state['index'])
                nav_window.mark_completed(nav_state['index'])
                # Update completion state in project file
                if project_file and current_item:
                    set_completion_state(project_file, "annotation",
                                       _get_micrograph_key(current_item), "completed")

                # Clear current collector's segmentations for new file
                if current_collector:
                    current_collector.clear_segmentations()

                # Determine new index
                if action == 'next':
                    new_index = nav_state['index'] + 1
                elif action == 'prev':
                    new_index = max(0, nav_state['index'] - 1)
                elif action == 'goto':
                    new_index = target if target is not None else nav_state['index']
                else:
                    continue

                # Bounds check
                if new_index >= len(items):
                    print("\n  [OK] Reached end of file list.")
                    break

                nav_state['index'] = new_index
                nav_window.set_current(new_index)

                # Load new file
                if not load_file(new_index):
                    # Skip to next if failed
                    nav_state['action'] = 'next'
                    continue

                # Clear VRAM if using CUDA
                if device == "cuda" or (device is None and torch.cuda.is_available()):
                    torch.cuda.empty_cache()

    except KeyboardInterrupt:
        print("\n\n  [WARNING] Session interrupted by user")
        save_current_segmentations()
    finally:
        # Cleanup
        if current_collector:
            current_collector.close_figure()
        nav_window.destroy()

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
