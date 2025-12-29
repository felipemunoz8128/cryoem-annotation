"""Extract labels and areas from annotation results."""

from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json
import csv
import math
from tqdm import tqdm

from cryoem_annotation.io.metadata import load_metadata


def _detect_multi_grid_structure(results_folder: Path) -> bool:
    """
    Detect if results folder has multi-grid structure.

    Multi-grid structure: results/{Grid}/{micrograph}/metadata.json
    Single-folder structure: results/{micrograph}/metadata.json

    Returns:
        True if multi-grid structure detected, False otherwise.
    """
    for item in results_folder.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            # Check if this subdir contains further subdirs with metadata.json
            nested_metadata = list(item.glob("*/metadata.json"))
            if nested_metadata:
                return True
    return False


def extract_segmentation_data(
    results_folder: Path,
    pixel_size_override: Optional[float] = None
) -> Tuple[List[Dict], List[Dict], int]:
    """
    Extract labels and areas from all metadata.json files.

    Args:
        results_folder: Path to annotation results folder
        pixel_size_override: Override pixel size in nm/pixel for all micrographs

    Returns:
        Tuple of (metadata_list, results_list, total_micrographs)
    """
    metadata_list = []
    results_list = []
    seg_counter = 0  # Global segmentation counter

    # Detect multi-grid vs single-folder structure
    is_multi_grid = _detect_multi_grid_structure(results_folder)

    if is_multi_grid:
        metadata_files = list(results_folder.glob("*/*/metadata.json"))
    else:
        metadata_files = list(results_folder.glob("*/metadata.json"))

    total_micrographs = len(metadata_files)

    if total_micrographs == 0:
        print(f"No metadata.json files found in {results_folder}")
        return metadata_list, results_list, 0

    mode_str = "multi-grid" if is_multi_grid else "single-folder"
    print(f"\nDetected {mode_str} structure")
    print(f"Found {len(metadata_files)} metadata file(s)\n")

    for metadata_file in tqdm(sorted(metadata_files), desc="Extracting", unit="file"):
        micrograph_name = metadata_file.parent.name

        # Extract grid name from path hierarchy for multi-grid structure
        if is_multi_grid:
            grid_name = metadata_file.parent.parent.name
        else:
            grid_name = None

        display_name = f"{grid_name}/{micrograph_name}" if grid_name else micrograph_name
        print(f"Processing: {display_name}")

        metadata = load_metadata(metadata_file)
        if metadata is None:
            continue

        segmentations = metadata.get('segmentations', [])

        if len(segmentations) == 0:
            print(f"  [WARNING] No segmentations found")
            continue

        # Get pixel size: CLI override > metadata value > None
        pixel_size_nm = pixel_size_override
        if pixel_size_nm is None:
            pixel_size_nm = metadata.get('pixel_size_nm')

        # Extract data for each segmentation
        for seg in segmentations:
            seg_counter += 1
            seg_id = seg_counter
            click_index = seg.get('click_index')

            # Get area and calculate diameter
            area_pixels = seg.get('mask_area')
            diameter_nm = None
            if area_pixels is not None and pixel_size_nm is not None:
                # Convert area to equivalent circle diameter
                # Area_nm2 = area_pixels * (pixel_size_nm)^2
                # Diameter = 2 * sqrt(Area / pi) = sqrt(4 * Area / pi)
                area_nm2 = area_pixels * (pixel_size_nm ** 2)
                diameter_nm = math.sqrt(4 * area_nm2 / math.pi)

            # Metadata entry
            metadata_entry = {
                'segmentation_id': seg_id,
                'grid_name': grid_name,
                'micrograph_name': micrograph_name,
                'click_index': click_index,
                'click_coords': seg.get('click_coords'),
                'mask_score': seg.get('mask_score'),
                'area_pixels': area_pixels,
            }
            metadata_list.append(metadata_entry)

            # Results entry
            results_entry = {
                'segmentation_id': seg_id,
                'label': seg.get('label'),
                'diameter_nm': diameter_nm,
            }
            results_list.append(results_entry)

        labeled_count = sum(1 for s in segmentations if s.get('label') is not None)
        print(f"  [OK] Extracted {len(segmentations)} segmentation(s) ({labeled_count} labeled)")

    return metadata_list, results_list, total_micrographs


def save_metadata_csv(data: List[Dict], output_file: Path) -> None:
    """Save metadata entries to CSV file."""
    if len(data) == 0:
        print("\nNo metadata to save.")
        return

    fieldnames = ['segmentation_id', 'grid_name', 'micrograph_name', 'click_index',
                  'click_coords', 'mask_score', 'area_pixels']

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for entry in data:
            row = entry.copy()
            if row['click_coords'] is not None:
                row['click_coords'] = f"[{row['click_coords'][0]}, {row['click_coords'][1]}]"
            writer.writerow(row)

    print(f"[OK] Saved {len(data)} entries to {output_file}")


def save_results_csv(data: List[Dict], output_file: Path) -> None:
    """Save results entries to CSV file."""
    if len(data) == 0:
        print("\nNo results to save.")
        return

    fieldnames = ['segmentation_id', 'label', 'diameter_nm']

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for entry in data:
            row = entry.copy()
            # Format diameter_nm with reasonable precision
            if row['diameter_nm'] is not None:
                row['diameter_nm'] = f"{row['diameter_nm']:.2f}"
            writer.writerow(row)

    print(f"[OK] Saved {len(data)} entries to {output_file}")


def save_to_json(metadata: List[Dict], results: List[Dict], output_file: Path) -> None:
    """Save combined data to JSON file."""
    if len(metadata) == 0:
        print("\nNo data to save.")
        return

    # Combine metadata and results by segmentation_id
    combined = []
    results_by_id = {r['segmentation_id']: r for r in results}

    for meta in metadata:
        seg_id = meta['segmentation_id']
        entry = {**meta, **results_by_id.get(seg_id, {})}
        combined.append(entry)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)

    print(f"[OK] Saved {len(combined)} entries to {output_file}")


def print_summary(metadata: List[Dict], results: List[Dict], total_micrographs: int) -> None:
    """Print summary statistics."""
    if len(results) == 0:
        return

    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)

    total_segmentations = len(results)
    labeled_segmentations = sum(1 for d in results if d['label'] is not None)
    unlabeled_segmentations = total_segmentations - labeled_segmentations

    print(f"Total segmentations: {total_segmentations}")
    print(f"  Labeled: {labeled_segmentations}")
    print(f"  Unlabeled: {unlabeled_segmentations}")

    if labeled_segmentations > 0:
        label_counts = {}
        for d in results:
            if d['label'] is not None:
                label = d['label']
                label_counts[label] = label_counts.get(label, 0) + 1

        print(f"\nLabel distribution:")
        # Sort labels: strings alphabetically, integers numerically
        str_labels = sorted([l for l in label_counts.keys() if isinstance(l, str)])
        int_labels = sorted([l for l in label_counts.keys() if isinstance(l, int)])

        for label in str_labels:
            print(f"  {label}: {label_counts[label]} object(s)")
        for label in int_labels:
            print(f"  Label {label} (legacy): {label_counts[label]} object(s)")

    # Area statistics in pixels (from metadata)
    areas_pixels = [d['area_pixels'] for d in metadata if d['area_pixels'] is not None]
    if areas_pixels:
        print(f"\nMask area statistics (pixels):")
        print(f"  Total objects: {len(areas_pixels)}")
        print(f"  Mean area: {sum(areas_pixels) / len(areas_pixels):.1f}")
        print(f"  Median area: {sorted(areas_pixels)[len(areas_pixels)//2]:.1f}")
        print(f"  Min area: {min(areas_pixels)}")
        print(f"  Max area: {max(areas_pixels)}")

    # Diameter statistics in nm (if available)
    diameters_nm = [d['diameter_nm'] for d in results if d['diameter_nm'] is not None]
    if diameters_nm:
        print(f"\nEquivalent diameter statistics (nm):")
        print(f"  Total objects: {len(diameters_nm)}")
        print(f"  Mean diameter: {sum(diameters_nm) / len(diameters_nm):.2f}")
        print(f"  Median diameter: {sorted(diameters_nm)[len(diameters_nm)//2]:.2f}")
        print(f"  Min diameter: {min(diameters_nm):.2f}")
        print(f"  Max diameter: {max(diameters_nm):.2f}")

    # Micrograph counts
    micrographs_with_segs = len(set(d['micrograph_name'] for d in metadata))
    micrographs_without_segs = total_micrographs - micrographs_with_segs
    print(f"\nMicrographs processed: {total_micrographs}")
    print(f"  With segmentations: {micrographs_with_segs}")
    print(f"  Without segmentations: {micrographs_without_segs}")
    if total_micrographs > 0:
        print(f"  Average segmentations per micrograph: {total_segmentations / total_micrographs:.1f}")

    # Per-grid breakdown (only if multi-grid data exists)
    grids = set(d['grid_name'] for d in metadata if d['grid_name'] is not None)
    if grids:
        # Build mapping from segmentation_id to results for label info
        results_by_id = {r['segmentation_id']: r for r in results}

        print(f"\nPer-grid breakdown:")
        for grid in sorted(grids):
            grid_metadata = [d for d in metadata if d['grid_name'] == grid]
            grid_micrographs = len(set(d['micrograph_name'] for d in grid_metadata))
            grid_segmentations = len(grid_metadata)
            grid_labeled = sum(
                1 for d in grid_metadata
                if results_by_id.get(d['segmentation_id'], {}).get('label') is not None
            )
            grid_unlabeled = grid_segmentations - grid_labeled
            print(f"  {grid}:")
            print(f"    Micrographs: {grid_micrographs}")
            print(f"    Segmentations: {grid_segmentations} ({grid_labeled} labeled, {grid_unlabeled} unlabeled)")

    print("=" * 60)


def extract_results(
    results_folder: Path,
    micrograph_folder: Optional[Path] = None,
    output_path: Optional[Path] = None,
    output_format: str = "csv",
    pixel_size_override: Optional[float] = None,
) -> None:
    """
    Extract results from annotation folder.

    Args:
        results_folder: Path to annotation results folder
        micrograph_folder: Path to micrograph folder (for accurate total count)
        output_path: Base path for output files (default: results in results_folder)
        output_format: Output format: "csv", "json", or "both" (default: "csv")
        pixel_size_override: Override pixel size in nm/pixel for all micrographs
    """
    from cryoem_annotation.core.image_loader import get_image_files

    print("=" * 60)
    print("Extract Labels and Areas from Segmentations")
    print("=" * 60)
    print(f"Annotation results folder: {results_folder}")
    if micrograph_folder is not None:
        print(f"Micrograph folder: {micrograph_folder}")
    if pixel_size_override is not None:
        print(f"Pixel size override: {pixel_size_override} nm/pixel")
    print("=" * 60)

    if not results_folder.exists():
        print(f"[ERROR] Annotation results folder not found: {results_folder}")
        return

    # Extract data
    metadata, results, annotated_micrographs = extract_segmentation_data(results_folder, pixel_size_override)

    if len(metadata) == 0:
        print("\n[ERROR] No segmentation data found.")
        return

    # Get total micrograph count from folder if provided, otherwise use annotated count
    if micrograph_folder is not None:
        total_micrographs = len(get_image_files(micrograph_folder))
    else:
        total_micrographs = annotated_micrographs

    # Print summary
    print_summary(metadata, results, total_micrographs)

    # Determine output base path
    if output_path is None:
        output_base = results_folder / "extraction"
    else:
        # Remove extension if provided to use as base
        output_base = output_path.with_suffix('')

    # Save to files
    print(f"\nSaving results...")
    if output_format in ["csv", "both"]:
        metadata_csv = Path(str(output_base) + "_metadata.csv")
        results_csv = Path(str(output_base) + "_results.csv")
        save_metadata_csv(metadata, metadata_csv)
        save_results_csv(results, results_csv)

    if output_format in ["json", "both"]:
        json_path = Path(str(output_base) + ".json")
        save_to_json(metadata, results, json_path)

    print("\n" + "=" * 60)
    print("Extraction complete!")
    print("=" * 60)
