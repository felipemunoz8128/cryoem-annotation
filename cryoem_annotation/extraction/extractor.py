"""Extract labels and areas from annotation results."""

from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json
import csv
import math
from tqdm import tqdm

from cryoem_annotation.io.metadata import load_metadata


def extract_segmentation_data(
    results_folder: Path,
    pixel_size_override: Optional[float] = None
) -> Tuple[List[Dict], List[Dict]]:
    """
    Extract labels and areas from all metadata.json files.

    Args:
        results_folder: Path to annotation results folder
        pixel_size_override: Override pixel size in nm/pixel for all micrographs

    Returns:
        Tuple of (metadata_list, results_list) where each entry has a segmentation_id
    """
    metadata_list = []
    results_list = []

    metadata_files = list(results_folder.glob("*/metadata.json"))

    if len(metadata_files) == 0:
        print(f"No metadata.json files found in {results_folder}")
        return metadata_list, results_list

    print(f"\nFound {len(metadata_files)} metadata file(s)\n")

    for metadata_file in tqdm(sorted(metadata_files), desc="Extracting", unit="file"):
        micrograph_name = metadata_file.parent.name
        print(f"Processing: {micrograph_name}")

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
            click_index = seg.get('click_index')

            # Create unique segmentation ID
            seg_id = f"{micrograph_name}_seg{click_index:03d}"

            # Metadata entry
            metadata_entry = {
                'segmentation_id': seg_id,
                'micrograph_name': micrograph_name,
                'click_index': click_index,
                'click_coords': seg.get('click_coords'),
                'mask_score': seg.get('mask_score'),
            }
            metadata_list.append(metadata_entry)

            # Results entry
            area_pixels = seg.get('mask_area')
            diameter_nm = None
            if area_pixels is not None and pixel_size_nm is not None:
                # Convert area to equivalent circle diameter
                # Area_nm2 = area_pixels * (pixel_size_nm)^2
                # Diameter = 2 * sqrt(Area / pi) = sqrt(4 * Area / pi)
                area_nm2 = area_pixels * (pixel_size_nm ** 2)
                diameter_nm = math.sqrt(4 * area_nm2 / math.pi)

            results_entry = {
                'segmentation_id': seg_id,
                'label': seg.get('label'),
                'area_pixels': area_pixels,
                'diameter_nm': diameter_nm,
            }
            results_list.append(results_entry)

        labeled_count = sum(1 for s in segmentations if s.get('label') is not None)
        print(f"  [OK] Extracted {len(segmentations)} segmentation(s) ({labeled_count} labeled)")

    return metadata_list, results_list


def save_metadata_csv(data: List[Dict], output_file: Path) -> None:
    """Save metadata entries to CSV file."""
    if len(data) == 0:
        print("\nNo metadata to save.")
        return

    fieldnames = ['segmentation_id', 'micrograph_name', 'click_index',
                  'click_coords', 'mask_score']

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

    fieldnames = ['segmentation_id', 'label', 'area_pixels', 'diameter_nm']

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


def print_summary(metadata: List[Dict], results: List[Dict]) -> None:
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

    # Area statistics in pixels
    areas_pixels = [d['area_pixels'] for d in results if d['area_pixels'] is not None]
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
    micrograph_names = set(d['micrograph_name'] for d in metadata)
    print(f"\nMicrographs processed: {len(micrograph_names)}")
    print(f"  Average segmentations per micrograph: {total_segmentations / len(micrograph_names):.1f}")
    print("=" * 60)


def extract_results(
    results_folder: Path,
    output_path: Optional[Path] = None,
    output_format: str = "csv",
    pixel_size_override: Optional[float] = None,
) -> None:
    """
    Extract results from annotation folder.

    Args:
        results_folder: Path to annotation results folder
        output_path: Base path for output files (default: results in results_folder)
        output_format: Output format: "csv", "json", or "both" (default: "csv")
        pixel_size_override: Override pixel size in nm/pixel for all micrographs
    """
    print("=" * 60)
    print("Extract Labels and Areas from Segmentations")
    print("=" * 60)
    print(f"Annotation results folder: {results_folder}")
    if pixel_size_override is not None:
        print(f"Pixel size override: {pixel_size_override} nm/pixel")
    print("=" * 60)

    if not results_folder.exists():
        print(f"[ERROR] Annotation results folder not found: {results_folder}")
        return

    # Extract data
    metadata, results = extract_segmentation_data(results_folder, pixel_size_override)

    if len(metadata) == 0:
        print("\n[ERROR] No segmentation data found.")
        return

    # Print summary
    print_summary(metadata, results)

    # Determine output base path
    if output_path is None:
        output_base = results_folder / "results"
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
