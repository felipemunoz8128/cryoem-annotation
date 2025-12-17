"""Extract labels and areas from annotation results."""

from pathlib import Path
from typing import List, Dict, Optional
import json
import csv

from cryoem_annotation.io.metadata import load_metadata


def extract_segmentation_data(results_folder: Path) -> List[Dict]:
    """
    Extract labels and areas from all metadata.json files.
    
    Args:
        results_folder: Path to annotation results folder
    
    Returns:
        List of dictionaries with segmentation data
    """
    all_data = []
    
    metadata_files = list(results_folder.glob("*/metadata.json"))
    
    if len(metadata_files) == 0:
        print(f"No metadata.json files found in {results_folder}")
        return all_data
    
    print(f"\nFound {len(metadata_files)} metadata file(s)\n")
    
    for metadata_file in sorted(metadata_files):
        micrograph_name = metadata_file.parent.name
        print(f"Processing: {micrograph_name}")
        
        metadata = load_metadata(metadata_file)
        if metadata is None:
            continue
        
        segmentations = metadata.get('segmentations', [])
        
        if len(segmentations) == 0:
            print(f"  [WARNING] No segmentations found")
            continue
        
        # Extract data for each segmentation
        for seg in segmentations:
            data_entry = {
                'micrograph_name': micrograph_name,
                'click_index': seg.get('click_index'),
                'label': seg.get('label'),
                'mask_area': seg.get('mask_area'),
                'click_coords': seg.get('click_coords'),
                'mask_score': seg.get('mask_score'),
            }
            all_data.append(data_entry)
        
        labeled_count = sum(1 for s in segmentations if s.get('label') is not None)
        print(f"  [OK] Extracted {len(segmentations)} segmentation(s) ({labeled_count} labeled)")
    
    return all_data


def save_to_csv(data: List[Dict], output_file: Path) -> None:
    """Save extracted data to CSV file."""
    if len(data) == 0:
        print("\nNo data to save.")
        return
    
    fieldnames = ['micrograph_name', 'click_index', 'label', 'mask_area', 
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
    
    print(f"\n[OK] Saved {len(data)} entries to {output_file}")


def save_to_json(data: List[Dict], output_file: Path) -> None:
    """Save extracted data to JSON file."""
    if len(data) == 0:
        print("\nNo data to save.")
        return
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"[OK] Saved {len(data)} entries to {output_file}")


def print_summary(data: List[Dict]) -> None:
    """Print summary statistics."""
    if len(data) == 0:
        return
    
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)
    
    total_segmentations = len(data)
    labeled_segmentations = sum(1 for d in data if d['label'] is not None)
    unlabeled_segmentations = total_segmentations - labeled_segmentations
    
    print(f"Total segmentations: {total_segmentations}")
    print(f"  Labeled: {labeled_segmentations}")
    print(f"  Unlabeled: {unlabeled_segmentations}")
    
    if labeled_segmentations > 0:
        label_counts = {}
        for d in data:
            if d['label'] is not None:
                label = d['label']
                label_counts[label] = label_counts.get(label, 0) + 1
        
        print(f"\nLabel distribution:")
        for label in sorted(label_counts.keys()):
            print(f"  Label {label}: {label_counts[label]} object(s)")
    
    areas = [d['mask_area'] for d in data if d['mask_area'] is not None]
    if areas:
        print(f"\nMask area statistics (pixels):")
        print(f"  Total objects: {len(areas)}")
        print(f"  Mean area: {sum(areas) / len(areas):.1f}")
        print(f"  Median area: {sorted(areas)[len(areas)//2]:.1f}")
        print(f"  Min area: {min(areas)}")
        print(f"  Max area: {max(areas)}")
    
    micrograph_counts = {}
    for d in data:
        name = d['micrograph_name']
        if name not in micrograph_counts:
            micrograph_counts[name] = {'total': 0, 'labeled': 0}
        micrograph_counts[name]['total'] += 1
        if d['label'] is not None:
            micrograph_counts[name]['labeled'] += 1
    
    print(f"\nMicrographs processed: {len(micrograph_counts)}")
    print(f"  Average segmentations per micrograph: {total_segmentations / len(micrograph_counts):.1f}")
    print("=" * 60)


def extract_results(
    results_folder: Path,
    output_path: Optional[Path] = None,
    output_format: str = "csv",
) -> None:
    """
    Extract results from annotation folder.
    
    Args:
        results_folder: Path to annotation results folder
        output_path: Path to output file (default: results.csv/json in results_folder)
        output_format: Output format: "csv", "json", or "both" (default: "csv")
    """
    print("=" * 60)
    print("Extract Labels and Areas from Segmentations")
    print("=" * 60)
    print(f"Annotation results folder: {results_folder}")
    print("=" * 60)
    
    if not results_folder.exists():
        print(f"[ERROR] Annotation results folder not found: {results_folder}")
        return
    
    # Extract data
    data = extract_segmentation_data(results_folder)
    
    if len(data) == 0:
        print("\n[ERROR] No segmentation data found.")
        return
    
    # Print summary
    print_summary(data)
    
    # Determine output paths
    if output_path is None:
        if output_format == "json":
            output_path = results_folder / "results.json"
        else:
            output_path = results_folder / "results.csv"
    
    # Save to files
    print(f"\nSaving results...")
    if output_format in ["csv", "both"]:
        csv_path = output_path if output_format == "csv" else output_path.with_suffix('.csv')
        save_to_csv(data, csv_path)
    
    if output_format in ["json", "both"]:
        json_path = output_path if output_format == "json" else output_path.with_suffix('.json')
        save_to_json(data, json_path)
    
    print("\n" + "=" * 60)
    print("Extraction complete!")
    print("=" * 60)

