"""Interactive labeling tool for segmentations."""

from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from cryoem_annotation.core.image_loader import load_micrograph, get_image_files
from cryoem_annotation.core.image_processing import normalize_image
from cryoem_annotation.core.colors import generate_label_colors
from cryoem_annotation.io.metadata import load_metadata, save_metadata
from cryoem_annotation.io.masks import load_mask_binary

# Try to import tkinter for dialog boxes
try:
    import tkinter as tk
    from tkinter import messagebox
    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False

# Configure matplotlib backend
_backend_set = False
_backend_options = ['Qt5Agg', 'MacOSX', 'TkAgg']

for backend_name in _backend_options:
    try:
        matplotlib.use(backend_name)
        import matplotlib.pyplot as plt
        test_fig = plt.figure()
        plt.close(test_fig)
        _backend_set = True
        break
    except Exception:
        continue

if not _backend_set:
    print("Warning: Could not set interactive backend. Using default.")


def get_contour_from_mask(mask: np.ndarray) -> Optional[np.ndarray]:
    """Get contour points from a binary mask."""
    if mask is None:
        return None
    
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return None
    
    largest_contour = max(contours, key=cv2.contourArea)
    epsilon = 0.002 * cv2.arcLength(largest_contour, True)
    simplified = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    return simplified.reshape(-1, 2)


def point_in_mask(point: Tuple[int, int], mask: np.ndarray) -> bool:
    """Check if a point is inside a mask."""
    if mask is None:
        return False
    x, y = point
    if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
        return mask[y, x]
    return False


class SegmentationLabeler:
    """Interactive tool for labeling segmentations."""
    
    def __init__(self, image: np.ndarray, segmentations: List[Dict], title: str = "Label Segmentations"):
        """Initialize labeler."""
        self.image = image
        self.segmentations = segmentations  # Will be modified in place
        self.title = title
        self.fig = None
        self.ax = None
        self.finished = False
        
        self.current_label = 1  # Default label is 1
        
        self.contour_patches = []
        self.text_artists = []
        
        # Color palette for labels
        self.label_colors = generate_label_colors(10)
    
    def _update_title(self):
        """Update the figure title."""
        title_text = (f"{self.title}\n"
                     f"Active Label: {self.current_label} | "
                     f"Left-click: Assign label | Right-click: Finish | "
                     f"'0'-'9': Set Label")
        if self.ax:
            self.ax.set_title(title_text, fontsize=14, fontweight='bold')
    
    def _draw_segmentations(self):
        """Draw all segmentations as outlines with labels."""
        # Clear existing patches and text
        for patch, _ in self.contour_patches:
            patch.remove()
        for text in self.text_artists:
            text.remove()
        self.contour_patches = []
        self.text_artists = []
        
        # Draw each segmentation
        for i, seg_data in enumerate(self.segmentations):
            mask = seg_data.get('mask')
            if mask is None:
                continue
            
            contour = get_contour_from_mask(mask)
            if contour is None or len(contour) < 3:
                continue
            
            label = seg_data.get('label')
            if label is None:
                color = [0.0, 1.0, 1.0]  # Bright cyan for unlabeled
                linewidth = 1.0
                linestyle = '--'
            else:
                label_idx = (label - 1) % 10 if label > 0 else 0
                color = self.label_colors[label_idx]
                linewidth = 1.0
                linestyle = '-'
            
            polygon = Polygon(contour, closed=True, fill=False, 
                            edgecolor=color, linewidth=linewidth, 
                            linestyle=linestyle, alpha=0.5, zorder=5)
            self.ax.add_patch(polygon)
            self.contour_patches.append((polygon, i))
            
            # Add label text at centroid
            if label is not None:
                M = cv2.moments(mask.astype(np.uint8))
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    text = self.ax.text(cx, cy, f"L{label}", 
                                       color='white', fontsize=10, fontweight='bold',
                                       ha='center', va='center',
                                       bbox=dict(boxstyle='round', facecolor=color, alpha=0.7),
                                       zorder=6)
                    self.text_artists.append(text)
    
    def on_click(self, event):
        """Handle mouse clicks."""
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        
        # Right click to finish
        if event.button == 3 or event.button == 2:
            unlabeled_count = sum(1 for s in self.segmentations if s.get('label') is None)
            
            if unlabeled_count > 0:
                if TKINTER_AVAILABLE:
                    root = tk.Tk()
                    root.withdraw()
                    root.attributes('-topmost', True)
                    response = messagebox.askyesno(
                        "Unlabeled Segmentations",
                        f"There are {unlabeled_count} segmentation(s) missing labels.\n"
                        f"Are you sure you want to move to the next image?",
                        parent=root
                    )
                    root.destroy()
                    if not response:
                        print(f"\n  Staying on current image. {unlabeled_count} segmentation(s) still unlabeled.")
                        return
                else:
                    print(f"\n  WARNING: There are {unlabeled_count} segmentation(s) missing labels.")
                    response = input("  Are you sure you want to move to the next image? (yes/no): ").strip().lower()
                    if response not in ['yes', 'y']:
                        print(f"  Staying on current image. {unlabeled_count} segmentation(s) still unlabeled.")
                        return
            
            self.finished = True
            print(f"\n  [OK] Right-click detected. Finished labeling.")
            try:
                plt.close(self.fig)
            except:
                pass
            return
        
        # Left click to assign label
        if event.button == 1:
            x, y = int(event.xdata), int(event.ydata)
            
            clicked_seg = None
            clicked_idx = None
            
            for i, seg_data in enumerate(self.segmentations):
                mask = seg_data.get('mask')
                if mask is not None and point_in_mask((x, y), mask):
                    clicked_seg = seg_data
                    clicked_idx = i
                    break
            
            if clicked_seg is not None:
                old_label = clicked_seg.get('label')
                clicked_seg['label'] = self.current_label
                print(f"  [OK] Assigned label {self.current_label} to segmentation {clicked_idx + 1} (was: {old_label})")
                self._draw_segmentations()
                self.fig.canvas.draw()
            else:
                print(f"  Click at ({x}, {y}) is not inside any segmentation")
    
    def on_key(self, event):
        """Handle keyboard events for label selection."""
        if event.key.isdigit():
            label_num = int(event.key)
            if 0 <= label_num <= 9:
                self.current_label = label_num if label_num > 0 else 10
                self._update_title()
                self.fig.canvas.draw()
                print(f"  -> Active label set to: {self.current_label}")
    
    def label_segmentations(self) -> List[Dict]:
        """Display image with segmentations and collect labels."""
        try:
            self.fig, self.ax = plt.subplots(figsize=(12, 12))
        except Exception as e:
            if "macOS" in str(e) or "2600" in str(e) or "1600" in str(e):
                print(f"\n  [ERROR] Backend version check failed: {e}")
                print("  Falling back to console input method...")
                return self._label_fallback()
            else:
                raise
        
        self.ax.imshow(self.image, cmap='gray')
        self._update_title()
        self.ax.axis('off')
        
        self._draw_segmentations()
        
        cid_click = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        cid_key = self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
        plt.tight_layout()
        
        print("\n  Figure window opened. Label segmentations by clicking on them.")
        print("  Instructions:")
        print("    - Press '0'-'9': Set active label (0 = label 10)")
        print("    - Left-click: Assign active label to clicked segmentation")
        print("    - Right-click: Finish and proceed to next micrograph")
        print(f"  Current active label: {self.current_label}")
        print(f"  Unlabeled segmentations: {sum(1 for s in self.segmentations if s.get('label') is None)}")
        print()
        
        try:
            plt.show(block=True)
        except Exception as e:
            if "macOS" in str(e) or "2600" in str(e) or "1600" in str(e):
                print(f"\n  [ERROR] Could not display figure: {e}")
                print("  Falling back to console input method...")
                try:
                    plt.close(self.fig)
                except:
                    pass
                return self._label_fallback()
            else:
                raise
        
        try:
            self.fig.canvas.mpl_disconnect(cid_click)
            self.fig.canvas.mpl_disconnect(cid_key)
        except:
            pass
        
        labeled_count = sum(1 for s in self.segmentations if s.get('label') is not None)
        print(f"  Final labeled count: {labeled_count}/{len(self.segmentations)}")
        
        return self.segmentations
    
    def _label_fallback(self) -> List[Dict]:
        """Fallback method: label via console input."""
        print("\n  =========================================")
        print("  FALLBACK MODE: Manual Label Input")
        print("  =========================================")
        print("  Enter segmentation number and label.")
        print("  Format: seg_num,label (e.g., 1,2)")
        print("  Press Enter with empty input when done.")
        print("  =========================================\n")
        
        print(f"  {len(self.segmentations)} segmentations to label\n")
        
        while True:
            try:
                input_str = input("  Enter 'seg_num,label' or press Enter to finish: ").strip()
                if not input_str:
                    break
                
                parts = input_str.split(',')
                if len(parts) == 2:
                    seg_num, label = map(int, parts)
                    if 1 <= seg_num <= len(self.segmentations):
                        self.segmentations[seg_num - 1]['label'] = label
                        print(f"    [OK] Labeled segmentation {seg_num} as {label}")
                    else:
                        print(f"    [ERROR] Segmentation number out of range (1-{len(self.segmentations)})")
                else:
                    print("    [ERROR] Invalid format. Use: seg_num,label (e.g., 1,2)")
            except ValueError:
                print("    ✗ Invalid format. Use: seg_num,label (e.g., 1,2)")
            except (EOFError, KeyboardInterrupt):
                print("\n  Interrupted.")
                break
        
        return self.segmentations


def load_segmentations(results_folder: Path, micrograph_name: str) -> Optional[Dict]:
    """Load segmentations for a micrograph from annotation results."""
    micrograph_stem = Path(micrograph_name).stem
    annotation_dir = results_folder / micrograph_stem
    
    if not annotation_dir.exists():
        return None
    
    metadata_file = annotation_dir / "metadata.json"
    if not metadata_file.exists():
        return None
    
    metadata = load_metadata(metadata_file)
    if metadata is None:
        return None
    
    # Load binary masks
    masks = []
    for seg in metadata['segmentations']:
        click_idx = seg['click_index']
        mask_file = annotation_dir / f"mask_{click_idx:03d}_binary.png"
        mask = load_mask_binary(mask_file)
        masks.append(mask)
    
    # Add masks to segmentations
    for i, seg in enumerate(metadata['segmentations']):
        seg['mask'] = masks[i]
        if 'label' not in seg:
            seg['label'] = None
    
    return metadata


def label_segmentations(
    results_folder: Path,
    micrograph_folder: Path,
) -> None:
    """
    Label segmentations for all micrographs.
    
    Args:
        results_folder: Path to annotation results folder
        micrograph_folder: Path to micrograph folder
    """
    print("=" * 60)
    print("Interactive Segmentation Labeling Tool")
    print("=" * 60)
    print(f"Micrograph folder: {micrograph_folder}")
    print(f"Annotation results folder: {results_folder}")
    print("=" * 60)
    
    # Get micrograph files that have annotations
    image_files = get_image_files(micrograph_folder)
    
    annotated_files = []
    for f in image_files:
        stem = f.stem
        annotation_dir = results_folder / stem
        if annotation_dir.exists() and (annotation_dir / "metadata.json").exists():
            annotated_files.append(f)
    
    if len(annotated_files) == 0:
        print(f"No micrographs with annotations found in {micrograph_folder}")
        return
    
    print(f"\nFound {len(annotated_files)} micrograph(s) with segmentations\n")
    print("=" * 60)
    
    # Process each file
    for idx, file_path in enumerate(annotated_files):
        print(f"\n[{idx+1}/{len(annotated_files)}] Labeling: {file_path.name}")
        print("-" * 60)
        
        micrograph = load_micrograph(file_path)
        if micrograph is None:
            print(f"  [ERROR] Failed to load {file_path.name}, skipping...")
            continue
        
        micrograph_display = normalize_image(micrograph, percentile=(1, 99))
        
        metadata = load_segmentations(results_folder, file_path.name)
        if metadata is None:
            print(f"  [ERROR] No segmentations found for {file_path.name}, skipping...")
            continue
        
        segmentations = metadata['segmentations']
        print(f"  [OK] Loaded {len(segmentations)} segmentation(s)")
        
        already_labeled = sum(1 for s in segmentations if s.get('label') is not None)
        if already_labeled > 0:
            print(f"  -> {already_labeled} already labeled, {len(segmentations) - already_labeled} unlabeled")
        
        labeler = SegmentationLabeler(
            micrograph_display,
            segmentations,
            title=file_path.name
        )
        
        labeled_segmentations = labeler.label_segmentations()
        
        # Save updated metadata
        annotation_dir = results_folder / file_path.stem
        
        # Remove mask arrays before saving
        serializable_segmentations = []
        for seg in labeled_segmentations:
            seg_copy = {k: v for k, v in seg.items() if k != 'mask'}
            serializable_segmentations.append(seg_copy)
        
        metadata['segmentations'] = serializable_segmentations
        metadata['labeling_timestamp'] = datetime.now().isoformat()
        
        save_metadata(metadata, annotation_dir / "metadata.json")
        
        final_labeled = sum(1 for s in labeled_segmentations if s.get('label') is not None)
        print(f"  [OK] Saved labels: {final_labeled}/{len(labeled_segmentations)} labeled")
    
    print("\n" + "=" * 60)
    print(f"[OK] Labeling complete!")
    print(f"  Processed {len(annotated_files)} micrograph(s)")
    print(f"  Results updated in: {results_folder}")
    print("=" * 60)

