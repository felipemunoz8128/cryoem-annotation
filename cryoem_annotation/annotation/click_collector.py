"""Click collection with real-time SAM segmentation."""

from typing import List, Dict, Tuple, Optional
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from cryoem_annotation.core.colors import generate_label_colors


def create_bounded_overlay(mask: np.ndarray, color: list) -> Tuple[Optional[np.ndarray], Optional[list]]:
    """Create overlay only for mask bounding box region.

    This is an optimized approach that creates a small RGBA array covering
    only the mask's bounding box instead of the full image. For typical
    cryo-EM masks (~1-5% coverage), this reduces memory by 90%+.

    Args:
        mask: Boolean mask array
        color: RGBA color list [r, g, b, a]

    Returns:
        tuple: (overlay_array, extent) for ax.imshow(), or (None, None) if empty
    """
    # Find bounding box of non-zero pixels
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if not np.any(rows) or not np.any(cols):
        return None, None

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # Add small padding for cleaner edges
    pad = 2
    rmin = max(0, rmin - pad)
    rmax = min(mask.shape[0] - 1, rmax + pad)
    cmin = max(0, cmin - pad)
    cmax = min(mask.shape[1] - 1, cmax + pad)

    # Create overlay only for the bounding box region
    cropped_mask = mask[rmin:rmax+1, cmin:cmax+1]
    overlay = np.zeros((*cropped_mask.shape, 4), dtype=np.float32)
    overlay[cropped_mask] = color

    # Extent for matplotlib imshow: [left, right, bottom, top]
    extent = [cmin, cmax+1, rmax+1, rmin]

    return overlay, extent

# Configure matplotlib backend for interactivity
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

import matplotlib.pyplot as plt
plt.ion()  # Keep interactive mode ON


class RealTimeClickCollector:
    """Collects mouse clicks and shows SAM segmentations in real-time."""
    
    def __init__(self, image: np.ndarray, micrograph_rgb: np.ndarray, 
                 predictor, title: str = "Click on objects"):
        """
        Initialize click collector.
        
        Args:
            image: Image for display (grayscale, normalized)
            micrograph_rgb: RGB image for SAM (3-channel)
            predictor: SAM predictor instance
            title: Window title
        """
        self.image = image  # For display
        self.micrograph_rgb = micrograph_rgb  # For SAM
        self.predictor = predictor
        self.title = title
        self.fig = None
        self.ax = None
        self.finished = False
        
        # Store clicks and their corresponding segmentations
        self.clicks = []  # List of (x, y) tuples
        self.segmentations = []  # List of dicts with mask, score, etc.

        # Track individual matplotlib artists for efficient undo
        self.mask_overlays = []  # List of mask overlay artists
        self.click_markers = []  # List of click marker artists
        self.click_texts = []  # List of click text artists

        # Color palette for masks
        base_colors = generate_label_colors(50)
        self.colors = [[*rgb, 1.0] for rgb in base_colors]  # Add alpha channel
    
    def on_click(self, event):
        """Handle mouse clicks."""
        if event.inaxes != self.ax:
            return
        
        # Right click or middle click to finish
        if event.button == 3 or event.button == 2:
            self.finished = True
            print(f"\n  [OK] Right-click detected. Finished with {len(self.clicks)} segmentation(s).")
            try:
                plt.close(self.fig)
            except Exception:
                pass
            return
        
        # Left click to add point and segment
        if event.button == 1:
            x, y = int(event.xdata), int(event.ydata)
            
            print(f"  Processing click at ({x}, {y})...")
            
            # Run SAM prediction immediately
            point_coords = np.array([[x, y]])
            point_labels = np.array([1])  # Foreground point
            
            masks, scores, logits = self.predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True,  # Get 3 masks
            )
            
            # Select best mask (highest score)
            best_mask_idx = np.argmax(scores)
            best_mask = masks[best_mask_idx]
            best_score = scores[best_mask_idx]
            
            print(f"    [OK] Segmentation created (score: {best_score:.3f})")
            
            # Store the click and segmentation
            self.clicks.append((x, y))
            seg_data = {
                'click_index': len(self.clicks),
                'click_coords': [int(x), int(y)],
                'mask': best_mask,
                'mask_score': float(best_score),
                'mask_area': int(np.sum(best_mask)),
            }
            self.segmentations.append(seg_data)
            
            # Visual feedback: draw click marker (track for efficient undo)
            marker = self.ax.plot(x, y, 'r+', markersize=15, markeredgewidth=2, zorder=10)[0]
            self.click_markers.append(marker)

            text = self.ax.text(x + 5, y - 5, f"{len(self.clicks)}",
                               color='red', fontsize=12, fontweight='bold',
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                               zorder=11)
            self.click_texts.append(text)

            # Add mask overlay using bounded approach (90%+ memory reduction)
            color = self.colors[(len(self.segmentations) - 1) % len(self.colors)]
            overlay, extent = create_bounded_overlay(best_mask, [*color[:3], 0.4])
            if overlay is not None:
                im = self.ax.imshow(overlay, extent=extent, zorder=5, interpolation='nearest')
                self.mask_overlays.append(im)
            else:
                self.mask_overlays.append(None)  # Keep list aligned
            
            # Update display (use draw_idle for better performance)
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            
            print(f"  -> Total segmentations: {len(self.segmentations)}")
    
    def on_key(self, event):
        """Handle keyboard events for undo/delete."""
        if event.key in ['d', 'D', 'u', 'U', 'delete', 'backspace']:
            if len(self.segmentations) > 0:
                # Remove last segmentation data
                self.segmentations.pop()
                self.clicks.pop()

                # Efficient undo: remove individual artists instead of full redraw
                # Remove mask overlay
                if self.mask_overlays:
                    overlay = self.mask_overlays.pop()
                    if overlay is not None:
                        overlay.remove()

                # Remove click marker
                if self.click_markers:
                    marker = self.click_markers.pop()
                    marker.remove()

                # Remove click text
                if self.click_texts:
                    text = self.click_texts.pop()
                    text.remove()

                # Efficient redraw - only affected region
                self.fig.canvas.draw_idle()
                self.fig.canvas.flush_events()

                print(f"  Undone last segmentation. Remaining: {len(self.segmentations)}")
            else:
                print("  No segmentations to undo.")
    
    def _update_title(self):
        """Update the figure title."""
        title_text = (f"{self.title}\n"
                     f"Left-click: Segment | Right-click: Finish | "
                     f"'d'/'u': Undo")
        self.ax.set_title(title_text, fontsize=14, fontweight='bold')
    
    def collect_clicks(self) -> Tuple[List[Tuple[int, int]], List[Dict]]:
        """
        Display image and collect clicks with real-time segmentation.
        
        Returns:
            Tuple of (clicks list, segmentations list)
        """
        try:
            # Optimize figure for performance
            self.fig, self.ax = plt.subplots(figsize=(12, 12))
        except Exception as e:
            if "macOS" in str(e) or "2600" in str(e) or "1600" in str(e):
                print(f"\n  [ERROR] Backend version check failed: {e}")
                print("  Falling back to coordinate input method...")
                return self._collect_clicks_fallback()
            else:
                raise
        
        # Display base image (keep default interpolation for quality)
        self.base_image = self.ax.imshow(self.image, cmap='gray')
        # Disable autoscaling AFTER image is displayed (for better performance)
        self.ax.set_autoscale_on(False)
        self._update_title()
        self.ax.axis('off')
        
        # Connect event handlers
        cid_click = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        cid_key = self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
        plt.tight_layout()
        
        print("\n  Figure window opened. Click on objects in the image.")
        print("  Instructions:")
        print("    - Left-click: Segment an object (mask appears immediately)")
        print("    - Right-click: Finish and proceed")
        print("    - Press 'd' or 'u': Undo last segmentation")
        print()
        
        try:
            plt.show(block=True)
        except Exception as e:
            if "macOS" in str(e) or "2600" in str(e) or "1600" in str(e):
                print(f"\n  [ERROR] Could not display figure: {e}")
                print("  Falling back to coordinate input method...")
                try:
                    plt.close(self.fig)
                except Exception:
                    pass
                return self._collect_clicks_fallback()
            else:
                raise
        
        # Disconnect event handlers
        try:
            self.fig.canvas.mpl_disconnect(cid_click)
            self.fig.canvas.mpl_disconnect(cid_key)
        except Exception:
            pass
        
        print(f"  Final segmentation count: {len(self.segmentations)}")
        
        return self.clicks, self.segmentations
    
    def _collect_clicks_fallback(self) -> Tuple[List[Tuple[int, int]], List[Dict]]:
        """Fallback method: collect coordinates via console input."""
        print("\n  =========================================")
        print("  FALLBACK MODE: Manual Coordinate Input")
        print("  =========================================")
        print("  Since the GUI backend failed, please enter coordinates manually.")
        print("  Format: x,y (e.g., 100,200)")
        print("  Press Enter with empty input when done.")
        print("  =========================================\n")
        
        h, w = self.image.shape[:2]
        print(f"  Image size: {w} x {h}")
        print()
        
        clicks = []
        segmentations = []
        
        while True:
            try:
                coord_input = input(f"  Enter coordinate {len(clicks) + 1} (x,y) or press Enter to finish: ").strip()
                if not coord_input:
                    break
                
                x, y = map(int, coord_input.split(','))
                if 0 <= x < w and 0 <= y < h:
                    clicks.append((x, y))
                    print(f"    [OK] Added click {len(clicks)}: ({x}, {y})")
                    
                    # Run SAM prediction
                    point_coords = np.array([[x, y]])
                    point_labels = np.array([1])
                    masks, scores, logits = self.predictor.predict(
                        point_coords=point_coords,
                        point_labels=point_labels,
                        multimask_output=True,
                    )
                    best_mask_idx = np.argmax(scores)
                    best_mask = masks[best_mask_idx]
                    best_score = scores[best_mask_idx]
                    
                    seg_data = {
                        'click_index': len(clicks),
                        'click_coords': [int(x), int(y)],
                        'mask': best_mask,
                        'mask_score': float(best_score),
                        'mask_area': int(np.sum(best_mask)),
                    }
                    segmentations.append(seg_data)
                    print(f"    [OK] Segmentation created (score: {best_score:.3f})")
                else:
                    print(f"    [ERROR] Coordinates out of range. Image is {w}x{h}")
            except ValueError:
                print("    [ERROR] Invalid format. Use: x,y (e.g., 100,200)")
            except (EOFError, KeyboardInterrupt):
                print("\n  Interrupted.")
                break
        
        return clicks, segmentations

