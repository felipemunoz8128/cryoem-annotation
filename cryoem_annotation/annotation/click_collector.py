"""Click collection with real-time SAM segmentation."""

from typing import List, Dict, Tuple
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from cryoem_annotation.core.colors import generate_label_colors

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
        self.mask_overlays = []  # List of mask overlay artists for removal
        
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
            print(f"\n  ✓ Right-click detected. Finished with {len(self.clicks)} segmentation(s).")
            try:
                plt.close(self.fig)
            except:
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
            
            print(f"    ✓ Segmentation created (score: {best_score:.3f})")
            
            # Store the click and segmentation
            self.clicks.append((x, y))
            seg_data = {
                'click_index': len(self.clicks),
                'click_coords': [int(x), int(y)],
                'mask': best_mask,
                'mask_score': float(best_score),
                'all_scores': [float(s) for s in scores],
                'mask_area': int(np.sum(best_mask)),
            }
            self.segmentations.append(seg_data)
            
            # Visual feedback: draw click marker
            self.ax.plot(x, y, 'r+', markersize=15, markeredgewidth=2, zorder=10)
            self.ax.text(x + 5, y - 5, f"{len(self.clicks)}", 
                        color='red', fontsize=12, fontweight='bold',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                        zorder=11)
            
            # Add mask overlay
            color = self.colors[(len(self.segmentations) - 1) % len(self.colors)]
            mask_overlay = np.zeros((*best_mask.shape, 4))
            mask_overlay[best_mask] = [*color[:3], 0.4]  # Semi-transparent overlay
            im = self.ax.imshow(mask_overlay, zorder=5)
            self.mask_overlays.append(im)
            
            # Update display
            self.fig.canvas.draw()
            
            print(f"  → Total segmentations: {len(self.segmentations)}")
    
    def on_key(self, event):
        """Handle keyboard events for undo/delete."""
        if event.key in ['d', 'D', 'u', 'U', 'delete', 'backspace']:
            if len(self.segmentations) > 0:
                # Remove last segmentation
                self.segmentations.pop()
                self.clicks.pop()
                
                # Remove the mask overlay
                if self.mask_overlays:
                    overlay = self.mask_overlays.pop()
                    overlay.remove()
                
                # Redraw the image to remove click markers
                self.ax.clear()
                self.ax.imshow(self.image, cmap='gray')
                self._update_title()
                self.ax.axis('off')
                
                # Clear the list of old artists before repopulating
                self.mask_overlays = []
                
                # Redraw remaining clicks and masks
                for i, ((cx, cy), seg_data) in enumerate(zip(self.clicks, self.segmentations)):
                    # Click marker
                    self.ax.plot(cx, cy, 'r+', markersize=15, markeredgewidth=2, zorder=10)
                    self.ax.text(cx + 5, cy - 5, f"{i+1}", 
                                color='red', fontsize=12, fontweight='bold',
                                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                                zorder=11)
                    
                    # Mask overlay
                    color = self.colors[i % len(self.colors)]
                    mask = seg_data['mask']
                    mask_overlay = np.zeros((*mask.shape, 4))
                    mask_overlay[mask] = [*color[:3], 0.4]
                    im = self.ax.imshow(mask_overlay, zorder=5)
                    self.mask_overlays.append(im)
                
                self.fig.canvas.draw()
                print(f"  ✓ Undone last segmentation. Remaining: {len(self.segmentations)}")
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
            self.fig, self.ax = plt.subplots(figsize=(12, 12))
        except Exception as e:
            if "macOS" in str(e) or "2600" in str(e) or "1600" in str(e):
                print(f"\n  ✗ ERROR: Backend version check failed: {e}")
                print("  Falling back to coordinate input method...")
                return self._collect_clicks_fallback()
            else:
                raise
        
        # Display base image
        self.ax.imshow(self.image, cmap='gray')
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
                print(f"\n  ✗ ERROR: Could not display figure: {e}")
                print("  Falling back to coordinate input method...")
                try:
                    plt.close(self.fig)
                except:
                    pass
                return self._collect_clicks_fallback()
            else:
                raise
        
        # Disconnect event handlers
        try:
            self.fig.canvas.mpl_disconnect(cid_click)
            self.fig.canvas.mpl_disconnect(cid_key)
        except:
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
                    print(f"    ✓ Added click {len(clicks)}: ({x}, {y})")
                    
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
                        'all_scores': [float(s) for s in scores],
                        'mask_area': int(np.sum(best_mask)),
                    }
                    segmentations.append(seg_data)
                    print(f"    ✓ Segmentation created (score: {best_score:.3f})")
                else:
                    print(f"    ✗ Coordinates out of range. Image is {w}x{h}")
            except ValueError:
                print("    ✗ Invalid format. Use: x,y (e.g., 100,200)")
            except (EOFError, KeyboardInterrupt):
                print("\n  Interrupted.")
                break
        
        return clicks, segmentations

