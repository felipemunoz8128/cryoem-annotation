"""SAM model loading and management."""

from pathlib import Path
from typing import Optional
import numpy as np
import torch

# Try to import SAM
try:
    from segment_anything import sam_model_registry, SamPredictor
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    sam_model_registry = None
    SamPredictor = None

# Valid SAM model types
VALID_MODEL_TYPES = {'vit_b', 'vit_l', 'vit_h'}


class SAMModel:
    """Wrapper for SAM model and predictor with GPU memory optimization."""

    def __init__(self, model_type: str, checkpoint_path: Path, device: Optional[str] = None):
        """
        Initialize SAM model with optimized memory management.

        Args:
            model_type: SAM model type ("vit_b", "vit_l", or "vit_h")
            checkpoint_path: Path to SAM checkpoint file
            device: Device to use ("cuda", "cpu", or None for auto-detect)

        Raises:
            ImportError: If segment_anything is not installed
            ValueError: If model_type is not valid
            FileNotFoundError: If checkpoint file not found
        """
        if not SAM_AVAILABLE:
            raise ImportError(
                "segment_anything not installed. "
                "Install with: pip install git+https://github.com/facebookresearch/segment-anything.git"
            )

        # Validate model type early (fail fast)
        if model_type not in VALID_MODEL_TYPES:
            raise ValueError(
                f"Invalid model_type: '{model_type}'. "
                f"Valid options are: {', '.join(sorted(VALID_MODEL_TYPES))}"
            )

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model_type = model_type
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device

        # Find checkpoint if not found at specified path
        if not self.checkpoint_path.exists():
            alt_paths = [
                Path.home() / "Downloads" / self.checkpoint_path.name,
                Path("..") / self.checkpoint_path,
                Path(".") / self.checkpoint_path.name,
            ]
            for alt_path in alt_paths:
                if alt_path.exists():
                    self.checkpoint_path = alt_path
                    break

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"SAM checkpoint not found: {checkpoint_path}\n"
                "Please download a SAM checkpoint from:\n"
                "  ViT-B: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth\n"
                "  ViT-L: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth\n"
                "  ViT-H: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
            )

        # Load model with optimized settings
        self.sam = sam_model_registry[model_type](checkpoint=str(self.checkpoint_path))
        self.sam.to(device=device)
        self.sam.eval()

        # Create predictor
        self.predictor = SamPredictor(self.sam)

        # GPU memory optimization: clear cache after loading large model
        if device == "cuda":
            torch.cuda.empty_cache()
            mem_allocated = torch.cuda.memory_allocated(0) / 1e9
            mem_reserved = torch.cuda.memory_reserved(0) / 1e9
            print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")
            print(f"GPU Memory: allocated={mem_allocated:.2f}GB, reserved={mem_reserved:.2f}GB")
        else:
            print(f"Using device: {device} (CPU mode - will be slower)")

        # Select optimal inference context based on PyTorch version
        if hasattr(torch, 'inference_mode'):
            self._inference_context = torch.inference_mode
        else:
            self._inference_context = torch.no_grad
    
    def get_predictor(self):
        """Get the SAM predictor."""
        return self.predictor

    def predict_best_mask(
        self,
        point_coords: np.ndarray,
        point_labels: Optional[np.ndarray] = None,
        multimask_output: bool = True
    ):
        """Predict and return the best mask with optimized inference.

        This method wraps the predictor's predict() call with torch.inference_mode()
        for better memory efficiency during inference.

        Args:
            point_coords: Array of (x, y) coordinates, shape (N, 2)
            point_labels: Array of labels (1=foreground, 0=background), shape (N,)
                         If None, all points are treated as foreground.
            multimask_output: If True, returns 3 masks; if False, returns 1 mask.

        Returns:
            tuple: (best_mask, best_score, all_scores)
                - best_mask: Boolean mask array for the highest-scoring prediction
                - best_score: Confidence score of the best mask
                - all_scores: Array of all prediction scores
        """
        if point_labels is None:
            point_labels = np.ones(len(point_coords), dtype=np.int32)

        with self._inference_context():
            masks, scores, _ = self.predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=multimask_output,
            )

        best_idx = np.argmax(scores)
        return masks[best_idx], scores[best_idx], scores

    def clear_gpu_cache(self):
        """Clear GPU cache to free memory.

        Call this between processing different images to reduce memory fragmentation.
        """
        if self.device == "cuda":
            torch.cuda.empty_cache()


def load_sam_model(
    model_type: str = "vit_b",
    checkpoint_path: Optional[Path] = None,
    device: Optional[str] = None
) -> SAMModel:
    """
    Load SAM model.
    
    Args:
        model_type: SAM model type ("vit_b", "vit_l", or "vit_h")
        checkpoint_path: Path to SAM checkpoint file (default: "sam_vit_b_01ec64.pth")
        device: Device to use ("cuda", "cpu", or None for auto-detect)
    
    Returns:
        SAMModel instance
    """
    if checkpoint_path is None:
        checkpoint_path = Path("sam_vit_b_01ec64.pth")
    
    return SAMModel(model_type, checkpoint_path, device)

