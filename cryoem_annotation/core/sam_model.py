"""SAM model loading and management."""

from pathlib import Path
from typing import Optional
import torch

# Try to import SAM
try:
    from segment_anything import sam_model_registry, SamPredictor
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    sam_model_registry = None
    SamPredictor = None


class SAMModel:
    """Wrapper for SAM model and predictor."""
    
    def __init__(self, model_type: str, checkpoint_path: Path, device: Optional[str] = None):
        """
        Initialize SAM model.
        
        Args:
            model_type: SAM model type ("vit_b", "vit_l", or "vit_h")
            checkpoint_path: Path to SAM checkpoint file
            device: Device to use ("cuda", "cpu", or None for auto-detect)
        """
        if not SAM_AVAILABLE:
            raise ImportError(
                "segment_anything not installed. "
                "Install with: pip install git+https://github.com/facebookresearch/segment-anything.git"
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
        
        # Load model
        self.sam = sam_model_registry[model_type](checkpoint=str(self.checkpoint_path))
        self.sam.to(device=device)
        self.sam.eval()
        
        self.predictor = SamPredictor(self.sam)
        
        if device == "cuda":
            print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print(f"Using device: {device} (CPU mode - will be slower)")
    
    def get_predictor(self):
        """Get the SAM predictor."""
        return self.predictor


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

