"""Shared pytest fixtures for Cryo-EM Annotation Tool tests."""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import json
from typing import Dict, List, Any
from unittest.mock import MagicMock


# =============================================================================
# Image Fixtures
# =============================================================================

@pytest.fixture
def sample_grayscale_image():
    """Generate a 1024x1024 test image with realistic cryo-EM-like values."""
    np.random.seed(42)
    # Simulate cryo-EM contrast: mostly mid-gray with some particles
    base = np.random.randint(20000, 45000, (1024, 1024), dtype=np.uint16)
    return base


@pytest.fixture
def sample_4k_image():
    """Generate a 4096x4096 test image for performance testing."""
    np.random.seed(42)
    return np.random.randint(0, 65535, (4096, 4096), dtype=np.uint16)


@pytest.fixture
def sample_normalized_image(sample_grayscale_image):
    """Generate a normalized uint8 image."""
    from cryoem_annotation.core.image_processing import normalize_image
    return normalize_image(sample_grayscale_image)


@pytest.fixture
def sample_rgb_image(sample_normalized_image):
    """Generate an RGB image (as used for SAM input)."""
    gray = sample_normalized_image
    return np.stack([gray, gray, gray], axis=-1)


# =============================================================================
# Mask Fixtures
# =============================================================================

@pytest.fixture
def sample_binary_mask():
    """Generate a sparse binary mask (~3% coverage) simulating a particle."""
    mask = np.zeros((1024, 1024), dtype=bool)
    # Create circular region (typical particle shape)
    y, x = np.ogrid[:1024, :1024]
    center = (512, 512)
    radius = 50
    mask[(x - center[0])**2 + (y - center[1])**2 <= radius**2] = True
    return mask


@pytest.fixture
def sample_4k_sparse_mask():
    """Generate a sparse 4K mask (~1% coverage) for overlay benchmarks."""
    mask = np.zeros((4096, 4096), dtype=bool)
    y, x = np.ogrid[:4096, :4096]
    # Single particle
    mask[(x - 2048)**2 + (y - 2048)**2 <= 100**2] = True
    return mask


@pytest.fixture
def multiple_masks():
    """Generate multiple non-overlapping masks for testing."""
    masks = []
    centers = [(256, 256), (512, 512), (768, 768), (256, 768), (768, 256)]
    for cx, cy in centers:
        mask = np.zeros((1024, 1024), dtype=bool)
        y, x = np.ogrid[:1024, :1024]
        mask[(x - cx)**2 + (y - cy)**2 <= 40**2] = True
        masks.append(mask)
    return masks


# =============================================================================
# Segmentation Data Fixtures
# =============================================================================

@pytest.fixture
def sample_segmentation_data(sample_binary_mask) -> Dict[str, Any]:
    """Generate a single segmentation data dictionary."""
    return {
        'click_index': 1,
        'click_coords': (512, 512),
        'mask': sample_binary_mask,
        'mask_score': 0.95,
        'mask_area': int(np.sum(sample_binary_mask)),
        'label': None
    }


@pytest.fixture
def sample_segmentations(multiple_masks) -> List[Dict[str, Any]]:
    """Generate multiple segmentation data dictionaries."""
    centers = [(256, 256), (512, 512), (768, 768), (256, 768), (768, 256)]
    segmentations = []
    for i, (mask, (cx, cy)) in enumerate(zip(multiple_masks, centers)):
        segmentations.append({
            'click_index': i + 1,
            'click_coords': (cx, cy),
            'mask': mask,
            'mask_score': 0.9 - (i * 0.05),
            'mask_area': int(np.sum(mask)),
            'label': i if i < 3 else None  # Some labeled, some not
        })
    return segmentations


@pytest.fixture
def labeled_segmentations(sample_segmentations) -> List[Dict[str, Any]]:
    """Generate segmentations with all labels assigned."""
    for i, seg in enumerate(sample_segmentations):
        seg['label'] = i % 3
    return sample_segmentations


# =============================================================================
# File System Fixtures
# =============================================================================

@pytest.fixture
def temp_output_dir():
    """Create temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_image_file(temp_output_dir, sample_grayscale_image):
    """Create a temporary image file for testing."""
    import cv2
    img_path = temp_output_dir / "test_image.png"
    # Normalize to 8-bit for saving
    img_8bit = ((sample_grayscale_image / 65535.0) * 255).astype(np.uint8)
    cv2.imwrite(str(img_path), img_8bit)
    return img_path


@pytest.fixture
def temp_metadata_file(temp_output_dir, sample_segmentations):
    """Create a temporary metadata JSON file."""
    metadata_path = temp_output_dir / "metadata.json"
    # Prepare serializable metadata (without mask arrays)
    metadata = {
        'image_name': 'test_image.png',
        'total_segmentations': len(sample_segmentations),
        'segmentations': [
            {
                'click_index': s['click_index'],
                'click_coords': list(s['click_coords']),
                'mask_score': s['mask_score'],
                'mask_area': s['mask_area'],
                'label': s['label'],
                'mask_file': f"mask_{s['click_index']:03d}_binary.png"
            }
            for s in sample_segmentations
        ]
    }
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    return metadata_path


# =============================================================================
# Mock Fixtures
# =============================================================================

@pytest.fixture
def mock_sam_predictor():
    """Mock SAM predictor for unit tests (avoids loading heavy model)."""
    mock = MagicMock()

    def mock_predict(point_coords, point_labels, multimask_output=True):
        # Return 3 masks of decreasing quality
        h, w = 1024, 1024
        masks = np.zeros((3, h, w), dtype=bool)

        # Create circular masks centered on the click point
        x, y = int(point_coords[0][0]), int(point_coords[0][1])
        Y, X = np.ogrid[:h, :w]

        for i, radius in enumerate([50, 40, 30]):
            masks[i] = ((X - x)**2 + (Y - y)**2) <= radius**2

        scores = np.array([0.95, 0.85, 0.75])
        logits = np.zeros((3, 256, 256))

        return masks, scores, logits

    mock.predict = mock_predict
    return mock


@pytest.fixture
def mock_sam_model(mock_sam_predictor):
    """Mock SAMModel class instance."""
    mock = MagicMock()
    mock.predictor = mock_sam_predictor
    mock.device = "cpu"
    mock.sam = MagicMock()
    mock.sam.training = False
    return mock


# =============================================================================
# Configuration Fixtures
# =============================================================================

@pytest.fixture
def sample_config():
    """Sample configuration dictionary."""
    return {
        'sam': {
            'model_type': 'vit_b',
            'checkpoint_path': '/path/to/checkpoint.pth'
        },
        'output': {
            'save_masks': True,
            'save_overview': True
        },
        'image': {
            'normalize_percentile_low': 1,
            'normalize_percentile_high': 99
        }
    }


@pytest.fixture
def temp_config_file(temp_output_dir, sample_config):
    """Create a temporary config YAML file."""
    import yaml
    config_path = temp_output_dir / "config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(sample_config, f)
    return config_path


# =============================================================================
# Utility Fixtures
# =============================================================================

@pytest.fixture
def color_palette():
    """Generate a color palette for testing."""
    from cryoem_annotation.core.colors import generate_label_colors
    return generate_label_colors(10)


# =============================================================================
# Skip Markers
# =============================================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: mark test as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: mark test as requiring GPU"
    )
    config.addinivalue_line(
        "markers", "benchmark: mark test as a performance benchmark"
    )
