"""Tests for image loader module."""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import cv2

from cryoem_annotation.core.image_loader import (
    load_micrograph,
    load_micrograph_with_pixel_size,
    get_image_files,
    IMAGE_EXTENSIONS,
)


class TestLoadMicrograph:
    """Tests for load_micrograph function."""

    def test_load_png_image(self, temp_image_file):
        """Test loading a PNG image file."""
        result = load_micrograph(temp_image_file)
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert len(result.shape) == 2  # Grayscale

    def test_load_nonexistent_file(self):
        """Test loading a file that doesn't exist."""
        result = load_micrograph(Path("/nonexistent/path/image.png"))
        assert result is None

    def test_load_creates_grayscale(self, temp_output_dir):
        """Test that loaded images are converted to grayscale."""
        # Create a color image
        color_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        img_path = temp_output_dir / "color_test.png"
        cv2.imwrite(str(img_path), color_img)

        result = load_micrograph(img_path)
        assert result is not None
        assert len(result.shape) == 2  # Should be grayscale


class TestLoadMicrographWithPixelSize:
    """Tests for load_micrograph_with_pixel_size function."""

    def test_returns_none_pixel_size_for_png(self, temp_image_file):
        """Test that PNG files return None for pixel size."""
        data, pixel_size = load_micrograph_with_pixel_size(temp_image_file)
        assert data is not None
        assert pixel_size is None

    def test_nonexistent_file_returns_none_tuple(self):
        """Test that nonexistent files return (None, None)."""
        data, pixel_size = load_micrograph_with_pixel_size(Path("/nonexistent.mrc"))
        assert data is None
        assert pixel_size is None


class TestGetImageFiles:
    """Tests for get_image_files function."""

    def test_finds_supported_extensions(self, temp_output_dir):
        """Test that all supported extensions are found."""
        # Create test files
        for ext in ['.png', '.jpg', '.tif']:
            (temp_output_dir / f"test{ext}").touch()

        files = get_image_files(temp_output_dir)
        assert len(files) == 3

    def test_ignores_hidden_files(self, temp_output_dir):
        """Test that hidden files are ignored."""
        (temp_output_dir / "visible.png").touch()
        (temp_output_dir / ".hidden.png").touch()

        files = get_image_files(temp_output_dir)
        assert len(files) == 1
        assert files[0].name == "visible.png"

    def test_returns_sorted_list(self, temp_output_dir):
        """Test that files are returned sorted."""
        for name in ["c.png", "a.png", "b.png"]:
            (temp_output_dir / name).touch()

        files = get_image_files(temp_output_dir)
        assert [f.name for f in files] == ["a.png", "b.png", "c.png"]

    def test_custom_extensions(self, temp_output_dir):
        """Test filtering by custom extensions."""
        (temp_output_dir / "image.png").touch()
        (temp_output_dir / "image.jpg").touch()

        files = get_image_files(temp_output_dir, extensions={'.png'})
        assert len(files) == 1
        assert files[0].name == "image.png"

    def test_empty_directory(self, temp_output_dir):
        """Test empty directory returns empty list."""
        files = get_image_files(temp_output_dir)
        assert files == []
