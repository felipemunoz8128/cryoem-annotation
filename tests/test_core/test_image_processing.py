"""Tests for image processing module."""

import pytest
import numpy as np

from cryoem_annotation.core.image_processing import normalize_image


class TestNormalizeImage:
    """Tests for normalize_image function."""

    def test_output_is_uint8(self, sample_grayscale_image):
        """Test that output is uint8."""
        result = normalize_image(sample_grayscale_image)
        assert result.dtype == np.uint8

    def test_output_range(self, sample_grayscale_image):
        """Test that output is in range [0, 255]."""
        result = normalize_image(sample_grayscale_image)
        assert result.min() >= 0
        assert result.max() <= 255

    def test_preserves_shape(self, sample_grayscale_image):
        """Test that shape is preserved."""
        result = normalize_image(sample_grayscale_image)
        assert result.shape == sample_grayscale_image.shape

    def test_constant_image(self):
        """Test normalization of constant image."""
        constant = np.full((100, 100), 1000, dtype=np.uint16)
        result = normalize_image(constant)
        # Constant image should normalize to a single value
        assert result.min() == result.max()

    def test_zero_image(self):
        """Test normalization of all-zeros image."""
        zeros = np.zeros((100, 100), dtype=np.uint16)
        result = normalize_image(zeros)
        assert result.dtype == np.uint8

    def test_custom_percentile(self, sample_grayscale_image):
        """Test custom percentile range."""
        result = normalize_image(sample_grayscale_image, percentile=(5, 95))
        assert result.dtype == np.uint8
        assert result.min() >= 0
        assert result.max() <= 255

    def test_float_input(self):
        """Test normalization of float input."""
        float_img = np.random.rand(100, 100).astype(np.float32)
        result = normalize_image(float_img)
        assert result.dtype == np.uint8
