"""Tests for labeling module."""

import pytest
import numpy as np

from cryoem_annotation.labeling.labeler import (
    get_contour_from_mask,
    point_in_mask,
    CachedSegmentationData,
)


class TestGetContourFromMask:
    """Tests for get_contour_from_mask function."""

    def test_returns_contour_for_valid_mask(self, sample_binary_mask):
        """Test that a contour is returned for a valid mask."""
        contour = get_contour_from_mask(sample_binary_mask)

        assert contour is not None
        assert isinstance(contour, np.ndarray)
        assert contour.shape[1] == 2  # (N, 2) for x, y coordinates

    def test_returns_none_for_none_mask(self):
        """Test that None is returned for None input."""
        result = get_contour_from_mask(None)
        assert result is None

    def test_returns_none_for_empty_mask(self):
        """Test that None is returned for empty (all False) mask."""
        empty_mask = np.zeros((100, 100), dtype=bool)
        result = get_contour_from_mask(empty_mask)
        assert result is None


class TestPointInMask:
    """Tests for point_in_mask function."""

    def test_point_inside_mask(self, sample_binary_mask):
        """Test that a point inside the mask returns True."""
        # The sample mask is centered at (512, 512)
        result = point_in_mask((512, 512), sample_binary_mask)
        assert result is True

    def test_point_outside_mask(self, sample_binary_mask):
        """Test that a point outside the mask returns False."""
        result = point_in_mask((0, 0), sample_binary_mask)
        assert result is False

    def test_point_out_of_bounds(self, sample_binary_mask):
        """Test that out-of-bounds points return False."""
        result = point_in_mask((10000, 10000), sample_binary_mask)
        assert result is False

    def test_none_mask_returns_false(self):
        """Test that None mask returns False."""
        result = point_in_mask((100, 100), None)
        assert result is False


class TestCachedSegmentationData:
    """Tests for CachedSegmentationData class."""

    def test_contour_is_cached(self, sample_binary_mask):
        """Test that contour is computed once and cached."""
        cached = CachedSegmentationData(sample_binary_mask)

        # First access computes
        contour1 = cached.contour
        assert cached._computed_contour is True

        # Second access returns cached value
        contour2 = cached.contour
        assert contour1 is contour2  # Same object

    def test_centroid_is_cached(self, sample_binary_mask):
        """Test that centroid is computed once and cached."""
        cached = CachedSegmentationData(sample_binary_mask)

        # First access computes
        centroid1 = cached.centroid
        assert cached._computed_centroid is True

        # Second access returns cached value
        centroid2 = cached.centroid
        assert centroid1 == centroid2

    def test_centroid_near_mask_center(self, sample_binary_mask):
        """Test that centroid is near the mask center."""
        cached = CachedSegmentationData(sample_binary_mask)
        centroid = cached.centroid

        # Sample mask is centered at (512, 512) with radius 50
        assert centroid is not None
        cx, cy = centroid
        assert abs(cx - 512) < 5
        assert abs(cy - 512) < 5

    def test_handles_none_mask(self):
        """Test handling of None mask."""
        cached = CachedSegmentationData(None)

        assert cached.contour is None
        assert cached.centroid is None
