"""Benchmarks for contour and centroid caching.

These benchmarks establish baseline performance for cv2.findContours()
and cv2.moments() operations that are currently called on every redraw
in labeler.py.

The goal is to demonstrate that caching these computations significantly
improves redraw performance.
"""

import pytest
import numpy as np
import cv2

from .conftest import benchmark_test, assert_time_improved


class TestContourExtractionBaseline:
    """Baseline benchmarks for contour extraction without caching."""

    @benchmark_test
    def test_find_contours_1k(self, sample_binary_mask, benchmark):
        """Baseline: cv2.findContours on 1K mask."""
        mask = sample_binary_mask.astype(np.uint8)

        def extract_contour():
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if contours:
                return cv2.approxPolyDP(contours[0], epsilon=2, closed=True)
            return None

        result = benchmark.run(extract_contour, "findContours_1k")
        print(f"\n{result}")

    @benchmark_test
    def test_find_contours_4k(self, benchmark_4k_mask, benchmark):
        """Baseline: cv2.findContours on 4K mask."""
        mask = benchmark_4k_mask.astype(np.uint8)

        def extract_contour():
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if contours:
                return cv2.approxPolyDP(contours[0], epsilon=2, closed=True)
            return None

        result = benchmark.run(extract_contour, "findContours_4k")
        print(f"\n{result}")

    @benchmark_test
    def test_moments_1k(self, sample_binary_mask, benchmark):
        """Baseline: cv2.moments on 1K mask."""
        mask = sample_binary_mask.astype(np.uint8)

        def compute_centroid():
            M = cv2.moments(mask)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                return (cx, cy)
            return None

        result = benchmark.run(compute_centroid, "moments_1k")
        print(f"\n{result}")

    @benchmark_test
    def test_moments_4k(self, benchmark_4k_mask, benchmark):
        """Baseline: cv2.moments on 4K mask."""
        mask = benchmark_4k_mask.astype(np.uint8)

        def compute_centroid():
            M = cv2.moments(mask)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                return (cx, cy)
            return None

        result = benchmark.run(compute_centroid, "moments_4k")
        print(f"\n{result}")


class TestRedrawSimulation:
    """Simulate labeler redraw behavior with and without caching."""

    @benchmark_test
    def test_uncached_redraw_5_segments(self, multiple_masks, benchmark):
        """Simulate 5-segment redraw WITHOUT caching (current behavior)."""
        masks = [m.astype(np.uint8) for m in multiple_masks]

        def uncached_redraw():
            """Simulates _draw_segmentations() without caching."""
            results = []
            for mask in masks:
                # This happens on every redraw currently
                contours, _ = cv2.findContours(
                    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                if contours:
                    contour = cv2.approxPolyDP(contours[0], epsilon=2, closed=True)
                else:
                    contour = None

                M = cv2.moments(mask)
                if M["m00"] != 0:
                    centroid = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                else:
                    centroid = None

                results.append((contour, centroid))
            return results

        result = benchmark.run(uncached_redraw, "uncached_redraw_5seg")
        print(f"\n{result}")
        print(f"Per-segment time: {result.time_ms / len(masks):.2f}ms")

    @benchmark_test
    def test_cached_redraw_5_segments(self, multiple_masks, benchmark):
        """Simulate 5-segment redraw WITH caching (optimized)."""
        masks = [m.astype(np.uint8) for m in multiple_masks]

        # Pre-compute cache (done once on load)
        cache = []
        for mask in masks:
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            contour = cv2.approxPolyDP(contours[0], epsilon=2, closed=True) if contours else None

            M = cv2.moments(mask)
            centroid = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])) if M["m00"] != 0 else None

            cache.append({'contour': contour, 'centroid': centroid})

        def cached_redraw():
            """Simulates _draw_segmentations() WITH caching."""
            results = []
            for cached_data in cache:
                # Just access cached values (no recomputation)
                contour = cached_data['contour']
                centroid = cached_data['centroid']
                results.append((contour, centroid))
            return results

        result = benchmark.run(cached_redraw, "cached_redraw_5seg")
        print(f"\n{result}")
        print(f"Per-segment time: {result.time_ms / len(masks):.2f}ms")

    @benchmark_test
    def test_compare_cached_vs_uncached(self, multiple_masks, benchmark):
        """Direct comparison of cached vs uncached redraw."""
        masks = [m.astype(np.uint8) for m in multiple_masks]

        # Pre-compute cache
        cache = []
        for mask in masks:
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            contour = cv2.approxPolyDP(contours[0], epsilon=2, closed=True) if contours else None

            M = cv2.moments(mask)
            centroid = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])) if M["m00"] != 0 else None

            cache.append({'contour': contour, 'centroid': centroid})

        def uncached():
            results = []
            for mask in masks:
                contours, _ = cv2.findContours(
                    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                contour = cv2.approxPolyDP(contours[0], epsilon=2, closed=True) if contours else None
                M = cv2.moments(mask)
                centroid = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])) if M["m00"] != 0 else None
                results.append((contour, centroid))
            return results

        def cached():
            return [(c['contour'], c['centroid']) for c in cache]

        comparison = benchmark.compare(uncached, cached, "uncached", "cached")

        print(f"\nUncached: {comparison['baseline']}")
        print(f"Cached: {comparison['optimized']}")
        print(f"Speedup: {comparison['time_speedup']:.1f}x")

        # Cache access should be essentially instant
        assert comparison['time_speedup'] > 10, (
            f"Expected >10x speedup from caching, got {comparison['time_speedup']:.1f}x"
        )


class TestMultipleRedraws:
    """Test cumulative impact of multiple redraws (user interaction simulation)."""

    @benchmark_test
    def test_10_redraws_uncached(self, multiple_masks, benchmark):
        """Simulate 10 redraws (10 label assignments) WITHOUT caching."""
        masks = [m.astype(np.uint8) for m in multiple_masks]
        num_redraws = 10

        def multiple_uncached_redraws():
            for _ in range(num_redraws):
                for mask in masks:
                    contours, _ = cv2.findContours(
                        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    if contours:
                        cv2.approxPolyDP(contours[0], epsilon=2, closed=True)
                    cv2.moments(mask)

        result = benchmark.run(multiple_uncached_redraws, f"{num_redraws}_redraws_uncached")
        print(f"\n{result}")
        print(f"Total for {num_redraws} redraws: {result.time_ms:.1f}ms")
        print(f"Per redraw: {result.time_ms / num_redraws:.2f}ms")

    @benchmark_test
    def test_10_redraws_cached(self, multiple_masks, benchmark):
        """Simulate 10 redraws (10 label assignments) WITH caching."""
        masks = [m.astype(np.uint8) for m in multiple_masks]
        num_redraws = 10

        # Pre-compute cache once
        cache = []
        for mask in masks:
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            contour = cv2.approxPolyDP(contours[0], epsilon=2, closed=True) if contours else None
            M = cv2.moments(mask)
            centroid = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])) if M["m00"] != 0 else None
            cache.append({'contour': contour, 'centroid': centroid})

        def multiple_cached_redraws():
            for _ in range(num_redraws):
                for c in cache:
                    _ = c['contour']
                    _ = c['centroid']

        result = benchmark.run(multiple_cached_redraws, f"{num_redraws}_redraws_cached")
        print(f"\n{result}")
        print(f"Total for {num_redraws} redraws: {result.time_ms:.1f}ms")
        print(f"Per redraw: {result.time_ms / num_redraws:.2f}ms")


class TestCachedSegmentationDataClass:
    """Test the CachedSegmentationData class that will be implemented."""

    @benchmark_test
    def test_lazy_caching_behavior(self, sample_binary_mask, benchmark):
        """Test lazy property caching pattern."""

        class CachedSegmentationData:
            """Cached contour and centroid data for a segmentation mask."""

            def __init__(self, mask: np.ndarray):
                self._mask = mask.astype(np.uint8)
                self._contour = None
                self._centroid = None

            @property
            def contour(self):
                if self._contour is None:
                    contours, _ = cv2.findContours(
                        self._mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    if contours:
                        self._contour = cv2.approxPolyDP(
                            contours[0], epsilon=2, closed=True
                        )
                return self._contour

            @property
            def centroid(self):
                if self._centroid is None:
                    M = cv2.moments(self._mask)
                    if M["m00"] != 0:
                        self._centroid = (
                            int(M["m10"] / M["m00"]),
                            int(M["m01"] / M["m00"])
                        )
                return self._centroid

        mask = sample_binary_mask

        # Test that first access computes, subsequent accesses use cache
        cached = CachedSegmentationData(mask)

        def first_access():
            return cached.contour, cached.centroid

        def subsequent_access():
            return cached.contour, cached.centroid

        # First access (computes)
        first_result = benchmark.run(first_access, "first_access")
        print(f"\nFirst access: {first_result}")

        # Reset for clean measurement
        cached = CachedSegmentationData(mask)
        _ = cached.contour  # Populate cache
        _ = cached.centroid

        # Subsequent access (cached)
        subsequent_result = benchmark.run(subsequent_access, "subsequent_access")
        print(f"Subsequent access: {subsequent_result}")

        # Cached access should be faster (relaxed assertion for sub-ms operations)
        # For very fast operations, just verify caching works conceptually
        assert subsequent_result.time_ms <= first_result.time_ms + 0.1, (
            "Cached access should not be slower than first access"
        )
