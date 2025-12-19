"""Benchmarks for mask overlay operations.

These benchmarks establish baseline performance for the dense RGBA overlay
approach currently used in click_collector.py and annotator.py.

The goal is to demonstrate that sparse/bounded overlays significantly
reduce memory usage for typical cryo-EM segmentation masks.
"""

import pytest
import numpy as np

from .conftest import benchmark_test, assert_memory_improved


class TestDenseOverlayBaseline:
    """Baseline benchmarks for the current dense RGBA overlay approach."""

    @benchmark_test
    def test_dense_overlay_1k(self, sample_binary_mask, benchmark):
        """Baseline: Dense RGBA overlay for 1K image."""
        mask = sample_binary_mask

        def create_dense_overlay():
            overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
            overlay[mask] = [1.0, 0.0, 0.0, 0.4]
            return overlay

        result = benchmark.run(create_dense_overlay, "dense_overlay_1k")

        # Report results
        print(f"\n{result}")
        mask_coverage = np.sum(mask) / mask.size * 100
        print(f"Mask coverage: {mask_coverage:.2f}%")

        # 1K image * 4 channels * 4 bytes = 16MB expected
        assert result.memory_peak_mb > 10, "Expected significant memory for 1K overlay"

    @benchmark_test
    def test_dense_overlay_4k(self, benchmark_4k_mask, benchmark):
        """Baseline: Dense RGBA overlay for 4K image - the problem case."""
        mask = benchmark_4k_mask

        def create_dense_overlay():
            overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
            overlay[mask] = [1.0, 0.0, 0.0, 0.4]
            return overlay

        result = benchmark.run(create_dense_overlay, "dense_overlay_4k")

        # Report results
        print(f"\n{result}")
        mask_coverage = np.sum(mask) / mask.size * 100
        print(f"Mask coverage: {mask_coverage:.2f}%")

        # 4K image: 4096*4096*4*4 bytes = 256MB expected
        expected_mb = 4096 * 4096 * 4 * 4 / (1024 * 1024)
        print(f"Expected memory: {expected_mb:.0f}MB")

        # Verify the problem exists (we allocate ~256MB for sparse mask)
        assert result.memory_peak_mb > 200, (
            f"Expected >200MB for 4K overlay, got {result.memory_peak_mb:.1f}MB"
        )

    @benchmark_test
    def test_dense_overlay_multiple_masks(self, benchmark_multi_masks, benchmark):
        """Baseline: Multiple dense overlays (simulates annotation session)."""
        masks = benchmark_multi_masks

        def create_multiple_overlays():
            overlays = []
            colors = [
                [1.0, 0.0, 0.0, 0.4],
                [0.0, 1.0, 0.0, 0.4],
                [0.0, 0.0, 1.0, 0.4],
                [1.0, 1.0, 0.0, 0.4],
                [1.0, 0.0, 1.0, 0.4],
            ]
            for mask, color in zip(masks, colors):
                overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
                overlay[mask] = color
                overlays.append(overlay)
            return overlays

        result = benchmark.run(create_multiple_overlays, "dense_overlay_multiple")

        print(f"\n{result}")
        print(f"Number of masks: {len(masks)}")

        # 5 masks * 256MB each = 1.28GB total at peak
        # (though GC may reduce this)
        assert result.memory_peak_mb > 400, (
            f"Expected >400MB for multiple overlays, got {result.memory_peak_mb:.1f}MB"
        )


class TestBoundedOverlayOptimized:
    """Benchmarks for the optimized bounded overlay approach.

    These tests use the same create_bounded_overlay function that will be
    implemented in click_collector.py. We define it here first to validate
    the optimization before integrating it.
    """

    @staticmethod
    def create_bounded_overlay(mask: np.ndarray, color: list) -> tuple:
        """Create overlay only for mask bounding box region.

        This is the optimized implementation that will replace the dense approach.

        Returns:
            tuple: (overlay_array, extent) for ax.imshow()
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

    @benchmark_test
    def test_bounded_overlay_1k(self, sample_binary_mask, benchmark):
        """Optimized: Bounded RGBA overlay for 1K image."""
        mask = sample_binary_mask
        color = [1.0, 0.0, 0.0, 0.4]

        def create_overlay():
            return self.create_bounded_overlay(mask, color)

        result = benchmark.run(create_overlay, "bounded_overlay_1k")

        print(f"\n{result}")
        overlay, extent = self.create_bounded_overlay(mask, color)
        print(f"Bounded size: {overlay.shape if overlay is not None else 'None'}")

    @benchmark_test
    def test_bounded_overlay_4k(self, benchmark_4k_mask, benchmark):
        """Optimized: Bounded RGBA overlay for 4K image."""
        mask = benchmark_4k_mask
        color = [1.0, 0.0, 0.0, 0.4]

        def create_overlay():
            return self.create_bounded_overlay(mask, color)

        result = benchmark.run(create_overlay, "bounded_overlay_4k")

        print(f"\n{result}")
        overlay, extent = self.create_bounded_overlay(mask, color)
        if overlay is not None:
            print(f"Bounded size: {overlay.shape}")
            bounded_mb = overlay.nbytes / (1024 * 1024)
            dense_mb = mask.shape[0] * mask.shape[1] * 4 * 4 / (1024 * 1024)
            print(f"Bounded memory: {bounded_mb:.2f}MB vs Dense: {dense_mb:.0f}MB")
            print(f"Memory reduction: {(1 - bounded_mb/dense_mb)*100:.1f}%")

        # Should be much smaller than 256MB
        assert result.memory_peak_mb < 50, (
            f"Bounded overlay should use <50MB, got {result.memory_peak_mb:.1f}MB"
        )

    @benchmark_test
    def test_compare_dense_vs_bounded_4k(self, benchmark_4k_mask, benchmark):
        """Direct comparison: Dense vs Bounded for 4K mask."""
        mask = benchmark_4k_mask
        color = [1.0, 0.0, 0.0, 0.4]

        def dense_approach():
            overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
            overlay[mask] = color
            return overlay

        def bounded_approach():
            return self.create_bounded_overlay(mask, color)

        comparison = benchmark.compare(dense_approach, bounded_approach,
                                       "dense_4k", "bounded_4k")

        print(f"\nBaseline (dense): {comparison['baseline']}")
        print(f"Optimized (bounded): {comparison['optimized']}")
        print(f"Memory improvement: {comparison['memory_improvement_pct']:.1f}%")
        print(f"Memory reduction factor: {comparison['memory_reduction']:.1f}x")

        # Assert significant improvement
        assert_memory_improved(
            comparison['baseline'].memory_peak_mb,
            comparison['optimized'].memory_peak_mb,
            min_improvement_pct=80  # Expect >80% memory reduction
        )


class TestMaskCoverageImpact:
    """Test how mask coverage affects memory savings."""

    @pytest.mark.parametrize("coverage_pct", [1, 5, 10, 25, 50])
    @benchmark_test
    def test_coverage_impact(self, coverage_pct, benchmark):
        """Test memory impact at different mask coverage levels."""
        # Create mask with specified coverage
        size = 2048
        mask = np.zeros((size, size), dtype=bool)

        # Create circular mask with appropriate radius for coverage
        target_pixels = int(size * size * coverage_pct / 100)
        radius = int(np.sqrt(target_pixels / np.pi))

        y, x = np.ogrid[:size, :size]
        center = size // 2
        mask[(x - center)**2 + (y - center)**2 <= radius**2] = True

        actual_coverage = np.sum(mask) / mask.size * 100
        color = [1.0, 0.0, 0.0, 0.4]

        def dense_overlay():
            overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
            overlay[mask] = color
            return overlay

        def bounded_overlay():
            return TestBoundedOverlayOptimized.create_bounded_overlay(mask, color)

        comparison = benchmark.compare(dense_overlay, bounded_overlay)

        print(f"\nCoverage: {actual_coverage:.1f}%")
        print(f"Dense: {comparison['baseline'].memory_peak_mb:.1f}MB")
        print(f"Bounded: {comparison['optimized'].memory_peak_mb:.1f}MB")
        print(f"Reduction: {comparison['memory_improvement_pct']:.1f}%")

        # For low coverage (<10%), expect major improvement
        if coverage_pct < 10:
            assert comparison['memory_improvement_pct'] > 50
