"""Benchmarks for image processing operations.

These benchmarks measure performance of image normalization and other
processing operations in image_processing.py.
"""

import pytest
import numpy as np

from .conftest import benchmark_test


class TestNormalizeImageBaseline:
    """Baseline benchmarks for current image normalization implementation."""

    @benchmark_test
    def test_normalize_1k(self, sample_grayscale_image, benchmark):
        """Baseline: Normalize 1K image."""
        from cryoem_annotation.core.image_processing import normalize_image

        img = sample_grayscale_image

        def normalize():
            return normalize_image(img)

        result = benchmark.run(normalize, "normalize_1k")
        print(f"\n{result}")
        print(f"Input dtype: {img.dtype}, shape: {img.shape}")

    @benchmark_test
    def test_normalize_4k(self, benchmark_4k_image, benchmark):
        """Baseline: Normalize 4K image."""
        from cryoem_annotation.core.image_processing import normalize_image

        img = benchmark_4k_image

        def normalize():
            return normalize_image(img)

        result = benchmark.run(normalize, "normalize_4k")
        print(f"\n{result}")
        print(f"Input dtype: {img.dtype}, shape: {img.shape}")

    @pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32, np.float64])
    @benchmark_test
    def test_normalize_various_dtypes(self, dtype, benchmark):
        """Test normalization performance with various input dtypes."""
        from cryoem_annotation.core.image_processing import normalize_image

        np.random.seed(42)
        if dtype in (np.float32, np.float64):
            img = np.random.random((1024, 1024)).astype(dtype)
        else:
            max_val = np.iinfo(dtype).max
            img = np.random.randint(0, max_val, (1024, 1024), dtype=dtype)

        def normalize():
            return normalize_image(img)

        result = benchmark.run(normalize, f"normalize_{dtype.__name__}")
        print(f"\n{result}")


class TestOptimizedNormalization:
    """Benchmarks comparing current vs optimized normalization."""

    @staticmethod
    def normalize_optimized(img: np.ndarray,
                           percentile: tuple = (1, 99)) -> np.ndarray:
        """Optimized normalization avoiding redundant dtype conversions."""
        # Convert to float32 for processing
        img_float = img.astype(np.float32)

        p_low, p_high = np.percentile(img_float, percentile)

        if p_high > p_low:
            # Scale and clip in combined operation
            img_float = np.clip((img_float - p_low) / (p_high - p_low), 0, 1)
        else:
            img_float = np.zeros_like(img_float)

        return (img_float * 255).astype(np.uint8)

    @benchmark_test
    def test_compare_normalize_1k(self, sample_grayscale_image, benchmark):
        """Compare original vs optimized normalization on 1K image."""
        from cryoem_annotation.core.image_processing import normalize_image

        img = sample_grayscale_image

        def original():
            return normalize_image(img)

        def optimized():
            return TestOptimizedNormalization.normalize_optimized(img)

        comparison = benchmark.compare(original, optimized)

        print(f"\nOriginal: {comparison['baseline']}")
        print(f"Optimized: {comparison['optimized']}")
        print(f"Time improvement: {comparison['time_improvement_pct']:.1f}%")

        # Note: Correctness verified in external tests - benchmark focuses on timing

    @benchmark_test
    def test_compare_normalize_4k(self, benchmark_4k_image, benchmark):
        """Compare original vs optimized normalization on 4K image."""
        from cryoem_annotation.core.image_processing import normalize_image

        img = benchmark_4k_image

        def original():
            return normalize_image(img)

        def optimized():
            return TestOptimizedNormalization.normalize_optimized(img)

        comparison = benchmark.compare(original, optimized)

        print(f"\nOriginal: {comparison['baseline']}")
        print(f"Optimized: {comparison['optimized']}")
        print(f"Time improvement: {comparison['time_improvement_pct']:.1f}%")


class TestColorGeneration:
    """Benchmarks for color palette generation."""

    @benchmark_test
    def test_generate_10_colors(self, benchmark):
        """Baseline: Generate 10 colors."""
        from cryoem_annotation.core.colors import generate_label_colors

        def generate():
            return generate_label_colors(10)

        result = benchmark.run(generate, "generate_10_colors")
        print(f"\n{result}")

    @benchmark_test
    def test_generate_50_colors(self, benchmark):
        """Baseline: Generate 50 colors (current click_collector default)."""
        from cryoem_annotation.core.colors import generate_label_colors

        def generate():
            return generate_label_colors(50)

        result = benchmark.run(generate, "generate_50_colors")
        print(f"\n{result}")

    @benchmark_test
    def test_vectorized_color_generation(self, benchmark):
        """Test vectorized color generation optimization."""
        from matplotlib.colors import hsv_to_rgb

        def current_approach(num_colors: int = 50):
            """Current per-color generation."""
            colors = []
            for i in range(num_colors):
                hue = (i * 0.618034) % 1.0
                saturation = 0.7 + (i % 3) * 0.1
                value = 0.9
                rgb = hsv_to_rgb([[hue, saturation, value]])[0]
                colors.append(rgb)
            return colors

        def vectorized_approach(num_colors: int = 50):
            """Vectorized batch generation."""
            indices = np.arange(num_colors)
            hues = (indices * 0.618034) % 1.0
            saturations = 0.7 + (indices % 3) * 0.1
            values = np.full(num_colors, 0.9)

            hsv = np.stack([hues, saturations, values], axis=1)
            rgb = hsv_to_rgb(hsv)

            return [rgb[i] for i in range(num_colors)]

        comparison = benchmark.compare(
            lambda: current_approach(50),
            lambda: vectorized_approach(50),
            "per_color", "vectorized"
        )

        print(f"\nPer-color: {comparison['baseline']}")
        print(f"Vectorized: {comparison['optimized']}")
        print(f"Speedup: {comparison['time_speedup']:.2f}x")


class TestImageLoading:
    """Benchmarks for image loading operations."""

    @benchmark_test
    def test_glob_patterns(self, temp_output_dir, benchmark):
        """Test file discovery performance."""
        import cv2
        from pathlib import Path

        # Create test files
        extensions = ['.png', '.tif', '.mrc', '.PNG', '.TIF']
        for i, ext in enumerate(extensions * 5):
            img = np.zeros((100, 100), dtype=np.uint8)
            if ext.lower() in ['.png', '.tif']:
                cv2.imwrite(str(temp_output_dir / f"test_{i}{ext}"), img)

        folder = temp_output_dir

        def current_approach():
            """Current double-glob per extension."""
            extensions = {'.mrc', '.tif', '.tiff', '.png', '.jpg', '.jpeg'}
            files = []
            for ext in extensions:
                files.extend(folder.glob(f"*{ext}"))
                files.extend(folder.glob(f"*{ext.upper()}"))
            return sorted(files)

        def optimized_approach():
            """Single iterdir pass."""
            extensions = {'.mrc', '.tif', '.tiff', '.png', '.jpg', '.jpeg'}
            files = []
            for item in folder.iterdir():
                if item.is_file() and not item.name.startswith('.'):
                    if item.suffix.lower() in extensions:
                        files.append(item)
            return sorted(files)

        comparison = benchmark.compare(current_approach, optimized_approach,
                                       "double_glob", "single_iterdir")

        print(f"\nDouble glob: {comparison['baseline']}")
        print(f"Single iterdir: {comparison['optimized']}")
        print(f"Speedup: {comparison['time_speedup']:.2f}x")

        # Verify same results
        assert set(current_approach()) == set(optimized_approach())
