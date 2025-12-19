"""Benchmark utilities and fixtures for performance testing."""

import pytest
import time
import tracemalloc
import gc
from dataclasses import dataclass, field
from typing import Callable, Optional, List
import numpy as np
from functools import wraps


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    name: str
    time_ms: float
    memory_peak_mb: float
    memory_current_mb: float
    iterations: int
    time_std_ms: float = 0.0
    all_times_ms: List[float] = field(default_factory=list)

    def __str__(self):
        return (
            f"{self.name}: "
            f"time={self.time_ms:.2f}ms (std={self.time_std_ms:.2f}ms), "
            f"memory_peak={self.memory_peak_mb:.2f}MB"
        )


class BenchmarkRunner:
    """Runner for timing and memory benchmarks."""

    def __init__(self, warmup: int = 2, iterations: int = 10):
        self.warmup = warmup
        self.iterations = iterations

    def run(self, func: Callable, name: Optional[str] = None) -> BenchmarkResult:
        """Run benchmark with timing and memory measurement.

        Args:
            func: Function to benchmark (should take no arguments)
            name: Optional name for the benchmark

        Returns:
            BenchmarkResult with timing and memory data
        """
        name = name or getattr(func, '__name__', 'benchmark')

        # Force garbage collection before benchmarking
        gc.collect()

        # Warmup runs
        for _ in range(self.warmup):
            func()

        # Memory measurement (single run)
        gc.collect()
        tracemalloc.start()
        func()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        memory_current_mb = current / (1024 * 1024)
        memory_peak_mb = peak / (1024 * 1024)

        # Timing runs
        times = []
        for _ in range(self.iterations):
            gc.collect()
            start = time.perf_counter()
            func()
            elapsed = (time.perf_counter() - start) * 1000  # Convert to ms
            times.append(elapsed)

        return BenchmarkResult(
            name=name,
            time_ms=np.median(times),
            time_std_ms=np.std(times),
            memory_peak_mb=memory_peak_mb,
            memory_current_mb=memory_current_mb,
            iterations=self.iterations,
            all_times_ms=times
        )

    def compare(self, baseline_func: Callable, optimized_func: Callable,
                baseline_name: str = "baseline",
                optimized_name: str = "optimized") -> dict:
        """Compare baseline vs optimized implementation.

        Returns:
            dict with both results and improvement ratios
        """
        baseline = self.run(baseline_func, baseline_name)
        optimized = self.run(optimized_func, optimized_name)

        time_improvement = (baseline.time_ms - optimized.time_ms) / baseline.time_ms * 100
        memory_improvement = (baseline.memory_peak_mb - optimized.memory_peak_mb) / baseline.memory_peak_mb * 100

        return {
            'baseline': baseline,
            'optimized': optimized,
            'time_improvement_pct': time_improvement,
            'memory_improvement_pct': memory_improvement,
            'time_speedup': baseline.time_ms / optimized.time_ms if optimized.time_ms > 0 else float('inf'),
            'memory_reduction': baseline.memory_peak_mb / optimized.memory_peak_mb if optimized.memory_peak_mb > 0 else float('inf')
        }


@pytest.fixture
def benchmark():
    """Fixture providing a BenchmarkRunner instance."""
    return BenchmarkRunner(warmup=2, iterations=10)


@pytest.fixture
def quick_benchmark():
    """Faster benchmark runner for CI/development."""
    return BenchmarkRunner(warmup=1, iterations=5)


@pytest.fixture
def thorough_benchmark():
    """More thorough benchmark runner for accurate measurements."""
    return BenchmarkRunner(warmup=5, iterations=20)


# =============================================================================
# Benchmark-specific image fixtures (larger sizes for realistic testing)
# =============================================================================

@pytest.fixture
def benchmark_4k_image():
    """4096x4096 image for benchmarking."""
    np.random.seed(42)
    return np.random.randint(0, 65535, (4096, 4096), dtype=np.uint16)


@pytest.fixture
def benchmark_4k_mask():
    """4K sparse mask (~2% coverage) for overlay benchmarks."""
    mask = np.zeros((4096, 4096), dtype=bool)
    y, x = np.ogrid[:4096, :4096]
    # Create a circular particle mask
    mask[(x - 2048)**2 + (y - 2048)**2 <= 150**2] = True
    return mask


@pytest.fixture
def benchmark_multi_masks():
    """Multiple 4K masks for batch operations."""
    masks = []
    centers = [(1000, 1000), (2048, 2048), (3000, 3000),
               (1000, 3000), (3000, 1000)]
    for cx, cy in centers:
        mask = np.zeros((4096, 4096), dtype=bool)
        y, x = np.ogrid[:4096, :4096]
        mask[(x - cx)**2 + (y - cy)**2 <= 100**2] = True
        masks.append(mask)
    return masks


# =============================================================================
# Decorator for marking benchmark tests
# =============================================================================

def benchmark_test(func):
    """Decorator to mark a function as a benchmark test."""
    @wraps(func)
    @pytest.mark.benchmark
    @pytest.mark.slow
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


# =============================================================================
# Utility functions for benchmark assertions
# =============================================================================

def assert_memory_improved(baseline_mb: float, optimized_mb: float,
                           min_improvement_pct: float = 50):
    """Assert that memory usage improved by at least the specified percentage."""
    improvement = (baseline_mb - optimized_mb) / baseline_mb * 100
    assert improvement >= min_improvement_pct, (
        f"Memory improvement {improvement:.1f}% is less than required {min_improvement_pct}%"
    )


def assert_time_improved(baseline_ms: float, optimized_ms: float,
                         min_improvement_pct: float = 25):
    """Assert that execution time improved by at least the specified percentage."""
    improvement = (baseline_ms - optimized_ms) / baseline_ms * 100
    assert improvement >= min_improvement_pct, (
        f"Time improvement {improvement:.1f}% is less than required {min_improvement_pct}%"
    )
