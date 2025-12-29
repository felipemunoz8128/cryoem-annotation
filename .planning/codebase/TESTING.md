# Testing Patterns

**Analysis Date:** 2025-12-29

## Test Framework

**Runner:**
- Pytest 7.0+ (specified in `pyproject.toml`)
- pytest-cov 4.0+ for coverage

**Assertion Library:**
- Pytest built-in assert
- numpy.testing for array comparisons

**Run Commands:**
```bash
pytest tests/ -v                           # Run all tests
pytest tests/test_core/ -v                 # Single directory
pytest tests/test_core/test_image_loader.py -v  # Single file
pytest tests/test_core/test_image_loader.py::TestLoadMicrograph::test_load_png_image -v  # Single test
pytest --cov=cryoem_annotation tests/      # With coverage
```

## Test File Organization

**Location:**
- `tests/` directory at project root (separate from source)
- Structure mirrors `cryoem_annotation/` package

**Naming:**
- `test_*.py` for all test files
- `bench_*.py` for benchmark files in `tests/benchmarks/`

**Structure:**
```
tests/
├── conftest.py              # Shared fixtures (276 lines)
├── test_core/
│   ├── test_image_loader.py
│   └── test_image_processing.py
├── test_annotation/
│   └── test_annotator.py
├── test_labeling/
│   ├── test_labeler.py
│   └── test_categories.py
├── test_extraction/
│   └── test_extractor.py
├── test_io/
│   └── test_metadata.py
└── benchmarks/
    ├── conftest.py          # Benchmark fixtures
    ├── bench_contour_caching.py
    ├── bench_image_processing.py
    └── bench_mask_overlay.py
```

## Test Structure

**Suite Organization:**
```python
import pytest
import numpy as np

class TestLoadMicrograph:
    """Tests for load_micrograph function."""

    def test_load_png_image(self, temp_image_file):
        """Should load PNG as grayscale array."""
        result = load_micrograph(temp_image_file)
        assert result is not None
        assert result.ndim == 2

    def test_nonexistent_file(self, tmp_path):
        """Should return None for missing files."""
        result = load_micrograph(tmp_path / "missing.png")
        assert result is None

    def test_returns_normalized_uint8(self, temp_image_file):
        """Should return uint8 normalized image."""
        result = load_micrograph(temp_image_file)
        assert result.dtype == np.uint8
```

**Patterns:**
- Class-based grouping by function under test
- Method names: `test_<behavior_description>`
- Use fixtures for test data (avoid inline data creation)
- One assertion focus per test (multiple asserts OK if related)

## Mocking

**Framework:**
- pytest fixtures for dependency injection
- unittest.mock when needed for complex mocks

**Patterns:**
```python
# tests/conftest.py - Mock SAM predictor
@pytest.fixture
def mock_sam_predictor():
    """Mock SAM predictor to avoid loading real model."""
    predictor = MagicMock()
    predictor.set_image = MagicMock()
    predictor.predict = MagicMock(return_value=(
        np.zeros((3, 100, 100), dtype=bool),  # masks
        np.array([0.9, 0.8, 0.7]),            # scores
        np.zeros((3, 4))                       # logits
    ))
    return predictor
```

**What to Mock:**
- SAM model (heavy, requires checkpoint)
- File system when testing logic not I/O
- matplotlib display for headless testing

**What NOT to Mock:**
- Pure functions (test directly)
- numpy operations (use real arrays)
- JSON/metadata serialization (test actual format)

## Fixtures and Factories

**Shared Fixtures** (`tests/conftest.py`):

```python
# Image fixtures
@pytest.fixture
def sample_grayscale_image():
    """1024x1024 uint16 test image."""
    return np.random.randint(0, 65535, (1024, 1024), dtype=np.uint16)

@pytest.fixture
def sample_normalized_image():
    """uint8 normalized version for SAM input."""
    return np.random.randint(0, 255, (1024, 1024), dtype=np.uint8)

# Mask fixtures
@pytest.fixture
def sample_binary_mask():
    """Single circular mask centered in image."""
    mask = np.zeros((1024, 1024), dtype=bool)
    cv2.circle(mask.astype(np.uint8), (512, 512), 100, 1, -1)
    return mask.astype(bool)

# File system fixtures
@pytest.fixture
def temp_output_dir(tmp_path):
    """Temporary output directory."""
    output = tmp_path / "output"
    output.mkdir()
    return output

# Segmentation fixtures
@pytest.fixture
def sample_segmentations():
    """5 segmentations with mixed labels."""
    return [
        {"click_index": i, "click_coords": [100*i, 100*i],
         "mask_score": 0.95-i*0.1, "mask_area": 5000+i*100,
         "label": i if i < 3 else None}
        for i in range(5)
    ]
```

**Location:**
- Shared fixtures: `tests/conftest.py`
- Module-specific fixtures: `tests/test_*/conftest.py` (if needed)
- Factory functions: Inline in fixtures or conftest

## Coverage

**Requirements:**
- No enforced coverage target (not in CI)
- Focus on core utilities and I/O

**Configuration:**
- pytest-cov for coverage reports
- Run: `pytest --cov=cryoem_annotation tests/`

**View Coverage:**
```bash
pytest --cov=cryoem_annotation --cov-report=html tests/
open htmlcov/index.html
```

## Test Types

**Unit Tests:**
- Test single function/class in isolation
- Mock external dependencies (SAM, file system)
- Fast: <100ms per test
- Examples: `test_image_loader.py`, `test_metadata.py`

**Integration Tests:**
- Test module interactions
- Use real data structures, mock only external systems
- Examples: `test_extractor.py` (reads metadata, calculates metrics)

**Benchmark Tests:**
- Performance comparisons in `tests/benchmarks/`
- Custom `BenchmarkResult` dataclass with timing/memory metrics
- Run separately: `pytest tests/benchmarks/ -v`

**Example benchmark:**
```python
# tests/benchmarks/bench_contour_caching.py
def test_cached_vs_uncached_redraw(benchmark_masks):
    """Compare cached vs uncached contour calculation."""
    # Uncached timing
    uncached_result = BenchmarkResult.run(
        lambda: [get_contour_from_mask(m) for m in benchmark_masks],
        iterations=10
    )

    # Cached timing
    cached_data = [CachedSegmentationData(m) for m in benchmark_masks]
    cached_result = BenchmarkResult.run(
        lambda: [c.contour for c in cached_data],
        iterations=10
    )

    comparison = uncached_result.compare(cached_result)
    assert comparison['time_speedup'] > 10
```

## Common Patterns

**Async Testing:**
```python
# Not currently used - synchronous codebase
```

**Error Testing:**
```python
def test_invalid_model_type(self):
    """Should raise ValueError for unknown model type."""
    with pytest.raises(ValueError, match="Invalid model type"):
        SAMModel(model_type="invalid", checkpoint_path="test.pth")

def test_missing_checkpoint(self, tmp_path):
    """Should raise FileNotFoundError for missing checkpoint."""
    with pytest.raises(FileNotFoundError):
        SAMModel(model_type="vit_b", checkpoint_path=tmp_path / "missing.pth")
```

**Parametrized Tests:**
```python
@pytest.mark.parametrize("extension", [".png", ".jpg", ".tif", ".mrc"])
def test_supported_extensions(self, tmp_path, extension):
    """Should accept all supported image formats."""
    test_file = tmp_path / f"test{extension}"
    # Create test file...
    assert is_supported_format(test_file)
```

**Custom Markers:**
```python
# tests/conftest.py
def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "gpu: marks tests requiring GPU")
    config.addinivalue_line("markers", "benchmark: marks benchmark tests")

# Usage
@pytest.mark.slow
def test_full_annotation_workflow(self):
    ...

@pytest.mark.gpu
def test_sam_prediction_on_gpu(self):
    ...
```

---

*Testing analysis: 2025-12-29*
*Update when test patterns change*
