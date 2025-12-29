# Coding Conventions

**Analysis Date:** 2025-12-29

## Naming Patterns

**Files:**
- snake_case for all Python files: `sam_model.py`, `image_loader.py`, `click_collector.py`
- test_*.py for test files: `test_image_loader.py`, `test_metadata.py`
- bench_*.py for benchmark files: `bench_contour_caching.py`

**Functions:**
- snake_case for all functions: `load_micrograph()`, `get_image_files()`, `extract_results()`
- No special prefix for async functions (none used currently)
- Private functions prefixed with underscore: `_inference_context()`, `_save_overview_image()`

**Variables:**
- snake_case for variables: `pixel_size_nm`, `mask_area`, `current_file_path`
- UPPER_SNAKE_CASE for module constants: `IMAGE_EXTENSIONS`, `VALID_MODEL_TYPES`
- Underscore prefix for private attributes: `_computed_contour`, `_predictor`

**Types:**
- PascalCase for classes: `SAMModel`, `RealTimeClickCollector`, `CachedSegmentationData`
- No I prefix for interfaces (classes used directly)
- Type aliases inline: `Optional[np.ndarray]`, `List[Path]`

## Code Style

**Formatting:**
- Black formatter with config in `pyproject.toml`
- 100 character line length
- Double quotes for strings
- 4 space indentation
- Target version: Python 3.8

**Linting:**
- flake8 for linting
- mypy for type checking with `python_version = "3.8"`
- `warn_return_any = true`, `warn_unused_configs = true`

## Import Organization

**Order:**
1. Standard library: `import os`, `from pathlib import Path`
2. Third-party: `import numpy as np`, `import torch`
3. Local imports: `from cryoem_annotation.core import ...`
4. Type imports: `from typing import Optional, List, Dict`

**Grouping:**
- Blank line between groups
- Optional dependencies in try/except blocks:
  ```python
  try:
      import mrcfile
      MRC_AVAILABLE = True
  except ImportError:
      MRC_AVAILABLE = False
  ```

**Path Aliases:**
- No path aliases used
- Full relative imports: `from cryoem_annotation.core.sam_model import SAMModel`

## Error Handling

**Patterns:**
- Use `except Exception as e:` not bare `except:` (per CLAUDE.md)
- Throw exceptions with descriptive messages including context
- Catch at boundaries (CLI layer), let domain errors propagate

**Error Types:**
- `ValueError`: Invalid input (model type, configuration)
- `FileNotFoundError`: Missing files (checkpoint, micrographs)
- `ImportError`: Optional dependencies not installed
- Log error with context before raising when helpful

**Graceful Degradation:**
- Optional features (GPU, MRC support) fail silently with warnings
- matplotlib backend failures fall back to console input
- Missing pixel size continues without diameter calculation

## Logging

**Framework:**
- StatusLogger class in `cryoem_annotation/core/logging_utils.py`
- Console output via Click echo

**Patterns:**
- Prefix format: `[OK]`, `[ERROR]`, `[WARNING]` (ASCII only, no unicode)
- Include context: `[OK] Saved 5 segmentations to output/micro_001/`
- Progress bars: Use tqdm for batch operations

**Examples:**
```python
logger = StatusLogger()
logger.ok(f"Loaded {len(files)} micrographs")
logger.error(f"Failed to load {path}: {e}")
logger.warning(f"No pixel size found for {filename}")
```

## Comments

**When to Comment:**
- Explain "why" not "what": `# Retry 3 times because matplotlib can be flaky`
- Document non-obvious algorithms: `# epsilon = 0.002 * arc length for smoothing`
- Note platform-specific workarounds: `# macOS needs Qt5Agg for interactive mode`

**Docstrings:**
- Google-style docstrings for all public functions
- Include Args, Returns, Raises sections
- Type hints in signature, not duplicated in docstring

**Example:**
```python
def load_micrograph(file_path: Path) -> Optional[np.ndarray]:
    """
    Load a micrograph from file.

    Args:
        file_path: Path to the image file (MRC, PNG, TIFF, etc.)

    Returns:
        Grayscale image as numpy array, or None if loading fails.

    Raises:
        FileNotFoundError: If file doesn't exist.
    """
```

**TODO Comments:**
- Format: `# TODO: description` (no username, use git blame)
- Currently no TODO markers in codebase

## Function Design

**Size:**
- Keep under 50 lines where possible
- Extract helpers for complex logic
- Large functions exist in annotation/labeling due to event loops

**Parameters:**
- Max 3-4 parameters preferred
- Use options object for many parameters: `Config` class
- Destructure in function body, not parameter list

**Return Values:**
- Explicit return statements
- Return early for guard clauses: `if not path.exists(): return None`
- Use Optional[T] for functions that may return None

## Module Design

**Exports:**
- Named exports exclusively
- Export from `__init__.py` via `__all__` list
- Lazy imports for heavy modules in package `__init__.py`

**Barrel Files:**
- Each package has `__init__.py` with explicit exports
- Example: `cryoem_annotation/core/__init__.py` exports main utilities
- Lazy loading via `__getattr__()` for fast CLI startup

**Example:**
```python
# cryoem_annotation/__init__.py
__all__ = [
    'annotate_micrographs',
    'label_segmentations',
    'extract_results',
    ...
]

def __getattr__(name):
    if name == 'annotate_micrographs':
        from .annotation.annotator import annotate_micrographs
        return annotate_micrographs
    ...
```

---

*Convention analysis: 2025-12-29*
*Update when patterns change*
