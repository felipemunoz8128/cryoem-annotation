# Codebase Structure

**Analysis Date:** 2025-12-29

## Directory Layout

```
cryoem-annotation/
├── cryoem_annotation/       # Main Python package
│   ├── cli/                # CLI entry points (Click commands)
│   ├── core/               # Shared utilities and services
│   ├── annotation/         # Interactive segmentation workflow
│   ├── labeling/           # Interactive label assignment
│   ├── extraction/         # Data export (CSV/JSON)
│   ├── io/                 # Metadata and mask I/O
│   ├── navigation/         # File navigation UI (Tkinter)
│   └── gui/                # Reserved for future GUI (empty)
├── tests/                  # Test suite
│   ├── test_core/          # Core module tests
│   ├── test_annotation/    # Annotation tests
│   ├── test_labeling/      # Labeling tests
│   ├── test_extraction/    # Extraction tests
│   ├── test_io/            # I/O tests
│   └── benchmarks/         # Performance benchmarks
├── environment.yml         # CPU conda environment
├── environment-gpu.yml     # GPU conda environment
├── pyproject.toml          # Project configuration
├── README.md               # User documentation
└── CLAUDE.md               # Claude Code guidance
```

## Directory Purposes

**cryoem_annotation/cli/**
- Purpose: Click-based CLI command definitions
- Contains: `annotate.py`, `label.py`, `extract.py`
- Key files: Each file defines one CLI command
- Subdirectories: None

**cryoem_annotation/core/**
- Purpose: Shared utilities and foundational services
- Contains: SAM wrapper, image loading, processing, colors, logging
- Key files: `sam_model.py` (SAMModel class), `image_loader.py` (load_micrograph)
- Subdirectories: None

**cryoem_annotation/annotation/**
- Purpose: Interactive segmentation with SAM
- Contains: Main annotation loop, click event handling
- Key files: `annotator.py` (annotate_micrographs), `click_collector.py` (RealTimeClickCollector)
- Subdirectories: None

**cryoem_annotation/labeling/**
- Purpose: Interactive label assignment for segmentations
- Contains: Labeling loop, category management, caching
- Key files: `labeler.py` (SegmentationLabeler), `categories.py` (LabelCategories)
- Subdirectories: None

**cryoem_annotation/extraction/**
- Purpose: Export annotated data to CSV/JSON
- Contains: Data aggregation, metric calculation
- Key files: `extractor.py` (extract_results, print_summary)
- Subdirectories: None

**cryoem_annotation/io/**
- Purpose: File I/O for metadata and masks
- Contains: JSON metadata handling, binary mask PNG I/O
- Key files: `metadata.py` (save_metadata, load_metadata), `masks.py` (save_mask_binary)
- Subdirectories: None

**cryoem_annotation/navigation/**
- Purpose: File navigation window for browsing results
- Contains: Tkinter-based file list UI
- Key files: `nav_window.py` (NavigationWindow class)
- Subdirectories: None

**tests/**
- Purpose: Pytest test suite
- Contains: Unit tests organized by module, benchmarks
- Key files: `conftest.py` (276 lines of fixtures)
- Subdirectories: `test_core/`, `test_annotation/`, `test_labeling/`, `test_extraction/`, `test_io/`, `benchmarks/`

## Key File Locations

**Entry Points:**
- `cryoem_annotation/cli/annotate.py` - `cryoem-annotate` command
- `cryoem_annotation/cli/label.py` - `cryoem-label` command
- `cryoem_annotation/cli/extract.py` - `cryoem-extract` command

**Configuration:**
- `pyproject.toml` - Project metadata, dependencies, CLI entry points
- `cryoem_annotation/config.py` - Config class with YAML + env var loading
- `environment.yml` - CPU conda environment
- `environment-gpu.yml` - GPU conda environment

**Core Logic:**
- `cryoem_annotation/core/sam_model.py` - SAMModel wrapper class
- `cryoem_annotation/core/image_loader.py` - load_micrograph(), get_image_files()
- `cryoem_annotation/annotation/click_collector.py` - RealTimeClickCollector, create_bounded_overlay()
- `cryoem_annotation/labeling/labeler.py` - SegmentationLabeler, CachedSegmentationData
- `cryoem_annotation/extraction/extractor.py` - extract_results(), print_summary()

**Testing:**
- `tests/conftest.py` - Comprehensive fixtures (images, masks, mocks)
- `tests/test_core/*.py` - Core module unit tests
- `tests/benchmarks/*.py` - Performance benchmarks

**Documentation:**
- `README.md` - User-facing installation and usage guide
- `CLAUDE.md` - Instructions for Claude Code when working in this repo

## Naming Conventions

**Files:**
- snake_case.py: All Python modules (`sam_model.py`, `click_collector.py`)
- test_*.py: Test files (`test_image_loader.py`)
- bench_*.py: Benchmark files (`bench_contour_caching.py`)
- UPPERCASE.md: Important project files (`README.md`, `CLAUDE.md`)

**Directories:**
- lowercase: All directories (`core/`, `annotation/`, `tests/`)
- test_: Test directories mirror source structure

**Special Patterns:**
- `__init__.py`: Package exports with `__all__` lists
- `conftest.py`: Pytest fixtures for test directories

## Where to Add New Code

**New CLI Command:**
- Primary code: `cryoem_annotation/cli/{command-name}.py`
- Entry point: Add to `pyproject.toml` `[project.scripts]`
- Tests: `tests/test_cli/test_{command-name}.py`

**New Core Utility:**
- Implementation: `cryoem_annotation/core/{name}.py`
- Export from: `cryoem_annotation/core/__init__.py`
- Tests: `tests/test_core/test_{name}.py`

**New Workflow Stage:**
- Implementation: `cryoem_annotation/{stage}/`
- CLI wrapper: `cryoem_annotation/cli/{stage}.py`
- Tests: `tests/test_{stage}/`

**New I/O Format:**
- Implementation: `cryoem_annotation/io/{format}.py`
- Export from: `cryoem_annotation/io/__init__.py`
- Tests: `tests/test_io/test_{format}.py`

**Utilities:**
- Shared helpers: `cryoem_annotation/core/`
- Type definitions: Inline in modules (no separate types file)

## Special Directories

**cryoem_annotation/gui/**
- Purpose: Reserved for future GUI development
- Source: Currently empty
- Committed: Yes (placeholder for future work)

**tests/benchmarks/**
- Purpose: Performance benchmarking tests
- Source: Custom benchmark framework with `BenchmarkResult` dataclass
- Committed: Yes
- Run with: `pytest tests/benchmarks/ -v`

---

*Structure analysis: 2025-12-29*
*Update when directory structure changes*
