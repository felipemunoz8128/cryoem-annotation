# Packaging Plan: Cryo-EM Annotation Tool

## Overview
Transform the current workspace into a polished, shareable Python package for interactive annotation and analysis of cryo-electron microscopy micrographs using Segment Anything Model (SAM).

## Goals
1. **User-friendly installation** - Simple pip install or git clone
2. **CLI interface** - Command-line tools instead of editing scripts
3. **Configuration management** - Config files instead of hardcoded paths
4. **Professional structure** - Standard Python package layout
5. **Comprehensive documentation** - README, usage examples, API docs
6. **Easy to extend** - Modular, well-organized code

---

## Phase 1: Project Structure

### Proposed Directory Structure
```
cryoem-annotation/
├── cryoem_annotation/          # Main package directory
│   ├── __init__.py
│   ├── core/                   # Core functionality
│   │   ├── __init__.py
│   │   ├── sam_model.py        # SAM model loading/management
│   │   ├── image_loader.py     # MRC/image loading utilities
│   │   ├── image_processing.py # Normalization, contrast enhancement
│   │   └── colors.py           # Color generation utilities
│   ├── annotation/             # Annotation tools
│   │   ├── __init__.py
│   │   ├── annotator.py        # Main annotation class
│   │   └── click_collector.py  # Click collection logic
│   ├── labeling/               # Labeling tools
│   │   ├── __init__.py
│   │   └── labeler.py         # Labeling interface
│   ├── extraction/             # Data extraction
│   │   ├── __init__.py
│   │   └── extractor.py       # Data extraction logic
│   ├── io/                     # I/O utilities
│   │   ├── __init__.py
│   │   ├── metadata.py        # Metadata handling
│   │   └── masks.py           # Mask saving/loading
│   └── cli/                    # CLI entry points
│       ├── __init__.py
│       ├── annotate.py         # CLI for annotation
│       ├── label.py            # CLI for labeling
│       └── extract.py          # CLI for extraction
├── tests/                      # Unit tests
│   ├── __init__.py
│   ├── test_image_loader.py
│   ├── test_metadata.py
│   └── test_extraction.py
├── examples/                    # Example workflows
│   ├── basic_workflow.py
│   └── sample_config.yaml
├── docs/                        # Documentation
│   ├── installation.md
│   ├── usage.md
│   └── api.md
├── .github/
│   └── workflows/
│       └── ci.yml              # GitHub Actions CI
├── .gitignore
├── LICENSE                      # MIT or Apache 2.0
├── README.md                    # Main documentation
├── pyproject.toml              # Modern Python packaging
├── setup.py                    # Fallback for older pip
├── requirements.txt            # Dependencies
├── requirements-dev.txt        # Dev dependencies
└── CHANGELOG.md                # Version history
```

---

## Phase 2: Configuration Management

### Current Issues
- Hardcoded paths in scripts (`/Volumes/SB_lab_18TB/...`)
- Configuration scattered across files
- No easy way to change settings without editing code

### Solution: Configuration System

#### Option A: YAML Config Files (Recommended)
```yaml
# config.yaml
micrograph_folder: "/path/to/micrographs"
sam_model:
  type: "vit_b"  # vit_b, vit_l, or vit_h
  checkpoint_path: "sam_vit_b_01ec64.pth"
output:
  folder: "annotation_results"
  create_overview: true
  save_masks: true
image:
  extensions: [".mrc", ".tif", ".tiff", ".png", ".jpg", ".jpeg"]
  normalization:
    method: "percentile"  # "minmax" or "percentile"
    percentile_range: [1, 99]
```

#### Option B: Environment Variables + Config File
- Use `.env` files for sensitive paths
- YAML for other settings

#### Implementation
- Create `cryoem_annotation/config.py` module
- Load config from:
  1. Command-line arguments (highest priority)
  2. Config file (e.g., `~/.cryoem_annotation/config.yaml`)
  3. Environment variables
  4. Default values

---

## Phase 3: CLI Interface

### Command Structure
```bash
# Install package
pip install cryoem-annotation

# Or from source
git clone https://github.com/username/cryoem-annotation
cd cryoem-annotation
pip install -e .

# Usage
cryoem-annotate --micrographs /path/to/images --checkpoint sam_vit_b.pth
cryoem-label --results annotation_results --micrographs /path/to/images
cryoem-extract --results annotation_results --output results.csv
```

### CLI Commands

#### 1. `cryoem-annotate`
```bash
cryoem-annotate [OPTIONS]

Options:
  --micrographs PATH       Path to micrograph folder (required)
  --checkpoint PATH        Path to SAM checkpoint file (required)
  --model-type TEXT        SAM model type: vit_b, vit_l, or vit_h [default: vit_b]
  --output PATH            Output folder [default: annotation_results]
  --config PATH            Path to config file
  --device TEXT            Device: cuda, cpu, or auto [default: auto]
```

#### 2. `cryoem-label`
```bash
cryoem-label [OPTIONS]

Options:
  --results PATH           Path to annotation results folder (required)
  --micrographs PATH       Path to micrograph folder (required)
  --config PATH            Path to config file
```

#### 3. `cryoem-extract`
```bash
cryoem-extract [OPTIONS]

Options:
  --results PATH           Path to annotation results folder (required)
  --output PATH            Output file path [default: results.csv]
  --format TEXT            Output format: csv, json, or both [default: csv]
  --pixel-size FLOAT      Pixel size in meters (for area conversion)
```

### Implementation
- Use `click` or `argparse` for CLI
- Entry points in `pyproject.toml`:
```toml
[project.scripts]
cryoem-annotate = "cryoem_annotation.cli.annotate:main"
cryoem-label = "cryoem_annotation.cli.label:main"
cryoem-extract = "cryoem_annotation.cli.extract:main"
```

---

## Phase 4: Code Refactoring

### Key Refactoring Tasks

#### 4.1 Extract Common Utilities
- **Image Loading**: Move MRC/image loading to `core/image_loader.py`
- **Normalization**: Move to `core/image_processing.py`
- **Color Generation**: Move to `core/colors.py`
- **Metadata Handling**: Move to `io/metadata.py`

#### 4.2 Separate Concerns
- **Annotation Logic**: Extract `RealTimeClickCollector` → `annotation/click_collector.py`
- **SAM Integration**: Extract SAM loading → `core/sam_model.py`
- **Labeling Logic**: Extract `SegmentationLabeler` → `labeling/labeler.py`

#### 4.3 Error Handling
- Add proper exception classes
- Better error messages
- Graceful fallbacks

#### 4.4 Code Quality
- Type hints throughout
- Docstrings (Google or NumPy style)
- Remove duplicate code
- Consistent naming conventions

---

## Phase 5: Packaging Setup

### pyproject.toml
```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "cryoem-annotation"
version = "0.1.0"
description = "Interactive annotation tool for cryo-EM micrographs using SAM"
readme = "README.md"
requires-python = ">=3.8"
license = {text = "MIT"}
authors = [
    {name = "Your Name", email = "your.email@example.com"}
]
keywords = ["cryo-em", "segmentation", "annotation", "sam", "microscopy"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Topic :: Scientific/Engineering :: Image Processing",
]

dependencies = [
    "numpy>=1.21.0",
    "matplotlib>=3.5.0",
    "opencv-python>=4.5.0",
    "torch>=1.7.0",
    "torchvision>=0.8.0",
    "mrcfile>=1.4.0",
    "pyyaml>=6.0",
    "click>=8.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
    "black>=23.0",
    "flake8>=6.0",
    "mypy>=1.0",
]

[project.scripts]
cryoem-annotate = "cryoem_annotation.cli.annotate:main"
cryoem-label = "cryoem_annotation.cli.label:main"
cryoem-extract = "cryoem_annotation.cli.extract:main"

[tool.setuptools.packages.find]
where = ["."]
include = ["cryoem_annotation*"]
```

### requirements.txt
Keep for backward compatibility:
```
numpy>=1.21.0
matplotlib>=3.5.0
opencv-python>=4.5.0
torch>=1.7.0
torchvision>=0.8.0
mrcfile>=1.4.0
pyyaml>=6.0
click>=8.0
```

Note: SAM must be installed separately:
```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
```

---

## Phase 6: Documentation

### README.md Structure
1. **Title & Badges** (PyPI, license, Python version)
2. **Quick Start** - Installation and basic usage
3. **Features** - What the package does
4. **Installation** - Detailed install instructions
5. **Usage** - Examples for each command
6. **Configuration** - How to configure
7. **Examples** - Link to examples folder
8. **Contributing** - How to contribute
9. **License** - License information
10. **Citation** - How to cite if used in research

### Additional Documentation
- **docs/installation.md**: Detailed installation (including SAM checkpoint)
- **docs/usage.md**: Comprehensive usage guide
- **docs/api.md**: API documentation (auto-generated with Sphinx?)
- **examples/**: Jupyter notebooks or Python scripts

---

## Phase 7: Testing & Quality

### Testing Strategy
- **Unit Tests**: Test individual functions (image loading, normalization, etc.)
- **Integration Tests**: Test full workflows
- **Mock Tests**: Mock SAM model for faster testing

### Code Quality Tools
- **Black**: Code formatting
- **flake8**: Linting
- **mypy**: Type checking
- **pytest**: Testing framework

### CI/CD (GitHub Actions)
- Run tests on push/PR
- Check code formatting
- Type checking
- (Optional) Build and publish to PyPI

---

## Phase 8: Example Data & Workflows

### Example Structure
```
examples/
├── README.md
├── basic_workflow.py          # Complete workflow example
├── config_example.yaml        # Example config file
└── notebooks/
    ├── 01_annotation.ipynb
    ├── 02_labeling.ipynb
    └── 03_analysis.ipynb
```

### Sample Workflow Script
```python
#!/usr/bin/env python3
"""
Example workflow: Annotate → Label → Extract
"""
from pathlib import Path
from cryoem_annotation import annotate, label, extract

# Step 1: Annotate
annotate.annotate_micrographs(
    micrograph_folder=Path("data/micrographs"),
    checkpoint_path=Path("sam_vit_b_01ec64.pth"),
    output_folder=Path("results/annotations")
)

# Step 2: Label
label.label_segmentations(
    results_folder=Path("results/annotations"),
    micrograph_folder=Path("data/micrographs")
)

# Step 3: Extract
extract.extract_results(
    results_folder=Path("results/annotations"),
    output_path=Path("results/final_data.csv")
)
```

---

## Phase 9: Migration Checklist

### Code Changes
- [ ] Create package structure (`cryoem_annotation/`)
- [ ] Move and refactor `interactive_annotation.py` → `annotation/annotator.py`
- [ ] Move and refactor `label_segmentations.py` → `labeling/labeler.py`
- [ ] Move and refactor `extract_labels_areas.py` → `extraction/extractor.py`
- [ ] Extract common utilities to `core/`
- [ ] Create CLI modules in `cli/`
- [ ] Add configuration management
- [ ] Remove hardcoded paths
- [ ] Add type hints
- [ ] Add docstrings
- [ ] Add error handling

### Packaging
- [ ] Create `pyproject.toml`
- [ ] Create `setup.py` (if needed)
- [ ] Update `requirements.txt`
- [ ] Create `requirements-dev.txt`
- [ ] Add `.gitignore`
- [ ] Add `LICENSE`
- [ ] Create `CHANGELOG.md`

### Documentation
- [ ] Write comprehensive `README.md`
- [ ] Create `docs/` folder with guides
- [ ] Add docstrings to all functions/classes
- [ ] Create example scripts
- [ ] Add installation instructions for SAM

### Testing
- [ ] Set up pytest
- [ ] Write unit tests for core functions
- [ ] Write integration tests
- [ ] Set up GitHub Actions CI

### Polish
- [ ] Remove unused files (backup scripts, etc.)
- [ ] Clean up example data (or move to separate repo)
- [ ] Add badges to README
- [ ] Create GitHub repository
- [ ] Add issue templates
- [ ] Add pull request template

---

## Phase 10: Distribution

### Options
1. **PyPI** (Recommended for wide distribution)
   - `python -m build`
   - `twine upload dist/*`

2. **GitHub Releases**
   - Tag releases
   - Attach wheel files

3. **Conda Forge** (Optional, for scientific community)
   - Create conda recipe
   - Submit to conda-forge

---

## Implementation Order

### Week 1: Foundation
1. Create package structure
2. Refactor core utilities
3. Set up configuration system
4. Create basic CLI structure

### Week 2: Core Features
1. Refactor annotation tool
2. Refactor labeling tool
3. Refactor extraction tool
4. Integrate CLI commands

### Week 3: Polish
1. Write documentation
2. Add tests
3. Set up CI/CD
4. Create examples

### Week 4: Release
1. Final testing
2. Prepare for distribution
3. Create GitHub repo
4. Publish to PyPI (optional)

---

## Notes & Considerations

### SAM Checkpoint Handling
- **Issue**: SAM checkpoints are large (~350MB - 2.4GB)
- **Solution**: 
  - Don't include in package
  - Provide download script: `cryoem-download-checkpoint`
  - Document download URLs in README

### GPU/CPU Support
- Auto-detect CUDA availability
- Allow manual override via CLI/config
- Clear error messages if CUDA expected but not available

### Backward Compatibility
- Consider if users have existing `annotation_results/` folders
- Ensure new code can read old metadata format
- Provide migration script if needed

### Platform Support
- Test on Linux, macOS, Windows
- Handle matplotlib backend issues gracefully
- Document platform-specific requirements

---

## Success Criteria

✅ Users can install with `pip install cryoem-annotation`  
✅ Users can run tools via CLI without editing code  
✅ Configuration is external (files/env vars)  
✅ Code is well-organized and documented  
✅ Examples work out of the box  
✅ Tests pass  
✅ README is comprehensive and clear  

---

## Next Steps

1. **Review this plan** - Get feedback on structure and approach
2. **Create GitHub repo** - Set up repository structure
3. **Start Phase 1** - Begin refactoring code into package structure
4. **Iterate** - Build incrementally, test as you go

