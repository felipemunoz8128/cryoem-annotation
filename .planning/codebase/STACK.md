# Technology Stack

**Analysis Date:** 2025-12-29

## Languages

**Primary:**
- Python 3.8+ - All application code

**Secondary:**
- YAML - Configuration files (`environment.yml`, `environment-gpu.yml`)

## Runtime

**Environment:**
- Python 3.8-3.11 (specified in `pyproject.toml`)
- Conda environments for dependency management
- GPU support via CUDA 11.8/12.1 (optional)

**Package Manager:**
- Conda + pip
- Lockfile: No lockfile present (uses conda environment specs)

## Frameworks

**Core:**
- Click 8.0+ - CLI framework (`cryoem_annotation/cli/*.py`)
- Matplotlib - Interactive visualization and event handling

**Testing:**
- Pytest 7.0+ - Unit and integration testing
- pytest-cov 4.0+ - Coverage reporting

**Build/Dev:**
- setuptools 61.0+ - PEP 517/518 build system
- Black 23.0+ - Code formatting
- flake8 6.0+ - Linting
- mypy 1.0+ - Type checking

## Key Dependencies

**Critical:**
- `torch>=1.7.0` / `pytorch>=2.0.0` (GPU) - Deep learning framework
- `segment-anything` (git) - Meta's SAM model for segmentation
- `mrcfile>=1.4.0` - MRC file format support for cryo-EM
- `opencv-python>=4.5.0` - Image processing (`cv2`)
- `numpy>=1.21.0` - Numerical computing

**Infrastructure:**
- `matplotlib>=3.5.0` - Interactive plotting with event handling
- `click>=8.0` - CLI argument parsing
- `pyyaml>=6.0` - YAML configuration loading
- `tqdm>=4.0` - Progress bars for batch processing

## Configuration

**Environment:**
- YAML config files (optional) - `cryoem_annotation/config.py`
- Environment variables with `CRYOEM_*` prefix
  - `CRYOEM_MICROGRAPH_FOLDER` - Default micrograph folder
  - `CRYOEM_SAM_TYPE` - SAM model type (vit_b, vit_l, vit_h)
  - `CRYOEM_SAM_CHECKPOINT` - Path to SAM checkpoint file
  - `CRYOEM_OUTPUT_FOLDER` - Default output folder
- CLI arguments override all settings

**Build:**
- `pyproject.toml` - Project configuration
- `environment.yml` - CPU conda environment
- `environment-gpu.yml` - GPU conda environment with CUDA

## Platform Requirements

**Development:**
- macOS/Linux/Windows (any platform with Python 3.8+)
- Conda recommended for dependency management
- No external services required (operates entirely locally)

**Production:**
- Distributed as installable Python package
- `pip install -e .` for editable install
- SAM model checkpoint required (downloadable from Meta)
- GPU optional but recommended for performance

---

*Stack analysis: 2025-12-29*
*Update after major dependency changes*
