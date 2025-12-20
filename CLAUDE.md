# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Cryo-EM Annotation Tool for segmenting and labeling objects in cryo-electron microscopy micrographs using Meta's Segment Anything Model (SAM). The tool provides a three-stage workflow: annotate → label → extract.

## Commands

### Development Setup
```bash
conda env create -f environment.yml  # CPU
conda env create -f environment-gpu.yml  # GPU with CUDA
conda activate cryoem-annotation
pip install -e .
```

### Running Tests
```bash
pytest tests/ -v                    # All tests
pytest tests/test_core/ -v          # Single directory
pytest tests/test_core/test_image_loader.py::TestLoadMicrograph::test_load_png_image -v  # Single test
```

### CLI Tools
```bash
cryoem-annotate --micrographs <path> --checkpoint <sam.pth> --output <dir>
cryoem-label --results <dir> --micrographs <path>
cryoem-extract --results <dir> --output <file> --format csv
```

## Architecture

### Package Structure
```
cryoem_annotation/
├── core/           # Core utilities (SAM wrapper, image loading, processing)
├── annotation/     # Interactive segmentation workflow
├── labeling/       # Interactive labeling tool
├── extraction/     # Data export (CSV/JSON)
├── io/             # Metadata and mask I/O
├── cli/            # Click-based CLI entry points
└── config.py       # YAML + env var configuration
```

### Data Flow

1. **Annotation** (`annotator.py` → `click_collector.py`): Loads micrographs, runs SAM predictions on clicks, saves masks as PNGs and metadata as JSON
2. **Labeling** (`labeler.py`): Loads saved masks, enables label assignment via clicks, updates metadata.json
3. **Extraction** (`extractor.py`): Reads all metadata.json files, outputs split CSVs (`*_metadata.csv`, `*_results.csv`)

### Key Classes

- **`SAMModel`** (`core/sam_model.py`): Wrapper for SAM with GPU memory optimization and inference context management
- **`RealTimeClickCollector`** (`annotation/click_collector.py`): Matplotlib-based interactive click handler with real-time SAM predictions
- **`SegmentationLabeler`** (`labeling/labeler.py`): Interactive label assignment with `CachedSegmentationData` for performance
- **`StatusLogger`** (`core/logging_utils.py`): Consistent status message formatting

### Metadata Structure

Each micrograph gets a `metadata.json` with:
```json
{
  "filename": "micro.mrc",
  "pixel_size_nm": 0.5,
  "segmentations": [
    {"click_index": 1, "click_coords": [x, y], "mask_score": 0.95, "mask_area": 5000, "label": 2}
  ]
}
```

### Extraction Output

Two separate CSVs for easy analysis:
- `*_metadata.csv`: segmentation_id, micrograph_name, click_index, click_coords, mask_score
- `*_results.csv`: segmentation_id, label, area_pixels, area_nm2

### MRC Pixel Size

Pixel size is extracted from MRC headers via `mrcfile.voxel_size` (in Angstroms, converted to nm). CLI `--pixel-size` flag overrides header values.

## Code Patterns

- Use `[OK]`, `[ERROR]`, `[WARNING]` prefixes for status messages (no unicode symbols)
- Use `except Exception:` not bare `except:`
- Use `create_bounded_overlay()` for memory-efficient mask visualization (90%+ reduction)
- Use `CachedSegmentationData` for contour/centroid caching in labeling (50-80% faster redraws)
- Use `tqdm` for batch processing progress bars
