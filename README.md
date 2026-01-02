# Cryo-EM Annotation Tool

Interactive annotation tool for cryo-electron microscopy micrographs using Segment Anything Model (SAM).

Here is a [video](https://youtu.be/Dth3aiiHcMM?si=dGNzbGPEQ0RfgX8I) demonstrating how to use this project.

## Features

- **Real-time Segmentation**: Click on objects to instantly segment them using SAM
- **Multi-Grid Support**: Process multiple grids from motion correction pipelines (`motion_corrected/{Grid}/`)
- **Session Resume**: Previously completed files are marked on restart (`[x]` done, `[~]` partial, `[ ]` pending)
- **Labeling Tool**: Assign categorical labels to segmented objects
- **Data Extraction**: Export to CSV/JSON with physical measurements (nm)
- **MRC Support**: Native cryo-EM format with automatic pixel size extraction
- **GPU/CPU**: Automatic GPU detection with CPU fallback

## Installation

### Prerequisites
- **Conda** (Miniconda or Anaconda)
- **SAM Checkpoint**: [ViT-B (recommended, 350MB)](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)

### Install

```bash
git clone https://github.com/felipemunoz8128/cryoem-annotation.git
cd cryoem-annotation
conda env create -f environment.yml      # CPU (or environment-gpu.yml for GPU)
conda activate cryoem-annotation
pip install -e .
```

## Usage

### 1. Annotate

```bash
cryoem-annotate -m /path/to/micrographs -c sam_vit_b_01ec64.pth -o results
```

**Controls:**
- **Left-click**: Segment object
- **Right-click / Arrow keys**: Navigate files
- **'d' or 'u'**: Undo last segmentation
- **Escape**: Finish

### 2. Label

```bash
cryoem-label -r results -m /path/to/micrographs
```

**Controls:**
- **'0'-'9'**: Select label
- **Left-click**: Assign label to segmentation
- **Right-click / Arrow keys**: Navigate files
- **Escape**: Finish

### 3. Extract

```bash
cryoem-extract -r results -m /path/to/micrographs -o output -f csv
```

Creates two CSVs:
- `output_metadata.csv`: IDs, coordinates, scores, area (pixels)
- `output_results.csv`: Labels, diameter (nm)

## Project Workflow

After the first annotation run, a `.cryoem-project.json` file is created in your results folder. This saves your paths, so subsequent commands can auto-detect them:

```bash
cd results
cryoem-label          # No need to specify paths again
cryoem-extract -o my_data
```

## Command Reference

### cryoem-annotate
| Option | Description |
|--------|-------------|
| `-m, --micrographs` | Path to micrographs folder |
| `-c, --checkpoint` | Path to SAM checkpoint |
| `-o, --output` | Output folder (default: `annotation_results`) |
| `--model-type` | SAM model: `vit_b`, `vit_l`, `vit_h` (default: `vit_b`) |
| `--device` | `cuda`, `cpu`, or `auto` (default: `auto`) |

### cryoem-label
| Option | Description |
|--------|-------------|
| `-r, --results` | Path to annotation results |
| `-m, --micrographs` | Path to micrographs folder |

### cryoem-extract
| Option | Description |
|--------|-------------|
| `-r, --results` | Path to annotation results |
| `-m, --micrographs` | Path to micrographs (for accurate counts) |
| `-o, --output` | Output file base path |
| `-f, --format` | `csv`, `json`, or `both` (default: `csv`) |
| `-p, --pixel-size` | Override pixel size (nm/pixel) |

## Output Structure

```
results/
├── .cryoem-project.json     # Project config (auto-created)
├── Grid1/                   # Per-grid folders (multi-grid mode)
│   └── micrograph_001/
│       ├── metadata.json
│       ├── overview.png
│       └── mask_*.png
└── all_annotations.json
```

## Supported Formats

- **MRC** (`.mrc`) - with automatic pixel size extraction
- **Images**: `.tif`, `.tiff`, `.png`, `.jpg`, `.jpeg`

## Acknowledgements

Built with [Segment Anything Model](https://github.com/facebookresearch/segment-anything) by Meta AI Research ([Kirillov et al., 2023](https://arxiv.org/abs/2304.02643))
