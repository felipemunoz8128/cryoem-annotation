# Cryo-EM Annotation Tool

Interactive annotation and analysis tool for cryo-electron microscopy micrographs using Facebook's Segment Anything Model (SAM).

## Features

- **Interactive Annotation**: Click on objects in micrographs to automatically segment them using SAM
- **Real-time Segmentation**: See segmentation results immediately as you click
- **Labeling Tool**: Assign labels (0-9) to segmented objects
- **Data Extraction**: Export results to CSV or JSON format
- **Support for MRC files**: Native support for cryo-EM MRC file format
- **GPU/CPU Support**: Automatic GPU detection with CPU fallback

## Installation

> **📖 For detailed installation instructions, see [INSTALLATION.md](INSTALLATION.md)**

### Prerequisites

1. **Python 3.8+** (or use conda)
2. **SAM Checkpoint**: Download a SAM checkpoint file:
   - [ViT-B (smallest, recommended)](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) (~350MB)
   - [ViT-L](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth) (~1.2GB)
   - [ViT-H](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) (~2.4GB)

### Option 1: Conda Installation (Recommended)

The easiest way to install with all dependencies isolated:

**CPU-only (works everywhere):**
```bash
# Clone the repository
git clone https://github.com/felipemunoz8128/cryoem-annotation.git
cd cryoem-annotation

# Create conda environment (CPU version)
conda env create -f environment.yml

# Activate the environment
conda activate cryoem-annotation

# Install the package in development mode
pip install -e .
```

**GPU support (faster, requires NVIDIA GPU with CUDA):**
```bash
# Create conda environment with GPU support
conda env create -f environment-gpu.yml

# Activate the environment
conda activate cryoem-annotation-gpu

# Install the package in development mode
pip install -e .
```

**Note**: 
- `environment.yml` installs CPU-only PyTorch (works on all systems)
- `environment-gpu.yml` installs PyTorch with CUDA support (requires NVIDIA GPU)
- The tool will automatically use GPU if available, otherwise falls back to CPU

### Option 2: Install from Source (pip)

```bash
# Clone the repository
git clone https://github.com/felipemunoz8128/cryoem-annotation.git
cd cryoem-annotation

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package
pip install -e .

# Install Segment Anything Model
pip install git+https://github.com/facebookresearch/segment-anything.git
```

### Option 3: Install Dependencies Only

```bash
pip install -r requirements.txt
pip install git+https://github.com/facebookresearch/segment-anything.git
```

### GPU Support (Optional but Recommended)

For GPU acceleration, install PyTorch with CUDA support:

**With Conda:**
```bash
conda activate cryoem-annotation
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
```

**With pip:**
```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Quick Start

### 1. Annotate Micrographs

```bash
cryoem-annotate \
    --micrographs /path/to/micrographs \
    --checkpoint sam_vit_b_01ec64.pth \
    --output annotation_results
```

**Interactive Controls:**
- **Left-click**: Segment an object (mask appears immediately)
- **Right-click**: Finish and proceed to next micrograph
- **Press 'd' or 'u'**: Undo last segmentation

### 2. Label Segmentations

```bash
cryoem-label \
    --results annotation_results \
    --micrographs /path/to/micrographs
```

**Interactive Controls:**
- **Press '0'-'9'**: Set active label (0 = label 10)
- **Left-click**: Assign active label to clicked segmentation
- **Right-click**: Finish and proceed to next micrograph

### 3. Extract Results

```bash
cryoem-extract \
    --results annotation_results \
    --output results.csv \
    --format csv
```

## Configuration

You can use a YAML config file to avoid specifying paths every time:

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
    method: "percentile"
    percentile_range: [1, 99]
```

Then use it with:

```bash
cryoem-annotate --config config.yaml
```

## Command-Line Options

### `cryoem-annotate`

```
Options:
  --micrographs, -m PATH    Path to micrograph folder (required)
  --checkpoint, -c PATH     Path to SAM checkpoint file (required)
  --model-type TEXT         SAM model type: vit_b, vit_l, or vit_h [default: vit_b]
  --output, -o PATH          Output folder [default: annotation_results]
  --config PATH              Path to config file
  --device TEXT              Device: cuda, cpu, or auto [default: auto]
```

### `cryoem-label`

```
Options:
  --results, -r PATH        Path to annotation results folder (required)
  --micrographs, -m PATH     Path to micrograph folder (required)
  --config PATH              Path to config file
```

### `cryoem-extract`

```
Options:
  --results, -r PATH        Path to annotation results folder (required)
  --output, -o PATH         Output file path (default: results.csv/json)
  --format, -f TEXT         Output format: csv, json, or both [default: csv]
```

## Output Structure

After annotation, results are saved in the following structure:

```
annotation_results/
├── micrograph_name_1/
│   ├── metadata.json          # Annotation metadata
│   ├── overview.png           # Visualization
│   ├── mask_001_binary.png    # Binary masks
│   ├── mask_002_binary.png
│   └── ...
├── micrograph_name_2/
│   └── ...
└── all_annotations.json       # Combined results
```

## Python API

You can also use the package programmatically:

```python
from pathlib import Path
from cryoem_annotation import annotate_micrographs, label_segmentations, extract_results

# Annotate
annotate_micrographs(
    micrograph_folder=Path("data/micrographs"),
    checkpoint_path=Path("sam_vit_b_01ec64.pth"),
    output_folder=Path("results/annotations"),
    model_type="vit_b",
)

# Label
label_segmentations(
    results_folder=Path("results/annotations"),
    micrograph_folder=Path("data/micrographs"),
)

# Extract
extract_results(
    results_folder=Path("results/annotations"),
    output_path=Path("results/final_data.csv"),
    output_format="csv",
)
```

## Supported File Formats

- **MRC files**: Native cryo-EM format (`.mrc`)
- **Image files**: `.tif`, `.tiff`, `.png`, `.jpg`, `.jpeg`

## GPU Support

The tool automatically detects and uses CUDA if available. To force CPU mode:

```bash
cryoem-annotate --device cpu ...
```

## Troubleshooting

### "No module named 'segment_anything'"
Install SAM: `pip install git+https://github.com/facebookresearch/segment-anything.git`

### "Checkpoint not found"
Download a SAM checkpoint and specify the path with `--checkpoint`

### GUI/Backend Issues
On Linux clusters, you may need X11 forwarding:
```bash
ssh -X username@cluster
```

The tool has a fallback mode that uses console input if the GUI backend fails.

### CUDA/GPU Issues
Check CUDA availability:
```python
python -c "import torch; print(torch.cuda.is_available())"
```

Install PyTorch with CUDA support:
```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Citation

If you use this tool in your research, please cite:

- Segment Anything Model: [Kirillov et al., 2023](https://arxiv.org/abs/2304.02643)

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- Built using [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything) by Meta AI Research

