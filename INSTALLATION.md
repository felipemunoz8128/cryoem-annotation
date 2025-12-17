# Installation Guide

This guide provides detailed installation instructions for the Cryo-EM Annotation Tool.

## Quick Start (Conda - Recommended)

```bash
git clone https://github.com/felipemunoz8128/cryoem-annotation.git
cd cryoem-annotation
conda env create -f environment.yml
conda activate cryoem-annotation
pip install -e .
```

## Detailed Installation Options

### Option 1: Conda Environment (Easiest)

Conda provides the cleanest installation with all dependencies isolated.

#### Step 1: Install Conda

If you don't have conda installed:
- **Miniconda**: https://docs.conda.io/en/latest/miniconda.html
- **Anaconda**: https://www.anaconda.com/products/distribution

#### Step 2: Create Environment

**Choose CPU or GPU version:**

**CPU-only (works everywhere, slower):**
```bash
# Clone repository
git clone https://github.com/felipemunoz8128/cryoem-annotation.git
cd cryoem-annotation

# Create environment from file (CPU version)
conda env create -f environment.yml

# Activate environment
conda activate cryoem-annotation
```

**GPU support (faster, requires NVIDIA GPU with CUDA 11.8 or 12.1):**
```bash
# Clone repository
git clone https://github.com/felipemunoz8128/cryoem-annotation.git
cd cryoem-annotation

# Create environment with GPU support
conda env create -f environment-gpu.yml

# Activate environment
conda activate cryoem-annotation-gpu
```

**Note**: The GPU version requires:
- NVIDIA GPU with CUDA support
- CUDA 11.8 or 12.1 installed on your system
- If you have a different CUDA version, edit `environment-gpu.yml` and change `pytorch-cuda=11.8` to match your CUDA version

#### Step 3: Install Package

```bash
# Install in development mode (editable)
pip install -e .
```

#### Step 4: Verify Installation

```bash
# Check CLI commands are available
cryoem-annotate --help
cryoem-label --help
cryoem-extract --help
```

#### Managing the Conda Environment

```bash
# Deactivate when done
conda deactivate

# Remove environment (if needed)
conda env remove -n cryoem-annotation

# List all environments
conda env list
```

### Option 2: Python Virtual Environment

#### Step 1: Create Virtual Environment

```bash
# Clone repository
git clone https://github.com/felipemunoz8128/cryoem-annotation.git
cd cryoem-annotation

# Create virtual environment
python -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

#### Step 2: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install package and dependencies
pip install -e .

# Install Segment Anything Model
pip install git+https://github.com/facebookresearch/segment-anything.git
```

### Option 3: System-wide Installation (Not Recommended)

```bash
pip install -e .
pip install git+https://github.com/facebookresearch/segment-anything.git
```

**Warning**: This installs packages system-wide. Use a virtual environment or conda instead.

## GPU Support

### Check CUDA Availability

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

### Install PyTorch with CUDA (Conda)

**Option A: Use GPU environment file (recommended)**
```bash
# Use environment-gpu.yml which includes GPU support
conda env create -f environment-gpu.yml
conda activate cryoem-annotation-gpu
```

**Option B: Add GPU support to existing CPU environment**
```bash
conda activate cryoem-annotation
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
# For CUDA 12.1, use: pytorch-cuda=12.1
```

### Install PyTorch with CUDA (pip)

```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Download SAM Checkpoint

After installation, download a SAM checkpoint:

```bash
# ViT-B (recommended, ~350MB)
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

# Or download manually from:
# https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

## Troubleshooting

### "No module named 'segment_anything'"

```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
```

### "No module named 'mrcfile'"

```bash
pip install mrcfile
```

### Conda environment creation fails

Try updating conda:
```bash
conda update conda
conda env create -f environment.yml
```

### CUDA/GPU not detected

1. Check CUDA is installed: `nvcc --version`
2. Install PyTorch with CUDA support (see GPU Support section)
3. Verify: `python -c "import torch; print(torch.cuda.is_available())"`

### Matplotlib backend issues

The tool automatically tries different backends. If GUI doesn't work:
- Linux: Install `python3-tk`: `sudo apt-get install python3-tk`
- Mac: Usually works with default backend
- Windows: Usually works with default backend

## Verification

After installation, test that everything works:

```bash
# Activate environment
conda activate cryoem-annotation  # or: source venv/bin/activate

# Test imports
python -c "import cryoem_annotation; print('Package imported successfully')"

# Test CLI
cryoem-annotate --help
```

## Next Steps

1. Download a SAM checkpoint (see above)
2. Read the [README.md](README.md) for usage instructions
3. Check [examples/](examples/) for example workflows

