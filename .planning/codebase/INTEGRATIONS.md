# External Integrations

**Analysis Date:** 2025-12-29

## APIs & External Services

**Machine Learning Model - Segment Anything (SAM):**
- Meta's Segment Anything Model for image segmentation
  - Git repository: `https://github.com/facebookresearch/segment-anything.git`
  - SDK/Client: Direct import via `segment_anything` package
  - Auth: None (local model, no API key)
  - Model variants: ViT-B (350MB), ViT-L (1.2GB), ViT-H (2.4GB)
  - Checkpoint downloads: `https://dl.fbaipublicfiles.com/segment_anything/`
  - Integration file: `cryoem_annotation/core/sam_model.py`

**Email/SMS:**
- Not applicable (standalone local tool)

**External APIs:**
- None (operates entirely locally)

## Data Storage

**Databases:**
- Not applicable (file-based storage only)

**File Storage:**
- Local file system only
  - Input: Micrograph images (MRC, PNG, TIFF)
  - Output: Per-micrograph directories with:
    - `metadata.json` - Segmentation data and labels
    - `mask_*.png` - Binary mask images
    - `overview.png` - Visualization image
  - Export: CSV files (`*_metadata.csv`, `*_results.csv`)

**Caching:**
- In-memory only
  - `CachedSegmentationData` for contour/centroid caching
  - No persistent cache between sessions

## Authentication & Identity

**Auth Provider:**
- Not applicable (single-user local tool)

**OAuth Integrations:**
- Not applicable

## Monitoring & Observability

**Error Tracking:**
- Console output only via StatusLogger
- No external error tracking service

**Analytics:**
- Not applicable

**Logs:**
- stdout/stderr via Click echo
- StatusLogger prefixes: `[OK]`, `[ERROR]`, `[WARNING]`

## CI/CD & Deployment

**Hosting:**
- Distributed as Python package
- Installed locally via pip: `pip install -e .`
- No hosted deployment

**CI Pipeline:**
- Not currently configured
- Manual testing via pytest

## Environment Configuration

**Development:**
- Required env vars: None required, all optional
- Optional env vars:
  - `CRYOEM_MICROGRAPH_FOLDER` - Default micrograph directory
  - `CRYOEM_SAM_TYPE` - Model type (vit_b, vit_l, vit_h)
  - `CRYOEM_SAM_CHECKPOINT` - Path to checkpoint file
  - `CRYOEM_OUTPUT_FOLDER` - Default output directory
- Secrets location: None (no secrets required)
- Setup: `pip install -e .` + download SAM checkpoint

**Staging:**
- Not applicable (no staging environment)

**Production:**
- Same as development
- Runs on user's local machine

## Webhooks & Callbacks

**Incoming:**
- Not applicable

**Outgoing:**
- Not applicable

## File Format Integrations

**MRC (Electron Microscopy):**
- Library: `mrcfile>=1.4.0` (optional dependency)
- Purpose: Read cryo-EM micrographs with pixel size metadata
- Integration: `cryoem_annotation/core/image_loader.py`
- Features: Automatic voxel_size extraction (Angstroms → nanometers)
- Fallback: Works without mrcfile (PNG/TIFF only)

**Standard Images:**
- Library: `opencv-python` (cv2)
- Formats: PNG, JPEG, TIFF
- Integration: `cryoem_annotation/core/image_loader.py`

**Configuration:**
- Library: `pyyaml`
- Format: YAML config files (optional)
- Integration: `cryoem_annotation/config.py`

## Hardware Integrations

**GPU (CUDA):**
- Library: PyTorch with CUDA support
- Purpose: Accelerated SAM inference
- Detection: Automatic via `torch.cuda.is_available()`
- Fallback: CPU inference (slower but functional)
- Memory management: Explicit cache clearing via `torch.cuda.empty_cache()`
- Configuration: `--device [cuda|cpu|auto]` CLI flag

---

*Integration audit: 2025-12-29*
*Update when adding/removing external services*
