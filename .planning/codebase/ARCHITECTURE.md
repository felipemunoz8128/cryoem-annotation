# Architecture

**Analysis Date:** 2025-12-29

## Pattern Overview

**Overall:** Multi-stage CLI Application with Layered Architecture

**Key Characteristics:**
- Three-stage linear workflow (Annotate → Label → Extract)
- Interactive matplotlib-based visualization
- Metadata-driven state management (JSON + PNG masks)
- Single-user local execution (no server/database)

## Layers

**CLI Layer:**
- Purpose: Parse user input and route to domain functions
- Contains: Click command definitions, argument validation, help text
- Location: `cryoem_annotation/cli/*.py`
- Depends on: Domain layer for business logic
- Used by: User via terminal commands

**Domain Layer:**
- Purpose: Core business logic for each workflow stage
- Contains: Annotator, Labeler, Extractor orchestration
- Location: `cryoem_annotation/annotation/`, `cryoem_annotation/labeling/`, `cryoem_annotation/extraction/`
- Depends on: Core layer, I/O layer
- Used by: CLI layer

**Core Layer:**
- Purpose: Shared utilities and foundational services
- Contains: SAM model wrapper, image loading, processing, logging
- Location: `cryoem_annotation/core/*.py`
- Depends on: External libraries (torch, cv2, matplotlib)
- Used by: Domain layer

**I/O Layer:**
- Purpose: Metadata and mask file operations
- Contains: JSON metadata save/load, binary mask I/O
- Location: `cryoem_annotation/io/*.py`
- Depends on: Standard library, numpy
- Used by: Domain layer

## Data Flow

**Stage 1 - Annotation (cryoem-annotate):**

1. User loads micrograph folder via CLI
2. SAMModel initialized with checkpoint (`cryoem_annotation/core/sam_model.py`)
3. For each micrograph:
   - Load image and extract pixel size from MRC header
   - RealTimeClickCollector displays image (`cryoem_annotation/annotation/click_collector.py`)
   - User clicks generate real-time SAM mask predictions
   - Segmentations saved as binary PNG masks + metadata.json
4. Overview visualization generated per micrograph

**Stage 2 - Labeling (cryoem-label):**

1. User loads results directory
2. SegmentationLabeler displays existing masks (`cryoem_annotation/labeling/labeler.py`)
3. User assigns labels via keyboard (1=mature, 2=immature, etc.)
4. Labels updated in existing metadata.json files

**Stage 3 - Extraction (cryoem-extract):**

1. User specifies results directory and output format
2. Extractor scans all metadata.json files (`cryoem_annotation/extraction/extractor.py`)
3. Calculates metrics (area, diameter if pixel_size available)
4. Generates split CSV output: `*_metadata.csv` + `*_results.csv`

**State Management:**
- File-based: All state in per-micrograph directories
- No persistent in-memory state between runs
- Each CLI command is independent

## Key Abstractions

**SAMModel:**
- Purpose: Wrap Facebook's Segment Anything Model
- Location: `cryoem_annotation/core/sam_model.py`
- Examples: GPU memory optimization, inference context management
- Pattern: Singleton-like (one model instance per session)

**RealTimeClickCollector:**
- Purpose: Interactive click-based segmentation interface
- Location: `cryoem_annotation/annotation/click_collector.py`
- Examples: Real-time mask prediction, undo support, memory-efficient overlays
- Pattern: Event-driven observer (matplotlib callbacks)

**CachedSegmentationData:**
- Purpose: Lazy-loaded caching for contours/centroids
- Location: `cryoem_annotation/labeling/labeler.py`
- Examples: 50-80% faster redraws via cached geometry
- Pattern: Lazy initialization with memoization

**Config:**
- Purpose: YAML + environment variable configuration
- Location: `cryoem_annotation/config.py`
- Examples: Nested dict access with defaults
- Pattern: Cascading configuration (CLI > env vars > YAML > defaults)

## Entry Points

**cryoem-annotate:**
- Location: `cryoem_annotation/cli/annotate.py:main()`
- Triggers: `cryoem-annotate --micrographs <dir> --checkpoint <path>`
- Responsibilities: Initialize SAM, run annotation loop, save results

**cryoem-label:**
- Location: `cryoem_annotation/cli/label.py:main()`
- Triggers: `cryoem-label --results <dir>`
- Responsibilities: Load existing results, interactive labeling, update metadata

**cryoem-extract:**
- Location: `cryoem_annotation/cli/extract.py:main()`
- Triggers: `cryoem-extract --results <dir> --output <file>`
- Responsibilities: Aggregate data, calculate metrics, export CSV/JSON

## Error Handling

**Strategy:** Throw exceptions, catch at CLI layer, display friendly errors

**Patterns:**
- Domain layer raises ValueError, FileNotFoundError with context
- CLI layer catches and formats error messages with `[ERROR]` prefix
- Optional operations (pixel size extraction, GPU) gracefully degrade
- Matplotlib backend failures fall back to console input mode

## Cross-Cutting Concerns

**Logging:**
- StatusLogger class in `cryoem_annotation/core/logging_utils.py`
- Prefixes: `[OK]`, `[ERROR]`, `[WARNING]` (ASCII only)
- Console output via Click echo

**Memory Optimization:**
- GPU cache clearing on file transitions (`torch.cuda.empty_cache()`)
- Bounded overlays for mask visualization (90%+ memory reduction)
- Lazy imports for fast CLI startup

**Configuration:**
- Cascading: CLI args > environment vars > YAML config > defaults
- All tools accept `--config` flag for YAML file override

---

*Architecture analysis: 2025-12-29*
*Update when major patterns change*
