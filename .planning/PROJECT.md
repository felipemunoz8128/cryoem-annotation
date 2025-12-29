# CryoEM Annotation Tool v2

## Vision

Upgrade the cryo-EM annotation tool to integrate seamlessly with the motion correction transfer pipeline and support multi-grid datasets in a single session. The current tool works one micrograph folder at a time and doesn't understand the grid-based organization that the transfer pipeline produces. This upgrade will make the annotation workflow a natural continuation of the motion correction pipeline, allowing researchers to efficiently annotate particles across multiple grids without restarting the program or managing separate output directories.

The work combines two goals: adding new multi-grid functionality AND addressing the technical debt identified during codebase analysis. Rather than treating these as separate efforts, we'll refactor the code as we integrate — fixing concerns at the points where they intersect with the new multi-grid architecture.

## Problem

**Current state:**
- The annotation tool expects a flat folder of micrographs as input
- Users must run the tool separately for each grid, specifying new paths each time
- Output is organized per-micrograph but doesn't preserve grid context
- The transfer pipeline produces `motion_corrected/{Grid}/*.mrc` but the annotation tool doesn't understand this structure
- Technical debt has accumulated (large files, duplicate code, missing tests) making changes harder

**Pain points:**
- Friction when processing multi-grid datasets — constant path management
- Results from different grids get mixed or require manual organization
- Extraction/analysis loses grid context, making it hard to compare across grids
- Code complexity makes adding new features risky

**What this enables:**
- Researchers can point the tool at a dataset root and see all grids at once
- Navigation allows jumping between any micrograph from any grid
- Results automatically organized to mirror the input structure
- Cleaner codebase easier to maintain and extend

## Success Criteria

How we know this worked:

- [ ] Successfully annotate a real multi-grid dataset from the transfer pipeline
- [ ] Navigate between grids within a single session (grid-aware file browser)
- [ ] Output organized as `results/{Grid}/` mirroring `motion_corrected/{Grid}/`
- [ ] Extraction produces per-grid summaries in addition to combined output
- [ ] Key concerns from CONCERNS.md addressed during refactoring

## Scope

### Building
- Multi-grid input support: Accept `motion_corrected/{Grid}/` structure from transfer pipeline
- Grid-aware file browser: Show all grids in navigation, jump between any micrograph from any grid
- Grid-organized output: Results stored as `results/{Grid}/{micrograph}/` to mirror input
- Per-grid extraction: CSV/JSON output includes grid context, can summarize per-grid
- Refactored matplotlib utilities: Extract duplicate backend initialization to shared module
- Improved architecture: Break down large files where they touch multi-grid functionality

### Not Building
- Automated batch annotation (still requires user clicks)
- Real-time pipeline monitoring (won't watch for new files)
- Multi-user collaboration (single user, no shared state)
- Backward compatibility shims for old output format (clean break to new structure)

## Context

**Existing codebase:**
- Functional three-stage workflow: annotate → label → extract
- Well-tested core utilities (image loading, metadata I/O)
- SAM integration working, GPU memory management in place
- Codebase mapped in `.planning/codebase/` (7 documents)

**Transfer pipeline integration:**
- Input: `{dataset}/motion_corrected/{Grid}/*_DW.mrc` (dose-weighted micrographs)
- Grids named by the pipeline (Grid1, Grid2, etc.)
- Pixel size available in MRC headers (already supported)

**Technical debt to address (from CONCERNS.md):**
- Large files with multiple responsibilities (labeler.py 768 lines, click_collector.py 514 lines)
- Duplicate matplotlib backend initialization
- SAM prediction logic duplication in click_collector.py
- Missing `.env.example` documentation
- Platform-specific error detection is fragile

## Constraints

No hard constraints specified. Flexible on:
- Adding new dependencies if needed
- Breaking changes to internal APIs
- Restructuring the codebase

## Decisions Made

Key decisions from project exploration:

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Multi-grid UX | Grid-aware file browser | User wants to jump between any micrograph from any grid freely |
| Output structure | Mirror input exactly | `results/{Grid}/` matches `motion_corrected/{Grid}/` for consistency |
| Concerns vs integration | Both together | Refactor as we integrate, fix concerns at touchpoints |
| Priority | All equally important | Multi-grid, pipeline compat, and clean architecture all matter |

## Open Questions

Things to figure out during execution:

- [ ] How to display grid context in the matplotlib UI (title bar? separate panel?)
- [ ] Whether to add grid-level progress tracking (% complete per grid)
- [ ] Best way to structure the grid-aware navigation (tree view? flat list with grid prefix?)
- [ ] How extraction summaries should aggregate across grids

---
*Initialized: 2025-12-29*
