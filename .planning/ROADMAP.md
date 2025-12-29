# Roadmap: CryoEM Annotation Tool v2

## Overview

Upgrade the cryo-EM annotation tool to support multi-grid datasets from the motion correction transfer pipeline. The journey starts with cleaning up technical debt to make changes safer, then adds grid-aware data structures, extends the UI for multi-grid navigation, and finally updates extraction to produce grid-organized output.

## Domain Expertise

None

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: Foundation Refactoring** - Clean up technical debt, extract shared utilities
- [x] **Phase 2: Multi-Grid Data Model** - Grid-aware data structures and input parsing
- [ ] **Phase 3: Grid-Aware UI** - Navigation and display with grid context
- [ ] **Phase 4: Output & Extraction** - Grid-organized output and per-grid summaries

## Phase Details

### Phase 1: Foundation Refactoring
**Goal**: Clean up technical debt to make multi-grid integration safer. Extract duplicate matplotlib backend initialization to shared module, address SAM prediction logic duplication, improve code organization in large files.
**Depends on**: Nothing (first phase)
**Research**: Unlikely (internal code reorganization, established patterns)
**Plans**: 2 plans

Plans:
- [x] 01-01: Extract matplotlib backend utility, improve backend logging
- [x] 01-02: Deduplicate SAM prediction logic, add .env.example

Key work:
- Extract matplotlib backend initialization to shared utility
- Deduplicate SAM prediction logic in click_collector.py
- Add `.env.example` documentation
- Improve platform-specific error detection

### Phase 2: Multi-Grid Data Model
**Goal**: Add grid-aware data structures that understand the `motion_corrected/{Grid}/` structure from the transfer pipeline. Enable loading micrographs from multiple grids in a single session.
**Depends on**: Phase 1
**Research**: Unlikely (internal data structures, building on existing I/O code)
**Plans**: 1 plan

Plans:
- [x] 02-01: Create GridDataset class and MicrographItem dataclass with tests

Key work:
- Create GridDataset class to represent multi-grid structure
- Parse `motion_corrected/{Grid}/*.mrc` directory layout
- Track grid context for each micrograph
- Update session state to handle multiple grids

### Phase 3: Grid-Aware UI
**Goal**: Extend the matplotlib UI to display grid context and enable navigation between grids. Users can jump to any micrograph from any grid within a single session.
**Depends on**: Phase 2
**Research**: Unlikely (extending existing matplotlib UI patterns)
**Plans**: TBD

Key work:
- Add grid context to display (title bar or panel)
- Implement grid-aware file browser navigation
- Enable jumping between grids without restart
- Consider grid-level progress tracking

### Phase 4: Output & Extraction
**Goal**: Update output organization to mirror input structure (`results/{Grid}/`). Extend extraction to include grid context and produce per-grid summaries alongside combined output.
**Depends on**: Phase 3
**Research**: Unlikely (extending existing CSV/JSON output patterns)
**Plans**: TBD

Key work:
- Organize output as `results/{Grid}/{micrograph}/`
- Add grid column to extraction CSV output
- Produce per-grid summary statistics
- Update metadata.json to include grid context

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Foundation Refactoring | 2/2 | Complete | 2025-12-29 |
| 2. Multi-Grid Data Model | 1/1 | Complete | 2025-12-29 |
| 3. Grid-Aware UI | 0/TBD | Not started | - |
| 4. Output & Extraction | 0/TBD | Not started | - |
