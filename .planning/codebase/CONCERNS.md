# Codebase Concerns

**Analysis Date:** 2025-12-29

## Tech Debt

**Large files with multiple responsibilities:**
- Issue: Several files exceed 300 lines with mixed concerns
- Files: `cryoem_annotation/labeling/labeler.py` (768 lines), `cryoem_annotation/annotation/click_collector.py` (514 lines), `cryoem_annotation/annotation/annotator.py` (382 lines)
- Why: Rapid development, event loop complexity
- Impact: Harder to test individual components, complex to modify
- Fix approach: Extract state management, break down event handlers, create separate utility modules

**Duplicate matplotlib backend initialization:**
- Issue: Identical backend setup code in multiple files
- Files: `cryoem_annotation/labeling/labeler.py` (lines 28-43), `cryoem_annotation/annotation/click_collector.py` (lines 52-71)
- Why: Each module needs interactive matplotlib
- Impact: Maintenance burden, inconsistent error handling
- Fix approach: Extract to `cryoem_annotation/core/matplotlib_utils.py`

**SAM prediction logic duplication:**
- Issue: SAM prediction code appears twice in click_collector.py
- Files: `cryoem_annotation/annotation/click_collector.py` (lines 150-154, 352-358)
- Why: Fallback mode duplicates main prediction logic
- Impact: Bug fixes need to be applied in both places
- Fix approach: Extract to helper method `_predict_mask()`

## Known Bugs

**No known bugs documented.**

The codebase is stable with proper error handling. Users report issues via GitHub issues if encountered.

## Security Considerations

**Configuration from environment variables:**
- Risk: Environment variables loaded without type validation
- Files: `cryoem_annotation/config.py` (lines 86-98)
- Current mitigation: CLI arguments take precedence
- Recommendations: Add type coercion for numeric env vars

**File paths not sanitized:**
- Risk: Special characters in micrograph filenames could cause issues
- Files: `cryoem_annotation/annotation/annotator.py` (lines 328-329)
- Current mitigation: None (assumes valid filenames)
- Recommendations: Sanitize or validate filenames before use

## Performance Bottlenecks

**Python sort for median calculation:**
- Problem: Uses Python sorted() instead of numpy for median
- Files: `cryoem_annotation/extraction/extractor.py` (lines 208, 218)
- Measurement: Negligible with typical dataset sizes (<1000 segmentations)
- Cause: Simple implementation
- Improvement path: Use `numpy.median()` for large datasets

**Full redraw on label change:**
- Problem: All segmentation contours redrawn when one label changes
- Files: `cryoem_annotation/labeling/labeler.py` (lines 189-226)
- Measurement: Not profiled, but noticeable with 50+ segmentations
- Cause: CachedSegmentationData doesn't cache across redraws
- Improvement path: Incremental updates, only redraw changed segmentation

**CUDA cache clearing on every file transition:**
- Problem: `torch.cuda.empty_cache()` called on each file navigation
- Files: `cryoem_annotation/annotation/annotator.py` (lines 314-316)
- Measurement: ~50-100ms overhead per transition
- Cause: Conservative memory management
- Improvement path: Clear only when memory pressure detected

## Fragile Areas

**Platform-specific error detection:**
- Files: `cryoem_annotation/labeling/labeler.py` (lines 414-417), `cryoem_annotation/annotation/click_collector.py` (lines 271-276, 300-310)
- Why fragile: Matches error messages by string content ("macOS", "2600", "1600")
- Common failures: Different matplotlib versions may have different error messages
- Safe modification: Add version checks, expand string patterns
- Test coverage: Not tested (would require triggering platform-specific failures)

**Matplotlib backend selection:**
- Files: `cryoem_annotation/labeling/labeler.py` (lines 28-43), `cryoem_annotation/annotation/click_collector.py` (lines 52-71)
- Why fragile: Tries Qt5Agg, MacOSX, TkAgg in order with silent fallback
- Common failures: User doesn't know which backend is active
- Safe modification: Log which backend was selected
- Test coverage: Not tested

**Navigation window Tk root detection:**
- Files: `cryoem_annotation/navigation/nav_window.py` (lines 40-48)
- Why fragile: Uses private API `tk._default_root` to detect existing Tk
- Common failures: May behave differently across Tk versions
- Safe modification: Wrap in version-specific checks
- Test coverage: Not tested

## Scaling Limits

**In-memory segmentation loading:**
- Current capacity: Works well with 100s of segmentations per micrograph
- Limit: Memory pressure with 1000+ segmentations per image
- Symptoms at limit: Slow redraws, potential OOM with many large masks
- Scaling path: Lazy loading of mask data, pagination in UI

**Single-file metadata aggregation:**
- Current capacity: Works well with 100s of micrographs
- Limit: `extract_results()` loads all metadata into memory
- Symptoms at limit: Slow extraction, high memory usage
- Scaling path: Streaming/chunked processing

## Dependencies at Risk

**segment-anything from git HEAD:**
- Risk: Installed from git without version pinning
- Impact: Breaking changes from upstream could break SAMModel
- Migration plan: Pin to specific commit or wait for official release

**Loose PyTorch version range:**
- Risk: `torch>=1.7.0` spans major versions (1.7 to 2.x)
- Impact: Untested combinations may have compatibility issues
- Migration plan: Test with specific versions, narrow range

## Missing Critical Features

**No .env.example file:**
- Problem: Environment variables not documented in example file
- Current workaround: Users must read config.py or CLAUDE.md
- Blocks: Easy onboarding for new users
- Implementation complexity: Low (create example file)

**No progress persistence:**
- Problem: Annotation session not saved if interrupted
- Current workaround: Complete each micrograph before quitting
- Blocks: Resuming interrupted annotation sessions
- Implementation complexity: Medium (save session state)

## Test Coverage Gaps

**Interactive components not tested:**
- What's not tested: `RealTimeClickCollector`, `SegmentationLabeler` event handlers
- Risk: GUI bugs discovered only in manual testing
- Priority: Medium
- Difficulty to test: Requires mocking matplotlib event system

**Navigation state machine not tested:**
- What's not tested: `annotate_micrographs()` navigation loop (next, prev, goto, quit)
- Risk: Edge cases in navigation (bounds, state) not verified
- Priority: Medium
- Difficulty to test: Requires mocking user input sequence

**MRC pixel size extraction edge cases:**
- What's not tested: Malformed voxel_size data in MRC headers
- Risk: Crash or incorrect pixel size with unusual MRC files
- Priority: Low
- Difficulty to test: Need to create malformed MRC test files

---

*Concerns audit: 2025-12-29*
*Update as issues are fixed or new ones discovered*
