"""Tests for extraction module."""

import pytest
import json
import csv
from pathlib import Path

from cryoem_annotation.extraction.extractor import (
    extract_segmentation_data,
    save_metadata_csv,
    save_results_csv,
    _detect_multi_grid_structure,
)


@pytest.fixture
def sample_results_folder(temp_output_dir, sample_segmentations):
    """Create a sample results folder with metadata files."""
    # Create two micrograph result directories
    for mic_idx, mic_name in enumerate(["micro_001", "micro_002"]):
        mic_dir = temp_output_dir / mic_name
        mic_dir.mkdir()

        # Create metadata.json
        metadata = {
            "filename": f"{mic_name}.mrc",
            "pixel_size_nm": 0.5 if mic_idx == 0 else None,
            "segmentations": [
                {
                    "click_index": i + 1,
                    "click_coords": [100 * i, 200 * i],
                    "mask_score": 0.9 - (i * 0.1),
                    "mask_area": 5000 + (i * 1000),
                    "label": i if i < 2 else None,
                }
                for i in range(3)
            ],
        }
        with open(mic_dir / "metadata.json", "w") as f:
            json.dump(metadata, f)

    return temp_output_dir


class TestExtractSegmentationData:
    """Tests for extract_segmentation_data function."""

    def test_extracts_all_segmentations(self, sample_results_folder):
        """Test that all segmentations are extracted."""
        metadata, results, per_grid_data, total = extract_segmentation_data(sample_results_folder)

        # 2 micrographs x 3 segmentations each = 6 total
        assert len(metadata) == 6
        assert len(results) == 6
        assert total == 2
        # Single-folder mode should have empty per_grid_data
        assert per_grid_data == {}

    def test_creates_unique_segmentation_ids(self, sample_results_folder):
        """Test that segmentation IDs are unique integers."""
        metadata, _, _, _ = extract_segmentation_data(sample_results_folder)

        ids = [m["segmentation_id"] for m in metadata]
        assert len(ids) == len(set(ids))
        # IDs should be integers 1, 2, 3, ...
        assert all(isinstance(id, int) for id in ids)
        assert ids == list(range(1, len(ids) + 1))

    def test_calculates_diameter_nm_when_pixel_size_available(self, sample_results_folder):
        """Test diameter_nm calculation when pixel size is available."""
        metadata, results, _, _ = extract_segmentation_data(sample_results_folder)

        # Build lookup from segmentation_id to micrograph_name
        id_to_mic = {m["segmentation_id"]: m["micrograph_name"] for m in metadata}

        # First micrograph has pixel_size_nm=0.5
        micro_001_results = [r for r in results if id_to_mic[r["segmentation_id"]] == "micro_001"]
        assert all(r["diameter_nm"] is not None for r in micro_001_results)

        # Second micrograph has no pixel size
        micro_002_results = [r for r in results if id_to_mic[r["segmentation_id"]] == "micro_002"]
        assert all(r["diameter_nm"] is None for r in micro_002_results)

    def test_pixel_size_override(self, sample_results_folder):
        """Test that pixel_size_override works."""
        _, results, _, _ = extract_segmentation_data(sample_results_folder, pixel_size_override=1.0)

        # All should have diameter_nm calculated
        assert all(r["diameter_nm"] is not None for r in results)

    def test_empty_folder(self, temp_output_dir):
        """Test extraction from empty folder."""
        metadata, results, per_grid_data, total = extract_segmentation_data(temp_output_dir)

        assert metadata == []
        assert results == []
        assert per_grid_data == {}
        assert total == 0


class TestSaveMetadataCsv:
    """Tests for save_metadata_csv function."""

    def test_creates_csv_file(self, temp_output_dir):
        """Test that CSV file is created."""
        data = [
            {
                "segmentation_id": "test_seg001",
                "grid_name": None,
                "micrograph_name": "test",
                "click_index": 1,
                "click_coords": [100, 200],
                "mask_score": 0.95,
                "area_pixels": 5000,
            }
        ]
        output_path = temp_output_dir / "metadata.csv"

        save_metadata_csv(data, output_path)

        assert output_path.exists()

    def test_csv_has_correct_headers(self, temp_output_dir):
        """Test that CSV has correct headers."""
        data = [
            {
                "segmentation_id": "test_seg001",
                "grid_name": None,
                "micrograph_name": "test",
                "click_index": 1,
                "click_coords": [100, 200],
                "mask_score": 0.95,
                "area_pixels": 5000,
            }
        ]
        output_path = temp_output_dir / "metadata.csv"

        save_metadata_csv(data, output_path)

        with open(output_path) as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames

        expected = ["segmentation_id", "grid_name", "micrograph_name", "click_index",
                    "click_coords", "mask_score", "area_pixels"]
        assert headers == expected


class TestSaveResultsCsv:
    """Tests for save_results_csv function."""

    def test_creates_csv_file(self, temp_output_dir):
        """Test that CSV file is created."""
        data = [
            {
                "segmentation_id": "test_seg001",
                "label": 1,
                "diameter_nm": 39.89,
            }
        ]
        output_path = temp_output_dir / "results.csv"

        save_results_csv(data, output_path)

        assert output_path.exists()

    def test_csv_has_correct_headers(self, temp_output_dir):
        """Test that CSV has correct headers."""
        data = [
            {
                "segmentation_id": "test_seg001",
                "label": 1,
                "diameter_nm": 39.89,
            }
        ]
        output_path = temp_output_dir / "results.csv"

        save_results_csv(data, output_path)

        with open(output_path) as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames

        expected = ["segmentation_id", "label", "diameter_nm"]
        assert headers == expected

    def test_handles_none_diameter_nm(self, temp_output_dir):
        """Test that None diameter_nm is handled correctly."""
        data = [
            {
                "segmentation_id": "test_seg001",
                "label": 1,
                "diameter_nm": None,
            }
        ]
        output_path = temp_output_dir / "results.csv"

        save_results_csv(data, output_path)

        with open(output_path) as f:
            reader = csv.DictReader(f)
            row = next(reader)

        assert row["diameter_nm"] == ""


@pytest.fixture
def multi_grid_results_folder(temp_output_dir):
    """Create multi-grid results structure."""
    for grid_name in ["Grid1", "Grid2"]:
        for mic_name in ["micro_001", "micro_002"]:
            mic_dir = temp_output_dir / grid_name / mic_name
            mic_dir.mkdir(parents=True)
            metadata = {
                "filename": f"{mic_name}.mrc",
                "grid_name": grid_name,
                "pixel_size_nm": 0.5,
                "segmentations": [
                    {"click_index": 1, "click_coords": [100, 200],
                     "mask_score": 0.9, "mask_area": 5000, "label": 1}
                ],
            }
            with open(mic_dir / "metadata.json", "w") as f:
                json.dump(metadata, f)
    return temp_output_dir


class TestMultiGridExtraction:
    """Tests for multi-grid extraction functionality."""

    def test_detects_multi_grid_structure(self, multi_grid_results_folder):
        """Test that multi-grid structure is detected correctly."""
        is_multi_grid = _detect_multi_grid_structure(multi_grid_results_folder)
        assert is_multi_grid is True

    def test_detects_single_folder_structure(self, sample_results_folder):
        """Test that single-folder structure is detected correctly."""
        is_multi_grid = _detect_multi_grid_structure(sample_results_folder)
        assert is_multi_grid is False

    def test_extracts_grid_name_from_path(self, multi_grid_results_folder):
        """Test that grid_name is correctly extracted from multi-grid folder structure."""
        metadata, _, per_grid_data, _ = extract_segmentation_data(multi_grid_results_folder)

        # All entries should have grid_name
        assert all(m["grid_name"] is not None for m in metadata)

        # Should have entries from both grids
        grid_names = set(m["grid_name"] for m in metadata)
        assert grid_names == {"Grid1", "Grid2"}

        # Per-grid data should have both grids
        assert set(per_grid_data.keys()) == {"Grid1", "Grid2"}

    def test_grid_name_in_metadata_csv(self, multi_grid_results_folder, temp_output_dir):
        """Test that grid_name column appears in CSV output with correct values."""
        metadata, _, _, _ = extract_segmentation_data(multi_grid_results_folder)

        output_path = temp_output_dir / "output" / "metadata.csv"
        save_metadata_csv(metadata, output_path)

        # Read CSV and verify grid_name column
        with open(output_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # grid_name should be in headers
        assert "grid_name" in reader.fieldnames

        # All rows should have grid_name values
        grid_names = set(row["grid_name"] for row in rows)
        assert grid_names == {"Grid1", "Grid2"}

    def test_single_folder_has_none_grid_name(self, sample_results_folder):
        """Test that single-folder mode sets grid_name=None in output."""
        metadata, _, per_grid_data, _ = extract_segmentation_data(sample_results_folder)

        # All entries should have grid_name=None for single-folder
        assert all(m["grid_name"] is None for m in metadata)

        # Per-grid data should be empty for single-folder
        assert per_grid_data == {}

    def test_mixed_grids_separate_counts(self, temp_output_dir):
        """Test that extraction counts are accurate for multiple grids."""
        # Create unequal distribution: Grid1 has 3 mics, Grid2 has 1 mic
        for grid_name, mic_count, segs_per_mic in [("Grid1", 3, 2), ("Grid2", 1, 5)]:
            for i in range(mic_count):
                mic_dir = temp_output_dir / grid_name / f"micro_{i:03d}"
                mic_dir.mkdir(parents=True)
                metadata = {
                    "filename": f"micro_{i:03d}.mrc",
                    "grid_name": grid_name,
                    "pixel_size_nm": 0.5,
                    "segmentations": [
                        {"click_index": j + 1, "click_coords": [100, 200],
                         "mask_score": 0.9, "mask_area": 5000,
                         "label": 1 if j == 0 else None}
                        for j in range(segs_per_mic)
                    ],
                }
                with open(mic_dir / "metadata.json", "w") as f:
                    json.dump(metadata, f)

        metadata, results, per_grid_data, total = extract_segmentation_data(temp_output_dir)

        # Total: 3*2 + 1*5 = 11 segmentations, 4 micrographs
        assert len(metadata) == 11
        assert total == 4

        # Grid1: 3 micrographs, 6 segmentations
        grid1_metadata = [m for m in metadata if m["grid_name"] == "Grid1"]
        assert len(grid1_metadata) == 6
        assert len(set(m["micrograph_name"] for m in grid1_metadata)) == 3

        # Grid2: 1 micrograph, 5 segmentations
        grid2_metadata = [m for m in metadata if m["grid_name"] == "Grid2"]
        assert len(grid2_metadata) == 5
        assert len(set(m["micrograph_name"] for m in grid2_metadata)) == 1

        # Verify per-grid data has local IDs starting at 1
        grid1_per_grid_metadata, _ = per_grid_data["Grid1"]
        grid2_per_grid_metadata, _ = per_grid_data["Grid2"]

        # Grid1 local IDs: 1-6
        grid1_ids = [m["segmentation_id"] for m in grid1_per_grid_metadata]
        assert grid1_ids == list(range(1, 7))

        # Grid2 local IDs: 1-5
        grid2_ids = [m["segmentation_id"] for m in grid2_per_grid_metadata]
        assert grid2_ids == list(range(1, 6))

        # Per-grid metadata should NOT have grid_name field
        assert "grid_name" not in grid1_per_grid_metadata[0]
        assert "grid_name" not in grid2_per_grid_metadata[0]

    def test_include_grid_name_false_excludes_column(self, multi_grid_results_folder, temp_output_dir):
        """Test that include_grid_name=False excludes grid_name column from CSV."""
        metadata, _, per_grid_data, _ = extract_segmentation_data(multi_grid_results_folder)

        # Test with include_grid_name=False (like per-grid files)
        grid_metadata, _ = per_grid_data["Grid1"]
        output_path = temp_output_dir / "output" / "per_grid_metadata.csv"
        save_metadata_csv(grid_metadata, output_path, include_grid_name=False)

        # Read CSV and verify NO grid_name column
        with open(output_path) as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames

        assert "grid_name" not in headers
        expected = ["segmentation_id", "micrograph_name", "click_index",
                    "click_coords", "mask_score", "area_pixels"]
        assert headers == expected

    def test_include_grid_name_true_includes_column(self, multi_grid_results_folder, temp_output_dir):
        """Test that include_grid_name=True (default) includes grid_name column in CSV."""
        metadata, _, _, _ = extract_segmentation_data(multi_grid_results_folder)

        output_path = temp_output_dir / "output" / "summary_metadata.csv"
        save_metadata_csv(metadata, output_path, include_grid_name=True)

        # Read CSV and verify grid_name column is present
        with open(output_path) as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames

        assert "grid_name" in headers
        expected = ["segmentation_id", "grid_name", "micrograph_name", "click_index",
                    "click_coords", "mask_score", "area_pixels"]
        assert headers == expected


class TestPerGridExtractionOutput:
    """Tests for per-grid extraction output functionality."""

    def test_extract_results_creates_per_grid_files(self, multi_grid_results_folder):
        """Test that extract_results creates extraction files in each grid folder."""
        from cryoem_annotation.extraction.extractor import extract_results

        extract_results(multi_grid_results_folder, output_format="both")

        # Check per-grid files exist
        for grid_name in ["Grid1", "Grid2"]:
            grid_dir = multi_grid_results_folder / grid_name
            assert (grid_dir / "extraction_metadata.csv").exists(), f"Missing {grid_name}/extraction_metadata.csv"
            assert (grid_dir / "extraction_results.csv").exists(), f"Missing {grid_name}/extraction_results.csv"
            assert (grid_dir / "extraction.json").exists(), f"Missing {grid_name}/extraction.json"

    def test_extract_results_creates_summary_files(self, multi_grid_results_folder):
        """Test that extract_results creates summary files at results root."""
        from cryoem_annotation.extraction.extractor import extract_results

        extract_results(multi_grid_results_folder, output_format="both")

        # Check summary files exist at root
        assert (multi_grid_results_folder / "summary_metadata.csv").exists()
        assert (multi_grid_results_folder / "summary_results.csv").exists()
        assert (multi_grid_results_folder / "summary.json").exists()

    def test_per_grid_csv_has_local_ids(self, multi_grid_results_folder):
        """Test that per-grid CSVs have local IDs starting at 1."""
        from cryoem_annotation.extraction.extractor import extract_results

        extract_results(multi_grid_results_folder, output_format="csv")

        # Read Grid1 extraction CSV
        grid1_csv = multi_grid_results_folder / "Grid1" / "extraction_metadata.csv"
        with open(grid1_csv) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # Grid1 has 2 micrographs, 1 segmentation each = 2 total, IDs should be 1, 2
        ids = [int(row["segmentation_id"]) for row in rows]
        assert ids == [1, 2]

    def test_per_grid_csv_no_grid_name_column(self, multi_grid_results_folder):
        """Test that per-grid CSVs do NOT have grid_name column."""
        from cryoem_annotation.extraction.extractor import extract_results

        extract_results(multi_grid_results_folder, output_format="csv")

        # Read Grid1 extraction CSV
        grid1_csv = multi_grid_results_folder / "Grid1" / "extraction_metadata.csv"
        with open(grid1_csv) as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames

        assert "grid_name" not in headers

    def test_summary_csv_has_global_ids(self, multi_grid_results_folder):
        """Test that summary CSV has global IDs."""
        from cryoem_annotation.extraction.extractor import extract_results

        extract_results(multi_grid_results_folder, output_format="csv")

        # Read summary CSV
        summary_csv = multi_grid_results_folder / "summary_metadata.csv"
        with open(summary_csv) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # 2 grids x 2 micrographs x 1 segmentation = 4 total, IDs should be 1-4
        ids = [int(row["segmentation_id"]) for row in rows]
        assert ids == [1, 2, 3, 4]

    def test_summary_csv_has_grid_name_column(self, multi_grid_results_folder):
        """Test that summary CSV has grid_name column."""
        from cryoem_annotation.extraction.extractor import extract_results

        extract_results(multi_grid_results_folder, output_format="csv")

        # Read summary CSV
        summary_csv = multi_grid_results_folder / "summary_metadata.csv"
        with open(summary_csv) as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames

        assert "grid_name" in headers

    def test_single_folder_mode_unchanged(self, sample_results_folder):
        """Test that single-folder mode behavior is unchanged (extraction_* files)."""
        from cryoem_annotation.extraction.extractor import extract_results

        extract_results(sample_results_folder, output_format="both")

        # Check extraction files exist at root (single-folder mode)
        assert (sample_results_folder / "extraction_metadata.csv").exists()
        assert (sample_results_folder / "extraction_results.csv").exists()
        assert (sample_results_folder / "extraction.json").exists()

        # Summary files should NOT exist in single-folder mode
        assert not (sample_results_folder / "summary_metadata.csv").exists()
        assert not (sample_results_folder / "summary_results.csv").exists()
        assert not (sample_results_folder / "summary.json").exists()
