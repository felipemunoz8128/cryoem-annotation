"""Tests for extraction module."""

import pytest
import json
import csv
from pathlib import Path

from cryoem_annotation.extraction.extractor import (
    extract_segmentation_data,
    save_metadata_csv,
    save_results_csv,
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
        metadata, results, total = extract_segmentation_data(sample_results_folder)

        # 2 micrographs x 3 segmentations each = 6 total
        assert len(metadata) == 6
        assert len(results) == 6
        assert total == 2

    def test_creates_unique_segmentation_ids(self, sample_results_folder):
        """Test that segmentation IDs are unique integers."""
        metadata, _, _ = extract_segmentation_data(sample_results_folder)

        ids = [m["segmentation_id"] for m in metadata]
        assert len(ids) == len(set(ids))
        # IDs should be integers 1, 2, 3, ...
        assert all(isinstance(id, int) for id in ids)
        assert ids == list(range(1, len(ids) + 1))

    def test_calculates_diameter_nm_when_pixel_size_available(self, sample_results_folder):
        """Test diameter_nm calculation when pixel size is available."""
        metadata, results, _ = extract_segmentation_data(sample_results_folder)

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
        _, results, _ = extract_segmentation_data(sample_results_folder, pixel_size_override=1.0)

        # All should have diameter_nm calculated
        assert all(r["diameter_nm"] is not None for r in results)

    def test_empty_folder(self, temp_output_dir):
        """Test extraction from empty folder."""
        metadata, results, total = extract_segmentation_data(temp_output_dir)

        assert metadata == []
        assert results == []
        assert total == 0


class TestSaveMetadataCsv:
    """Tests for save_metadata_csv function."""

    def test_creates_csv_file(self, temp_output_dir):
        """Test that CSV file is created."""
        data = [
            {
                "segmentation_id": "test_seg001",
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

        expected = ["segmentation_id", "micrograph_name", "click_index",
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
