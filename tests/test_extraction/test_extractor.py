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
        metadata, results = extract_segmentation_data(sample_results_folder)

        # 2 micrographs x 3 segmentations each = 6 total
        assert len(metadata) == 6
        assert len(results) == 6

    def test_creates_unique_segmentation_ids(self, sample_results_folder):
        """Test that segmentation IDs are unique."""
        metadata, _ = extract_segmentation_data(sample_results_folder)

        ids = [m["segmentation_id"] for m in metadata]
        assert len(ids) == len(set(ids))

    def test_calculates_diameter_nm_when_pixel_size_available(self, sample_results_folder):
        """Test diameter_nm calculation when pixel size is available."""
        _, results = extract_segmentation_data(sample_results_folder)

        # First micrograph has pixel_size_nm=0.5
        micro_001_results = [r for r in results if "micro_001" in r["segmentation_id"]]
        assert all(r["diameter_nm"] is not None for r in micro_001_results)

        # Second micrograph has no pixel size
        micro_002_results = [r for r in results if "micro_002" in r["segmentation_id"]]
        assert all(r["diameter_nm"] is None for r in micro_002_results)

    def test_pixel_size_override(self, sample_results_folder):
        """Test that pixel_size_override works."""
        _, results = extract_segmentation_data(sample_results_folder, pixel_size_override=1.0)

        # All should have diameter_nm calculated
        assert all(r["diameter_nm"] is not None for r in results)

    def test_empty_folder(self, temp_output_dir):
        """Test extraction from empty folder."""
        metadata, results = extract_segmentation_data(temp_output_dir)

        assert metadata == []
        assert results == []


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
            }
        ]
        output_path = temp_output_dir / "metadata.csv"

        save_metadata_csv(data, output_path)

        with open(output_path) as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames

        expected = ["segmentation_id", "micrograph_name", "click_index",
                    "click_coords", "mask_score"]
        assert headers == expected


class TestSaveResultsCsv:
    """Tests for save_results_csv function."""

    def test_creates_csv_file(self, temp_output_dir):
        """Test that CSV file is created."""
        data = [
            {
                "segmentation_id": "test_seg001",
                "label": 1,
                "area_pixels": 5000,
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
                "area_pixels": 5000,
                "diameter_nm": 39.89,
            }
        ]
        output_path = temp_output_dir / "results.csv"

        save_results_csv(data, output_path)

        with open(output_path) as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames

        expected = ["segmentation_id", "label", "area_pixels", "diameter_nm"]
        assert headers == expected

    def test_handles_none_diameter_nm(self, temp_output_dir):
        """Test that None diameter_nm is handled correctly."""
        data = [
            {
                "segmentation_id": "test_seg001",
                "label": 1,
                "area_pixels": 5000,
                "diameter_nm": None,
            }
        ]
        output_path = temp_output_dir / "results.csv"

        save_results_csv(data, output_path)

        with open(output_path) as f:
            reader = csv.DictReader(f)
            row = next(reader)

        assert row["diameter_nm"] == ""
