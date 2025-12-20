"""Tests for metadata I/O module."""

import pytest
import json
from pathlib import Path

from cryoem_annotation.io.metadata import save_metadata, load_metadata


class TestSaveMetadata:
    """Tests for save_metadata function."""

    def test_save_creates_file(self, temp_output_dir):
        """Test that save_metadata creates a file."""
        metadata = {"test": "data", "number": 42}
        output_path = temp_output_dir / "test_metadata.json"

        save_metadata(metadata, output_path)

        assert output_path.exists()

    def test_saved_content_is_valid_json(self, temp_output_dir):
        """Test that saved content is valid JSON."""
        metadata = {"key": "value", "nested": {"a": 1}}
        output_path = temp_output_dir / "test.json"

        save_metadata(metadata, output_path)

        with open(output_path) as f:
            loaded = json.load(f)

        assert loaded == metadata

    def test_creates_parent_directories(self, temp_output_dir):
        """Test that parent directories are created."""
        output_path = temp_output_dir / "subdir" / "nested" / "meta.json"

        save_metadata({"test": True}, output_path)

        assert output_path.exists()


class TestLoadMetadata:
    """Tests for load_metadata function."""

    def test_load_existing_file(self, temp_metadata_file):
        """Test loading an existing metadata file."""
        result = load_metadata(temp_metadata_file)

        assert result is not None
        assert "segmentations" in result

    def test_load_nonexistent_file(self):
        """Test loading a nonexistent file returns None."""
        result = load_metadata(Path("/nonexistent/metadata.json"))

        assert result is None

    def test_load_invalid_json(self, temp_output_dir):
        """Test loading invalid JSON returns None."""
        invalid_path = temp_output_dir / "invalid.json"
        invalid_path.write_text("not valid json {{{")

        result = load_metadata(invalid_path)

        assert result is None


class TestMetadataRoundtrip:
    """Tests for save/load roundtrip."""

    def test_roundtrip_preserves_data(self, temp_output_dir):
        """Test that save then load preserves data."""
        original = {
            "filename": "test.mrc",
            "segmentations": [
                {"click_index": 1, "mask_score": 0.95},
                {"click_index": 2, "mask_score": 0.85},
            ],
            "pixel_size_nm": 0.5,
        }
        path = temp_output_dir / "roundtrip.json"

        save_metadata(original, path)
        loaded = load_metadata(path)

        assert loaded == original
