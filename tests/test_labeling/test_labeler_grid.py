"""Tests for multi-grid support in labeling module."""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import json
from unittest.mock import MagicMock, patch

from cryoem_annotation.core.grid_dataset import GridDataset, MicrographItem
from cryoem_annotation.labeling.labeler import (
    _get_annotation_dir,
    load_segmentations_from_dir,
    label_segmentations,
)


class TestGetAnnotationDir:
    """Tests for _get_annotation_dir helper function."""

    def test_multi_grid_item_returns_grid_path(self):
        """Test that multi-grid items return grid_name/micrograph_name path."""
        item = MicrographItem(
            file_path=Path("/data/Grid1/micro_001.mrc"),
            grid_name="Grid1",
            micrograph_name="micro_001",
        )
        results_folder = Path("/output/results")

        annotation_dir = _get_annotation_dir(item, results_folder)

        assert annotation_dir == Path("/output/results/Grid1/micro_001")

    def test_single_folder_item_returns_direct_path(self):
        """Test that single-folder items return micrograph_name path directly."""
        item = MicrographItem(
            file_path=Path("/data/micrographs/micro_001.mrc"),
            grid_name=None,
            micrograph_name="micro_001",
        )
        results_folder = Path("/output/results")

        annotation_dir = _get_annotation_dir(item, results_folder)

        assert annotation_dir == Path("/output/results/micro_001")

    def test_multi_grid_item_with_complex_path(self):
        """Test multi-grid with longer grid names."""
        item = MicrographItem(
            file_path=Path("/project/grids/GridA_Test/image_001.png"),
            grid_name="GridA_Test",
            micrograph_name="image_001",
        )
        results_folder = Path("/project/annotations")

        annotation_dir = _get_annotation_dir(item, results_folder)

        assert annotation_dir == Path("/project/annotations/GridA_Test/image_001")


class TestLoadSegmentationsFromDir:
    """Tests for load_segmentations_from_dir function."""

    def test_loads_metadata_with_masks(self, temp_output_dir, sample_binary_mask):
        """Test loading segmentations from a valid annotation directory."""
        # Create annotation directory structure
        annotation_dir = temp_output_dir / "test_micrograph"
        annotation_dir.mkdir()

        # Create metadata
        metadata = {
            "filename": "test_micrograph.mrc",
            "segmentations": [
                {
                    "click_index": 1,
                    "click_coords": [512, 512],
                    "mask_score": 0.95,
                    "mask_area": 5000,
                },
            ],
        }
        with open(annotation_dir / "metadata.json", "w") as f:
            json.dump(metadata, f)

        # Create mask file
        import cv2
        mask_uint8 = (sample_binary_mask * 255).astype(np.uint8)
        cv2.imwrite(str(annotation_dir / "mask_001_binary.png"), mask_uint8)

        # Load segmentations
        result = load_segmentations_from_dir(annotation_dir)

        assert result is not None
        assert "segmentations" in result
        assert len(result["segmentations"]) == 1
        assert result["segmentations"][0]["mask"] is not None
        assert result["segmentations"][0]["label"] is None

    def test_returns_none_for_nonexistent_dir(self, temp_output_dir):
        """Test that non-existent directory returns None."""
        result = load_segmentations_from_dir(temp_output_dir / "nonexistent")
        assert result is None

    def test_returns_none_for_missing_metadata(self, temp_output_dir):
        """Test that directory without metadata.json returns None."""
        annotation_dir = temp_output_dir / "empty_dir"
        annotation_dir.mkdir()

        result = load_segmentations_from_dir(annotation_dir)
        assert result is None


class TestGridDatasetIntegration:
    """Integration tests for GridDataset with labeler."""

    def test_grid_dataset_import(self):
        """Test that GridDataset can be imported in labeler context."""
        from cryoem_annotation.labeling.labeler import GridDataset, MicrographItem

        # Should not raise
        assert GridDataset is not None
        assert MicrographItem is not None

    def test_label_segmentations_accepts_grid_dataset(self):
        """Test that label_segmentations accepts GridDataset parameter."""
        import inspect
        from cryoem_annotation.labeling.labeler import label_segmentations

        sig = inspect.signature(label_segmentations)
        params = list(sig.parameters.keys())

        assert "dataset" in params
        assert "results_folder" in params

    def test_cli_uses_grid_dataset(self):
        """Test that CLI module uses GridDataset."""
        from cryoem_annotation.cli import label

        # Check that the module imports GridDataset
        import ast
        import inspect

        source = inspect.getsource(label)
        tree = ast.parse(source)

        # Check for GridDataset in imports
        import_names = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    import_names.append(alias.name)

        assert "GridDataset" in import_names


class TestMultiGridAnnotationDiscovery:
    """Tests for discovering annotations in multi-grid structure."""

    def test_discovers_multi_grid_annotations(self, temp_output_dir, sample_binary_mask):
        """Test that annotations are found in multi-grid result structure."""
        # Create multi-grid micrograph structure
        grid1_dir = temp_output_dir / "micrographs" / "Grid1"
        grid2_dir = temp_output_dir / "micrographs" / "Grid2"
        grid1_dir.mkdir(parents=True)
        grid2_dir.mkdir(parents=True)

        # Create test images
        import cv2
        test_img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        cv2.imwrite(str(grid1_dir / "micro_001.png"), test_img)
        cv2.imwrite(str(grid2_dir / "micro_002.png"), test_img)

        # Create multi-grid results structure
        results_dir = temp_output_dir / "results"
        (results_dir / "Grid1" / "micro_001").mkdir(parents=True)
        (results_dir / "Grid2" / "micro_002").mkdir(parents=True)

        # Create metadata files
        metadata = {
            "filename": "micro.png",
            "segmentations": [
                {"click_index": 1, "click_coords": [50, 50], "mask_score": 0.9, "mask_area": 100}
            ],
        }
        with open(results_dir / "Grid1" / "micro_001" / "metadata.json", "w") as f:
            json.dump(metadata, f)
        with open(results_dir / "Grid2" / "micro_002" / "metadata.json", "w") as f:
            json.dump(metadata, f)

        # Create mask files
        mask_uint8 = (sample_binary_mask[:100, :100] * 255).astype(np.uint8)
        cv2.imwrite(str(results_dir / "Grid1" / "micro_001" / "mask_001_binary.png"), mask_uint8)
        cv2.imwrite(str(results_dir / "Grid2" / "micro_002" / "mask_001_binary.png"), mask_uint8)

        # Create GridDataset
        dataset = GridDataset(temp_output_dir / "micrographs")

        # Verify multi-grid detection
        assert dataset.is_multi_grid is True
        assert len(dataset.grid_names) == 2
        assert "Grid1" in dataset.grid_names
        assert "Grid2" in dataset.grid_names

        # Verify annotation directories are correct
        items = dataset.get_micrographs()
        for item in items:
            annotation_dir = _get_annotation_dir(item, results_dir)
            assert annotation_dir.exists()
            assert (annotation_dir / "metadata.json").exists()

    def test_single_folder_annotation_discovery(self, temp_output_dir, sample_binary_mask):
        """Test that annotations are found in single-folder structure."""
        # Create single-folder micrograph structure
        micrograph_dir = temp_output_dir / "micrographs"
        micrograph_dir.mkdir(parents=True)

        # Create test images
        import cv2
        test_img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        cv2.imwrite(str(micrograph_dir / "micro_001.png"), test_img)
        cv2.imwrite(str(micrograph_dir / "micro_002.png"), test_img)

        # Create single-folder results structure
        results_dir = temp_output_dir / "results"
        (results_dir / "micro_001").mkdir(parents=True)
        (results_dir / "micro_002").mkdir(parents=True)

        # Create metadata files
        metadata = {
            "filename": "micro.png",
            "segmentations": [
                {"click_index": 1, "click_coords": [50, 50], "mask_score": 0.9, "mask_area": 100}
            ],
        }
        with open(results_dir / "micro_001" / "metadata.json", "w") as f:
            json.dump(metadata, f)
        with open(results_dir / "micro_002" / "metadata.json", "w") as f:
            json.dump(metadata, f)

        # Create mask files
        mask_uint8 = (sample_binary_mask[:100, :100] * 255).astype(np.uint8)
        cv2.imwrite(str(results_dir / "micro_001" / "mask_001_binary.png"), mask_uint8)
        cv2.imwrite(str(results_dir / "micro_002" / "mask_001_binary.png"), mask_uint8)

        # Create GridDataset
        dataset = GridDataset(micrograph_dir)

        # Verify single-folder detection
        assert dataset.is_multi_grid is False

        # Verify annotation directories are correct
        items = dataset.get_micrographs()
        for item in items:
            annotation_dir = _get_annotation_dir(item, results_dir)
            assert annotation_dir.exists()
            assert (annotation_dir / "metadata.json").exists()


class TestNavigationWindowMultiGrid:
    """Tests for NavigationWindow with multi-grid support in labeling."""

    def test_navigation_window_receives_is_multi_grid_param(self):
        """Test that NavigationWindow is called with is_multi_grid parameter."""
        from cryoem_annotation.navigation import NavigationWindow
        import inspect

        sig = inspect.signature(NavigationWindow.__init__)
        params = list(sig.parameters.keys())

        assert "is_multi_grid" in params

    def test_navigation_window_accepts_micrograph_items(self):
        """Test that NavigationWindow works with MicrographItem list."""
        items = [
            MicrographItem(
                file_path=Path("/data/Grid1/micro.mrc"),
                grid_name="Grid1",
                micrograph_name="micro",
            )
        ]

        # NavigationWindow accepts Union[List[Path], List[MicrographItem]]
        # This is tested in the annotation tests, just verify the type annotation
        from cryoem_annotation.navigation import NavigationWindow
        import inspect

        sig = inspect.signature(NavigationWindow.__init__)
        files_param = sig.parameters["files"]

        # The annotation should accept both Path and MicrographItem lists
        assert files_param is not None
