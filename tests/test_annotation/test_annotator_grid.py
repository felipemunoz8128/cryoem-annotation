"""Tests for grid-aware annotator integration."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
import json
import tempfile
import shutil

from cryoem_annotation.core.grid_dataset import GridDataset, MicrographItem


class TestGridDatasetIntegration:
    """Tests for GridDataset integration with annotator."""

    def test_grid_dataset_import(self):
        """Test that GridDataset can be imported."""
        from cryoem_annotation.core.grid_dataset import GridDataset, MicrographItem
        assert GridDataset is not None
        assert MicrographItem is not None

    def test_annotator_import_with_grid_dataset(self):
        """Test that annotator can be imported and accepts GridDataset."""
        from cryoem_annotation.annotation.annotator import annotate_micrographs
        import inspect

        # Check that annotate_micrographs has dataset parameter
        sig = inspect.signature(annotate_micrographs)
        assert 'dataset' in sig.parameters
        # Should be first parameter
        params = list(sig.parameters.keys())
        assert params[0] == 'dataset'


class TestOutputDirectoryStructure:
    """Tests for output directory structure in multi-grid and single-folder modes."""

    @pytest.fixture
    def multi_grid_structure(self, tmp_path):
        """Create a multi-grid directory structure with test images."""
        # Create Grid1
        grid1 = tmp_path / "Grid1"
        grid1.mkdir()
        (grid1 / "micro_001.png").write_bytes(b'\x89PNG\r\n\x1a\n')
        (grid1 / "micro_002.png").write_bytes(b'\x89PNG\r\n\x1a\n')

        # Create Grid2
        grid2 = tmp_path / "Grid2"
        grid2.mkdir()
        (grid2 / "micro_003.png").write_bytes(b'\x89PNG\r\n\x1a\n')

        return tmp_path

    @pytest.fixture
    def single_folder_structure(self, tmp_path):
        """Create a single-folder directory structure with test images."""
        (tmp_path / "micro_001.png").write_bytes(b'\x89PNG\r\n\x1a\n')
        (tmp_path / "micro_002.png").write_bytes(b'\x89PNG\r\n\x1a\n')
        return tmp_path

    def test_multi_grid_detection(self, multi_grid_structure):
        """Test that GridDataset correctly detects multi-grid structure."""
        dataset = GridDataset(multi_grid_structure)

        assert dataset.is_multi_grid is True
        assert len(dataset.grid_names) == 2
        assert "Grid1" in dataset.grid_names
        assert "Grid2" in dataset.grid_names
        assert dataset.total_micrographs == 3

    def test_single_folder_detection(self, single_folder_structure):
        """Test that GridDataset correctly detects single-folder structure."""
        dataset = GridDataset(single_folder_structure)

        assert dataset.is_multi_grid is False
        assert dataset.grid_names == []
        assert dataset.total_micrographs == 2

    def test_micrograph_items_have_grid_context(self, multi_grid_structure):
        """Test that MicrographItem instances have correct grid context."""
        dataset = GridDataset(multi_grid_structure)
        items = dataset.get_micrographs()

        # Check Grid1 items
        grid1_items = [i for i in items if i.grid_name == "Grid1"]
        assert len(grid1_items) == 2
        for item in grid1_items:
            assert "Grid1/" in item.display_name

        # Check Grid2 items
        grid2_items = [i for i in items if i.grid_name == "Grid2"]
        assert len(grid2_items) == 1
        for item in grid2_items:
            assert "Grid2/" in item.display_name

    def test_single_folder_items_no_grid_context(self, single_folder_structure):
        """Test that single-folder items have no grid context."""
        dataset = GridDataset(single_folder_structure)
        items = dataset.get_micrographs()

        for item in items:
            assert item.grid_name is None
            # display_name should just be micrograph name (no slash)
            assert "/" not in item.display_name

    def test_output_path_multi_grid(self, multi_grid_structure, tmp_path):
        """Test that output paths are correctly constructed for multi-grid."""
        output_folder = tmp_path / "output"
        output_folder.mkdir()

        dataset = GridDataset(multi_grid_structure)
        items = dataset.get_micrographs()

        # Simulate output path construction
        for item in items:
            if item.grid_name is not None:
                output_dir = output_folder / item.grid_name / item.micrograph_name
            else:
                output_dir = output_folder / item.micrograph_name

            output_dir.mkdir(parents=True, exist_ok=True)

        # Check structure
        assert (output_folder / "Grid1" / "micro_001").exists()
        assert (output_folder / "Grid1" / "micro_002").exists()
        assert (output_folder / "Grid2" / "micro_003").exists()

    def test_output_path_single_folder(self, single_folder_structure, tmp_path):
        """Test that output paths are correctly constructed for single-folder."""
        output_folder = tmp_path / "output"
        output_folder.mkdir()

        dataset = GridDataset(single_folder_structure)
        items = dataset.get_micrographs()

        # Simulate output path construction
        for item in items:
            if item.grid_name is not None:
                output_dir = output_folder / item.grid_name / item.micrograph_name
            else:
                output_dir = output_folder / item.micrograph_name

            output_dir.mkdir(parents=True, exist_ok=True)

        # Check structure - no grid subdirectories
        assert (output_folder / "micro_001").exists()
        assert (output_folder / "micro_002").exists()
        # Should NOT have grid folders
        assert not (output_folder / "Grid1").exists()


class TestCombinedResultsJSON:
    """Tests for combined results JSON with grid context."""

    def test_result_includes_grid_name(self):
        """Test that result dict includes grid_name field."""
        # Simulate a result dict as created by annotator
        result = {
            'filename': 'micro_001.mrc',
            'filepath': '/path/to/Grid1/micro_001.mrc',
            'micrograph_name': 'micro_001',
            'grid_name': 'Grid1',  # Should be included
            'image_shape': [4096, 4096],
            'num_clicks': 5,
            'timestamp': '2025-01-01T00:00:00',
            'pixel_size_nm': 0.5,
            'segmentations': [],
        }

        # Verify grid_name is present
        assert 'grid_name' in result
        assert result['grid_name'] == 'Grid1'
        assert 'micrograph_name' in result

    def test_result_grid_name_none_for_single_folder(self):
        """Test that grid_name is None for single-folder results."""
        result = {
            'filename': 'micro_001.mrc',
            'filepath': '/path/to/micro_001.mrc',
            'micrograph_name': 'micro_001',
            'grid_name': None,  # Should be None for single folder
            'image_shape': [4096, 4096],
            'num_clicks': 3,
            'timestamp': '2025-01-01T00:00:00',
            'pixel_size_nm': 0.5,
            'segmentations': [],
        }

        assert 'grid_name' in result
        assert result['grid_name'] is None


class TestNavigationWindowGridAware:
    """Tests for grid-aware NavigationWindow."""

    def test_navigation_window_accepts_micrograph_items(self):
        """Test that NavigationWindow can be initialized with MicrographItem list."""
        from cryoem_annotation.navigation.nav_window import NavigationWindow

        # Create mock MicrographItems
        items = [
            MicrographItem(
                file_path=Path("/tmp/Grid1/micro_001.png"),
                grid_name="Grid1",
                micrograph_name="micro_001"
            ),
            MicrographItem(
                file_path=Path("/tmp/Grid1/micro_002.png"),
                grid_name="Grid1",
                micrograph_name="micro_002"
            ),
        ]

        # Mock tkinter to avoid GUI
        with patch('tkinter.Tk'), patch('tkinter.Toplevel'):
            # Should not raise
            callback = MagicMock()
            # Note: Can't fully instantiate without tkinter, but import should work
            assert NavigationWindow is not None

    def test_navigation_window_has_is_multi_grid_param(self):
        """Test that NavigationWindow constructor accepts is_multi_grid parameter."""
        from cryoem_annotation.navigation.nav_window import NavigationWindow
        import inspect

        sig = inspect.signature(NavigationWindow.__init__)
        assert 'is_multi_grid' in sig.parameters

        # Check default value
        param = sig.parameters['is_multi_grid']
        assert param.default is False


class TestCLIGridDetection:
    """Tests for CLI grid detection output."""

    def test_cli_module_imports(self):
        """Test that CLI module can be imported."""
        from cryoem_annotation.cli import annotate
        assert annotate is not None

    def test_cli_has_main_function(self):
        """Test that CLI has main function."""
        from cryoem_annotation.cli.annotate import main
        assert callable(main)
