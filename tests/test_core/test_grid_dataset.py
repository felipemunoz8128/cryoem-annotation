"""Tests for GridDataset and MicrographItem classes."""

import pytest
from pathlib import Path

from cryoem_annotation.core.grid_dataset import GridDataset, MicrographItem


class TestMicrographItem:
    """Tests for MicrographItem dataclass."""

    def test_display_name_with_grid(self):
        """Verify display_name returns 'grid/name' format when grid is set."""
        item = MicrographItem(
            file_path=Path("/data/Grid1/micrograph_001.mrc"),
            grid_name="Grid1",
            micrograph_name="micrograph_001"
        )
        assert item.display_name == "Grid1/micrograph_001"

    def test_display_name_without_grid(self):
        """Verify display_name returns just name when grid is None."""
        item = MicrographItem(
            file_path=Path("/data/micrograph_001.mrc"),
            grid_name=None,
            micrograph_name="micrograph_001"
        )
        assert item.display_name == "micrograph_001"

    def test_str_representation(self):
        """Verify __str__ matches display_name."""
        item = MicrographItem(
            file_path=Path("/data/Grid2/mic_002.mrc"),
            grid_name="Grid2",
            micrograph_name="mic_002"
        )
        assert str(item) == item.display_name
        assert str(item) == "Grid2/mic_002"

    def test_frozen_dataclass(self):
        """Verify MicrographItem is immutable (frozen=True)."""
        item = MicrographItem(
            file_path=Path("/data/test.mrc"),
            grid_name="Grid1",
            micrograph_name="test"
        )
        with pytest.raises(AttributeError):
            item.grid_name = "Grid2"


class TestGridDataset:
    """Tests for GridDataset class."""

    def test_single_folder_detection(self, tmp_path):
        """Verify single-folder mode: is_multi_grid=False, grid_names=[]."""
        # Create flat folder with images
        (tmp_path / "mic1.mrc").touch()
        (tmp_path / "mic2.mrc").touch()
        (tmp_path / "mic3.mrc").touch()

        dataset = GridDataset(tmp_path)

        assert dataset.is_multi_grid is False
        assert dataset.grid_names == []
        assert dataset.total_micrographs == 3

    def test_multi_grid_detection(self, tmp_path):
        """Verify multi-grid mode: is_multi_grid=True, grid_names sorted."""
        # Create multi-grid structure
        grid1 = tmp_path / "Grid1"
        grid2 = tmp_path / "Grid2"
        grid3 = tmp_path / "Grid3"
        grid1.mkdir()
        grid2.mkdir()
        grid3.mkdir()

        (grid1 / "mic1.mrc").touch()
        (grid1 / "mic2.mrc").touch()
        (grid2 / "mic1.mrc").touch()
        (grid2 / "mic2.mrc").touch()
        (grid2 / "mic3.mrc").touch()
        (grid3 / "mic1.mrc").touch()

        dataset = GridDataset(tmp_path)

        assert dataset.is_multi_grid is True
        assert dataset.grid_names == ["Grid1", "Grid2", "Grid3"]

    def test_total_micrographs_count(self, tmp_path):
        """Verify total_micrographs counts correctly across grids."""
        grid1 = tmp_path / "Grid1"
        grid2 = tmp_path / "Grid2"
        grid1.mkdir()
        grid2.mkdir()

        (grid1 / "mic1.mrc").touch()
        (grid1 / "mic2.mrc").touch()
        (grid2 / "mic1.mrc").touch()
        (grid2 / "mic2.mrc").touch()
        (grid2 / "mic3.mrc").touch()

        dataset = GridDataset(tmp_path)

        assert dataset.total_micrographs == 5

    def test_get_micrographs_all(self, tmp_path):
        """Verify get_micrographs() returns all in sorted order."""
        grid1 = tmp_path / "GridA"
        grid2 = tmp_path / "GridB"
        grid1.mkdir()
        grid2.mkdir()

        (grid1 / "mic_z.mrc").touch()
        (grid1 / "mic_a.mrc").touch()
        (grid2 / "mic_m.mrc").touch()

        dataset = GridDataset(tmp_path)
        all_mics = dataset.get_micrographs()

        assert len(all_mics) == 3
        # Should be sorted by grid first (GridA before GridB)
        assert all_mics[0].grid_name == "GridA"
        assert all_mics[1].grid_name == "GridA"
        assert all_mics[2].grid_name == "GridB"

    def test_get_micrographs_by_grid(self, tmp_path):
        """Verify get_micrographs(grid_name) returns only that grid."""
        grid1 = tmp_path / "Grid1"
        grid2 = tmp_path / "Grid2"
        grid1.mkdir()
        grid2.mkdir()

        (grid1 / "mic1.mrc").touch()
        (grid1 / "mic2.mrc").touch()
        (grid2 / "mic1.mrc").touch()

        dataset = GridDataset(tmp_path)

        grid1_mics = dataset.get_micrographs("Grid1")
        grid2_mics = dataset.get_micrographs("Grid2")
        grid3_mics = dataset.get_micrographs("NonExistent")

        assert len(grid1_mics) == 2
        assert len(grid2_mics) == 1
        assert len(grid3_mics) == 0
        assert all(m.grid_name == "Grid1" for m in grid1_mics)

    def test_get_micrograph_count(self, tmp_path):
        """Verify get_micrograph_count returns correct per-grid counts."""
        grid1 = tmp_path / "Grid1"
        grid2 = tmp_path / "Grid2"
        grid1.mkdir()
        grid2.mkdir()

        (grid1 / "mic1.mrc").touch()
        (grid1 / "mic2.mrc").touch()
        (grid2 / "mic1.mrc").touch()
        (grid2 / "mic2.mrc").touch()
        (grid2 / "mic3.mrc").touch()

        dataset = GridDataset(tmp_path)

        assert dataset.get_micrograph_count("Grid1") == 2
        assert dataset.get_micrograph_count("Grid2") == 3
        assert dataset.get_micrograph_count("NonExistent") == 0

    def test_empty_directory(self, tmp_path):
        """Verify empty directory is handled gracefully."""
        dataset = GridDataset(tmp_path)

        assert dataset.is_multi_grid is False
        assert dataset.grid_names == []
        assert dataset.total_micrographs == 0
        assert dataset.get_micrographs() == []

    def test_skips_hidden_files(self, tmp_path):
        """Verify hidden files (starting with .) are ignored."""
        (tmp_path / "visible.mrc").touch()
        (tmp_path / ".hidden.mrc").touch()
        (tmp_path / ".DS_Store").touch()

        dataset = GridDataset(tmp_path)

        assert dataset.total_micrographs == 1
        mics = dataset.get_micrographs()
        assert len(mics) == 1
        assert mics[0].micrograph_name == "visible"

    def test_skips_hidden_directories(self, tmp_path):
        """Verify hidden directories are ignored."""
        visible_grid = tmp_path / "Grid1"
        hidden_grid = tmp_path / ".hidden_grid"
        visible_grid.mkdir()
        hidden_grid.mkdir()

        (visible_grid / "mic1.mrc").touch()
        (hidden_grid / "mic1.mrc").touch()

        dataset = GridDataset(tmp_path)

        assert dataset.is_multi_grid is True
        assert dataset.grid_names == ["Grid1"]
        assert dataset.total_micrographs == 1

    def test_mixed_extensions(self, tmp_path):
        """Verify all supported extensions are included."""
        (tmp_path / "image1.mrc").touch()
        (tmp_path / "image2.tif").touch()
        (tmp_path / "image3.tiff").touch()
        (tmp_path / "image4.png").touch()
        (tmp_path / "image5.jpg").touch()
        (tmp_path / "image6.jpeg").touch()
        (tmp_path / "readme.txt").touch()  # Should be ignored

        dataset = GridDataset(tmp_path)

        assert dataset.total_micrographs == 6
        names = {m.micrograph_name for m in dataset.get_micrographs()}
        assert names == {"image1", "image2", "image3", "image4", "image5", "image6"}

    def test_nonexistent_directory(self, tmp_path):
        """Verify nonexistent directory is handled gracefully."""
        nonexistent = tmp_path / "does_not_exist"
        dataset = GridDataset(nonexistent)

        assert dataset.is_multi_grid is False
        assert dataset.grid_names == []
        assert dataset.total_micrographs == 0

    def test_micrographs_have_correct_paths(self, tmp_path):
        """Verify MicrographItem file_path is correct absolute path."""
        grid = tmp_path / "Grid1"
        grid.mkdir()
        mic_file = grid / "test_mic.mrc"
        mic_file.touch()

        dataset = GridDataset(tmp_path)
        mics = dataset.get_micrographs()

        assert len(mics) == 1
        assert mics[0].file_path == mic_file
        assert mics[0].file_path.exists()
