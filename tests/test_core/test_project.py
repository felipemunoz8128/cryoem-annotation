"""Tests for project file management module."""

import json
import pytest
from pathlib import Path
from datetime import datetime
from unittest.mock import patch

from cryoem_annotation.core.project import (
    PROJECT_FILE,
    PROJECT_VERSION,
    find_project_file,
    load_project,
    save_project,
    resolve_paths,
)


class TestProjectFileConstant:
    """Tests for PROJECT_FILE constant."""

    def test_project_file_name(self):
        """Verify PROJECT_FILE is a hidden file."""
        assert PROJECT_FILE == ".cryoem-project.json"
        assert PROJECT_FILE.startswith(".")

    def test_project_version(self):
        """Verify PROJECT_VERSION is defined."""
        assert PROJECT_VERSION == 1


class TestFindProjectFile:
    """Tests for find_project_file function."""

    def test_find_project_file_exists(self, tmp_path):
        """Verify finds project file when it exists."""
        project_file = tmp_path / PROJECT_FILE
        project_file.write_text('{"version": 1}')

        result = find_project_file(tmp_path)

        assert result == project_file
        assert result.exists()

    def test_find_project_file_not_exists(self, tmp_path):
        """Verify returns None when project file doesn't exist."""
        result = find_project_file(tmp_path)

        assert result is None

    def test_find_project_file_nonexistent_directory(self, tmp_path):
        """Verify returns None for nonexistent directory."""
        nonexistent = tmp_path / "does_not_exist"

        result = find_project_file(nonexistent)

        assert result is None

    def test_find_project_file_none_uses_cwd(self, tmp_path):
        """Verify None argument uses current working directory."""
        project_file = tmp_path / PROJECT_FILE
        project_file.write_text('{"version": 1}')

        with patch('cryoem_annotation.core.project.Path') as mock_path:
            # Mock Path.cwd() to return tmp_path
            mock_path.cwd.return_value = tmp_path
            mock_path.return_value = tmp_path

            # The function should check cwd when start_path is None
            result = find_project_file(None)
            mock_path.cwd.assert_called_once()


class TestLoadProject:
    """Tests for load_project function."""

    def test_load_project_basic(self, tmp_path):
        """Verify loading a basic project file."""
        project_file = tmp_path / PROJECT_FILE
        data = {
            "version": 1,
            "micrographs": "/path/to/micrographs",
            "checkpoint": "/path/to/checkpoint.pth",
            "created": "2025-01-01T00:00:00",
            "updated": "2025-01-01T12:00:00"
        }
        project_file.write_text(json.dumps(data))

        result = load_project(project_file)

        assert result['version'] == 1
        assert result['micrographs'] == Path("/path/to/micrographs")
        assert result['checkpoint'] == Path("/path/to/checkpoint.pth")
        assert result['created'] == "2025-01-01T00:00:00"
        assert result['updated'] == "2025-01-01T12:00:00"

    def test_load_project_none_checkpoint(self, tmp_path):
        """Verify loading project with null checkpoint."""
        project_file = tmp_path / PROJECT_FILE
        data = {
            "version": 1,
            "micrographs": "/path/to/micrographs",
            "checkpoint": None,
        }
        project_file.write_text(json.dumps(data))

        result = load_project(project_file)

        assert result['micrographs'] == Path("/path/to/micrographs")
        assert result['checkpoint'] is None

    def test_load_project_file_not_found(self, tmp_path):
        """Verify FileNotFoundError for missing file."""
        nonexistent = tmp_path / "nonexistent.json"

        with pytest.raises(FileNotFoundError):
            load_project(nonexistent)

    def test_load_project_invalid_json(self, tmp_path):
        """Verify JSONDecodeError for invalid JSON."""
        project_file = tmp_path / PROJECT_FILE
        project_file.write_text("not valid json {")

        with pytest.raises(json.JSONDecodeError):
            load_project(project_file)


class TestSaveProject:
    """Tests for save_project function."""

    def test_save_project_creates_file(self, tmp_path):
        """Verify save_project creates project file."""
        results_folder = tmp_path / "results"
        micrographs = tmp_path / "micrographs"
        micrographs.mkdir()

        result = save_project(results_folder, micrographs)

        assert result.exists()
        assert result.name == PROJECT_FILE
        assert result.parent == results_folder

    def test_save_project_creates_results_folder(self, tmp_path):
        """Verify save_project creates results folder if needed."""
        results_folder = tmp_path / "new_results"
        micrographs = tmp_path / "micrographs"
        micrographs.mkdir()

        assert not results_folder.exists()

        save_project(results_folder, micrographs)

        assert results_folder.exists()

    def test_save_project_stores_absolute_paths(self, tmp_path):
        """Verify paths are stored as absolute paths."""
        results_folder = tmp_path / "results"
        micrographs = tmp_path / "micrographs"
        checkpoint = tmp_path / "model.pth"
        micrographs.mkdir()
        checkpoint.touch()

        project_file = save_project(results_folder, micrographs, checkpoint)

        with open(project_file) as f:
            data = json.load(f)

        # Verify paths are absolute
        assert Path(data['micrographs']).is_absolute()
        assert Path(data['checkpoint']).is_absolute()
        # Verify paths are resolved (no .. or . components)
        assert str(micrographs.resolve()) == data['micrographs']
        assert str(checkpoint.resolve()) == data['checkpoint']

    def test_save_project_with_checkpoint(self, tmp_path):
        """Verify checkpoint path is saved."""
        results_folder = tmp_path / "results"
        micrographs = tmp_path / "micrographs"
        checkpoint = tmp_path / "sam_vit_b.pth"
        micrographs.mkdir()
        checkpoint.touch()

        project_file = save_project(results_folder, micrographs, checkpoint)

        with open(project_file) as f:
            data = json.load(f)

        assert data['checkpoint'] is not None
        assert "sam_vit_b.pth" in data['checkpoint']

    def test_save_project_without_checkpoint(self, tmp_path):
        """Verify checkpoint can be None."""
        results_folder = tmp_path / "results"
        micrographs = tmp_path / "micrographs"
        micrographs.mkdir()

        project_file = save_project(results_folder, micrographs, checkpoint=None)

        with open(project_file) as f:
            data = json.load(f)

        assert data['checkpoint'] is None

    def test_save_project_has_timestamps(self, tmp_path):
        """Verify created and updated timestamps are set."""
        results_folder = tmp_path / "results"
        micrographs = tmp_path / "micrographs"
        micrographs.mkdir()

        before = datetime.now().isoformat()
        project_file = save_project(results_folder, micrographs)
        after = datetime.now().isoformat()

        with open(project_file) as f:
            data = json.load(f)

        # Timestamps should be between before and after
        assert 'created' in data
        assert 'updated' in data
        assert before <= data['created'] <= after
        assert before <= data['updated'] <= after

    def test_save_project_update_preserves_created(self, tmp_path):
        """Verify updating project preserves original created timestamp."""
        results_folder = tmp_path / "results"
        micrographs = tmp_path / "micrographs"
        micrographs.mkdir()

        # First save
        project_file = save_project(results_folder, micrographs)
        with open(project_file) as f:
            first_data = json.load(f)

        # Second save (update)
        new_micrographs = tmp_path / "new_micrographs"
        new_micrographs.mkdir()
        save_project(results_folder, new_micrographs)

        with open(project_file) as f:
            second_data = json.load(f)

        # Created should be preserved, updated should change
        assert first_data['created'] == second_data['created']
        assert first_data['updated'] <= second_data['updated']

    def test_save_project_has_version(self, tmp_path):
        """Verify version is set in project file."""
        results_folder = tmp_path / "results"
        micrographs = tmp_path / "micrographs"
        micrographs.mkdir()

        project_file = save_project(results_folder, micrographs)

        with open(project_file) as f:
            data = json.load(f)

        assert data['version'] == PROJECT_VERSION

    def test_save_project_file_has_trailing_newline(self, tmp_path):
        """Verify project file ends with newline."""
        results_folder = tmp_path / "results"
        micrographs = tmp_path / "micrographs"
        micrographs.mkdir()

        project_file = save_project(results_folder, micrographs)

        content = project_file.read_text()
        assert content.endswith('\n')


class TestResolvePaths:
    """Tests for resolve_paths function."""

    def test_resolve_with_explicit_results(self, tmp_path):
        """Verify explicit results path is used."""
        results = tmp_path / "results"
        results.mkdir()

        resolved = resolve_paths(results=results)

        assert resolved['results'] == results

    def test_resolve_with_explicit_micrographs(self, tmp_path):
        """Verify explicit micrographs path is used."""
        results = tmp_path / "results"
        micrographs = tmp_path / "micrographs"
        results.mkdir()
        micrographs.mkdir()

        resolved = resolve_paths(results=results, micrographs=micrographs)

        assert resolved['micrographs'] == micrographs

    def test_resolve_cli_args_override_project(self, tmp_path):
        """Verify CLI args take precedence over project file."""
        results = tmp_path / "results"
        results.mkdir()
        cli_micrographs = tmp_path / "cli_micrographs"
        cli_micrographs.mkdir()
        project_micrographs = tmp_path / "project_micrographs"
        project_micrographs.mkdir()

        # Create project file with different micrographs path
        save_project(results, project_micrographs)

        # CLI arg should override
        resolved = resolve_paths(results=results, micrographs=cli_micrographs)

        assert resolved['micrographs'] == cli_micrographs

    def test_resolve_loads_micrographs_from_project(self, tmp_path):
        """Verify micrographs path is loaded from project when not in CLI."""
        results = tmp_path / "results"
        micrographs = tmp_path / "micrographs"
        micrographs.mkdir()

        # Create project file
        save_project(results, micrographs)

        # Don't pass micrographs - should load from project
        resolved = resolve_paths(results=results)

        assert resolved['micrographs'] == micrographs.resolve()

    def test_resolve_loads_checkpoint_from_project(self, tmp_path):
        """Verify checkpoint path is loaded from project when not in CLI."""
        results = tmp_path / "results"
        micrographs = tmp_path / "micrographs"
        checkpoint = tmp_path / "model.pth"
        micrographs.mkdir()
        checkpoint.touch()

        # Create project file with checkpoint
        save_project(results, micrographs, checkpoint)

        # Don't pass checkpoint - should load from project
        resolved = resolve_paths(results=results)

        assert resolved['checkpoint'] == checkpoint.resolve()

    def test_resolve_auto_detect_from_cwd(self, tmp_path):
        """Verify auto-detection from CWD works."""
        results = tmp_path
        micrographs = tmp_path / "micrographs"
        micrographs.mkdir()

        # Create project file in tmp_path
        save_project(results, micrographs)

        # Mock CWD to be tmp_path
        import os
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            # Don't pass results - should auto-detect from CWD
            resolved = resolve_paths()
            assert resolved['results'] == Path.cwd()
            assert resolved['project_file'] is not None
        finally:
            os.chdir(original_cwd)

    def test_resolve_error_no_results_no_project(self, tmp_path):
        """Verify error when no results and no project file."""
        import click
        import os

        # Use an empty directory with no project file
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        original_cwd = os.getcwd()
        try:
            os.chdir(empty_dir)
            with pytest.raises(click.ClickException) as exc_info:
                resolve_paths()
            assert "No results folder specified" in str(exc_info.value)
        finally:
            os.chdir(original_cwd)

    def test_resolve_error_micrographs_required_but_missing(self, tmp_path):
        """Verify error when micrographs required but not found."""
        import click

        results = tmp_path / "results"
        results.mkdir()
        # No project file, no micrographs in CLI

        with pytest.raises(click.ClickException) as exc_info:
            resolve_paths(results=results, require_micrographs=True)
        assert "Micrographs path is required" in str(exc_info.value)

    def test_resolve_error_micrographs_path_not_exists(self, tmp_path):
        """Verify error when micrographs path doesn't exist."""
        import click

        results = tmp_path / "results"
        results.mkdir()
        nonexistent = tmp_path / "nonexistent"

        with pytest.raises(click.ClickException) as exc_info:
            resolve_paths(results=results, micrographs=nonexistent)
        assert "not found" in str(exc_info.value)

    def test_resolve_returns_project_file_path(self, tmp_path):
        """Verify project_file is returned in resolved dict."""
        results = tmp_path / "results"
        micrographs = tmp_path / "micrographs"
        micrographs.mkdir()

        project_file = save_project(results, micrographs)

        resolved = resolve_paths(results=results)

        assert resolved['project_file'] == project_file

    def test_resolve_checkpoint_warning_not_error(self, tmp_path):
        """Verify missing checkpoint is warning, not error."""
        results = tmp_path / "results"
        micrographs = tmp_path / "micrographs"
        checkpoint = tmp_path / "nonexistent.pth"
        micrographs.mkdir()

        # Create project with nonexistent checkpoint
        save_project(results, micrographs, checkpoint)

        # Should not raise error, just set checkpoint to None
        resolved = resolve_paths(results=results)

        assert resolved['checkpoint'] is None


class TestIntegration:
    """Integration tests for project workflow."""

    def test_full_workflow(self, tmp_path):
        """Test the complete workflow: save, find, load, resolve."""
        # Setup
        results = tmp_path / "results"
        micrographs = tmp_path / "data" / "grids"
        checkpoint = tmp_path / "models" / "sam.pth"
        micrographs.mkdir(parents=True)
        checkpoint.parent.mkdir(parents=True)
        checkpoint.touch()

        # 1. Save project (simulating annotate command)
        project_file = save_project(results, micrographs, checkpoint)
        assert project_file.exists()

        # 2. Find project (simulating label command in results dir)
        found = find_project_file(results)
        assert found == project_file

        # 3. Load project
        data = load_project(project_file)
        assert data['micrographs'] == micrographs.resolve()
        assert data['checkpoint'] == checkpoint.resolve()

        # 4. Resolve paths (simulating label command with just --results)
        resolved = resolve_paths(results=results, require_micrographs=True)
        assert resolved['results'] == results
        assert resolved['micrographs'] == micrographs.resolve()
        assert resolved['checkpoint'] == checkpoint.resolve()
        assert resolved['project_file'] == project_file
