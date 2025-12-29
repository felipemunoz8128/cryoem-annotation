"""Grid-aware data structures for multi-grid dataset support."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from cryoem_annotation.core.image_loader import IMAGE_EXTENSIONS, get_image_files


@dataclass(frozen=True)
class MicrographItem:
    """Immutable data class representing a single micrograph with grid context.

    Attributes:
        file_path: Absolute path to the micrograph file.
        grid_name: Grid identifier (e.g., "Grid1") or None for single-folder mode.
        micrograph_name: Filename stem (e.g., "micrograph_001_DW").
    """

    file_path: Path
    grid_name: Optional[str]
    micrograph_name: str

    @property
    def display_name(self) -> str:
        """Return display name with grid context if available.

        Returns:
            "{grid_name}/{micrograph_name}" if grid_name is set,
            "{micrograph_name}" if grid_name is None.
        """
        if self.grid_name is not None:
            return f"{self.grid_name}/{self.micrograph_name}"
        return self.micrograph_name

    def __str__(self) -> str:
        """Return display name for easy printing."""
        return self.display_name


class GridDataset:
    """Grid-aware dataset for managing micrographs from single or multi-grid structures.

    Detects whether the root path contains a multi-grid structure (subdirectories
    with image files) or a single-folder structure, and provides unified access
    to micrographs with grid context.

    Attributes:
        root_path: The root directory containing micrographs.
        is_multi_grid: True if multiple grid subdirectories were detected.
    """

    def __init__(self, root_path: Path) -> None:
        """Initialize GridDataset by scanning the directory structure.

        Args:
            root_path: Path to the directory containing micrographs.
                       Can be a flat folder or contain grid subdirectories.
        """
        self.root_path = root_path
        self._grids: Dict[Optional[str], List[MicrographItem]] = {}
        self._scan_directory()

    def _scan_directory(self) -> None:
        """Scan directory to detect multi-grid vs single-folder structure.

        Detection logic:
        - If root contains subdirectories with image files: multi-grid mode
        - Otherwise: single-folder mode (grid_name=None)
        """
        if not self.root_path.exists():
            return

        # Check for subdirectories with image files (multi-grid structure)
        subdirs_with_images: List[Path] = []
        for item in sorted(self.root_path.iterdir()):
            # Skip hidden directories
            if item.name.startswith('.'):
                continue
            if item.is_dir():
                # Check if this subdirectory contains image files
                images = get_image_files(item)
                if images:
                    subdirs_with_images.append(item)

        if subdirs_with_images:
            # Multi-grid mode: each subdirectory is a grid
            for subdir in subdirs_with_images:
                grid_name = subdir.name
                images = get_image_files(subdir)
                self._grids[grid_name] = [
                    MicrographItem(
                        file_path=img,
                        grid_name=grid_name,
                        micrograph_name=img.stem,
                    )
                    for img in images
                ]
        else:
            # Single-folder mode: root itself contains images
            images = get_image_files(self.root_path)
            if images:
                self._grids[None] = [
                    MicrographItem(
                        file_path=img,
                        grid_name=None,
                        micrograph_name=img.stem,
                    )
                    for img in images
                ]

    @property
    def is_multi_grid(self) -> bool:
        """Return True if multiple grids were detected."""
        # Multi-grid if we have any grid with a non-None name
        return any(name is not None for name in self._grids.keys())

    @property
    def grid_names(self) -> List[str]:
        """Return sorted list of grid names (empty list for single-folder mode)."""
        return sorted(name for name in self._grids.keys() if name is not None)

    @property
    def total_micrographs(self) -> int:
        """Return total count of micrographs across all grids."""
        return sum(len(items) for items in self._grids.values())

    def get_micrographs(self, grid_name: Optional[str] = None) -> List[MicrographItem]:
        """Get micrographs, optionally filtered by grid.

        Args:
            grid_name: If provided, return only micrographs from that grid.
                      If None, return all micrographs (sorted by grid then name).

        Returns:
            List of MicrographItem instances.
        """
        if grid_name is not None:
            return list(self._grids.get(grid_name, []))

        # Return all micrographs, sorted by grid name then micrograph name
        all_items: List[MicrographItem] = []
        # Sort keys: None comes first (single-folder), then alphabetical grid names
        sorted_keys = sorted(
            self._grids.keys(),
            key=lambda k: ("" if k is None else k),
        )
        for key in sorted_keys:
            all_items.extend(self._grids[key])
        return all_items

    def get_micrograph_count(self, grid_name: str) -> int:
        """Get count of micrographs for a specific grid.

        Args:
            grid_name: The grid name to query.

        Returns:
            Number of micrographs in the specified grid.
        """
        return len(self._grids.get(grid_name, []))
