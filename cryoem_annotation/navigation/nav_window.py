"""Tkinter navigation window for file selection."""

import tkinter as tk
from tkinter import ttk
from pathlib import Path
from typing import List, Callable, Set, Optional, Union, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from cryoem_annotation.core.grid_dataset import MicrographItem


class NavigationWindow:
    """Tkinter navigation panel for file selection.

    Provides a file list with:
    - Checkmarks for completed files
    - Highlighting for current file
    - Prev/Next buttons
    - Progress counter
    - Click-to-jump functionality
    - Grid-grouped display for multi-grid datasets
    """

    def __init__(
        self,
        files: Union[List[Path], List["MicrographItem"]],
        on_navigate: Callable[[str, Optional[int]], None],
        title: str = "Navigation",
        is_multi_grid: bool = False,
    ):
        """Initialize the navigation window.

        Args:
            files: List of file paths or MicrographItem instances to display
            on_navigate: Callback function(action, target_index)
                         action is 'next', 'prev', 'goto', or 'quit'
            title: Window title
            is_multi_grid: If True, display files grouped by grid with headers
        """
        self.files = files
        self.on_navigate = on_navigate
        self.current_index = 0
        self.completed: Set[int] = set()
        self.in_progress: Set[int] = set()
        self.is_multi_grid = is_multi_grid

        # For multi-grid mode: map listbox row index to file index
        # (some rows are headers, not files)
        self._row_to_file_index: Dict[int, int] = {}
        self._file_index_to_row: Dict[int, int] = {}
        self._header_rows: Set[int] = set()

        # Track per-grid completion for multi-grid mode
        self._grid_file_indices: Dict[Optional[str], List[int]] = {}
        if is_multi_grid:
            self._build_grid_indices()

        # Create Tk root if none exists, otherwise use Toplevel
        try:
            # Check if a Tk root already exists
            existing_root = tk._default_root
            if existing_root is None:
                self.root = tk.Tk()
            else:
                self.root = tk.Toplevel()
        except Exception:
            self.root = tk.Tk()
        self.root.title(title)
        self.root.geometry("320x450" if is_multi_grid else "280x400")
        self.root.resizable(True, True)

        # Prevent closing window from quitting app
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._setup_ui()
        self._refresh_list()

        # Ensure window is visible
        self.root.deiconify()
        self.root.lift()
        self.root.update()

    def _build_grid_indices(self) -> None:
        """Build mapping of grid names to file indices for progress tracking."""
        for i, item in enumerate(self.files):
            grid_name = getattr(item, 'grid_name', None)
            if grid_name not in self._grid_file_indices:
                self._grid_file_indices[grid_name] = []
            self._grid_file_indices[grid_name].append(i)

    def _get_display_name(self, item: Union[Path, "MicrographItem"]) -> str:
        """Get display name for a file item."""
        if hasattr(item, 'display_name'):
            # MicrographItem
            return item.display_name
        elif hasattr(item, 'name'):
            # Path
            return item.name
        else:
            return str(item)

    def _get_micrograph_name(self, item: Union[Path, "MicrographItem"]) -> str:
        """Get micrograph name (without grid prefix) for display."""
        if hasattr(item, 'micrograph_name'):
            return item.micrograph_name
        elif hasattr(item, 'stem'):
            return item.stem
        else:
            return str(item)

    def _setup_ui(self) -> None:
        """Set up the UI components."""
        # Main frame with padding
        main_frame = ttk.Frame(self.root, padding="5")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title label
        title_text = "Micrographs (Multi-Grid)" if self.is_multi_grid else "Micrographs"
        title_label = ttk.Label(
            main_frame,
            text=title_text,
            font=("TkDefaultFont", 11, "bold")
        )
        title_label.pack(anchor=tk.W, pady=(0, 5))

        # Listbox with scrollbar
        list_frame = ttk.Frame(main_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)

        self.scrollbar = ttk.Scrollbar(list_frame)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.listbox = tk.Listbox(
            list_frame,
            yscrollcommand=self.scrollbar.set,
            font=("TkFixedFont", 10),
            selectmode=tk.SINGLE,
            activestyle='none'
        )
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar.config(command=self.listbox.yview)

        # Bind click event
        self.listbox.bind('<<ListboxSelect>>', self._on_listbox_select)

        # Button frame
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=(10, 5))

        self.prev_btn = ttk.Button(
            btn_frame,
            text="< Prev",
            command=self.go_prev,
            width=10
        )
        self.prev_btn.pack(side=tk.LEFT, padx=(0, 5))

        self.next_btn = ttk.Button(
            btn_frame,
            text="Next >",
            command=self.go_next,
            width=10
        )
        self.next_btn.pack(side=tk.LEFT, padx=(0, 5))

        self.quit_btn = ttk.Button(
            btn_frame,
            text="Finish",
            command=self._on_quit,
            width=8
        )
        self.quit_btn.pack(side=tk.RIGHT)

        # Progress label
        self.progress_label = ttk.Label(
            main_frame,
            text="0/0 completed",
            font=("TkDefaultFont", 9)
        )
        self.progress_label.pack(anchor=tk.W, pady=(5, 0))

    def _refresh_list(self) -> None:
        """Refresh the listbox display."""
        self.listbox.delete(0, tk.END)
        self._row_to_file_index.clear()
        self._file_index_to_row.clear()
        self._header_rows.clear()

        if self.is_multi_grid:
            self._refresh_list_multi_grid()
        else:
            self._refresh_list_single()

        # Update progress
        self._update_progress()

        # Update button states
        self.prev_btn.config(state=tk.NORMAL if self.current_index > 0 else tk.DISABLED)
        self.next_btn.config(
            state=tk.NORMAL if self.current_index < len(self.files) - 1 else tk.DISABLED
        )

    def _refresh_list_single(self) -> None:
        """Refresh list for single-folder mode (original behavior)."""
        for i, file_item in enumerate(self.files):
            # Build display text with status indicator
            if i in self.completed:
                prefix = "[x]"
            elif i in self.in_progress:
                prefix = "[~]"
            elif i == self.current_index:
                prefix = " > "
            else:
                prefix = "[ ]"

            display_name = self._get_display_name(file_item)
            display_text = f"{prefix} {display_name}"
            row = self.listbox.size()
            self.listbox.insert(tk.END, display_text)

            # Track mapping (1:1 for single mode)
            self._row_to_file_index[row] = i
            self._file_index_to_row[i] = row

            # Color coding
            if i == self.current_index:
                self.listbox.itemconfig(row, fg="#0066cc", selectbackground="#0066cc")
            elif i in self.completed:
                self.listbox.itemconfig(row, fg="#228b22")  # Forest green
            elif i in self.in_progress:
                self.listbox.itemconfig(row, fg="#cc6600")  # Orange

        # Ensure current item is visible
        if 0 <= self.current_index < len(self.files):
            current_row = self._file_index_to_row.get(self.current_index, 0)
            self.listbox.see(current_row)
            self.listbox.selection_clear(0, tk.END)
            self.listbox.selection_set(current_row)

    def _refresh_list_multi_grid(self) -> None:
        """Refresh list for multi-grid mode with grouped display."""
        # Group files by grid
        current_grid: Optional[str] = None
        file_index = 0

        for item in self.files:
            grid_name = getattr(item, 'grid_name', None)

            # Insert grid header if new grid
            if grid_name != current_grid:
                current_grid = grid_name
                # Count files in this grid
                grid_count = len(self._grid_file_indices.get(grid_name, []))
                grid_completed = sum(
                    1 for idx in self._grid_file_indices.get(grid_name, [])
                    if idx in self.completed
                )

                header_text = f"── {grid_name or 'Files'} ({grid_completed}/{grid_count}) ──"
                row = self.listbox.size()
                self.listbox.insert(tk.END, header_text)
                self._header_rows.add(row)
                # Style header: dark gray, bold-like
                self.listbox.itemconfig(row, fg="#555555", selectbackground="#cccccc")

            # Insert file entry
            if file_index in self.completed:
                prefix = "[x]"
            elif file_index in self.in_progress:
                prefix = "[~]"
            elif file_index == self.current_index:
                prefix = " > "
            else:
                prefix = "[ ]"

            # Use just micrograph name (grid already shown in header)
            micrograph_name = self._get_micrograph_name(item)
            display_text = f"  {prefix} {micrograph_name}"
            row = self.listbox.size()
            self.listbox.insert(tk.END, display_text)

            # Track mapping
            self._row_to_file_index[row] = file_index
            self._file_index_to_row[file_index] = row

            # Color coding
            if file_index == self.current_index:
                self.listbox.itemconfig(row, fg="#0066cc", selectbackground="#0066cc")
            elif file_index in self.completed:
                self.listbox.itemconfig(row, fg="#228b22")  # Forest green
            elif file_index in self.in_progress:
                self.listbox.itemconfig(row, fg="#cc6600")  # Orange

            file_index += 1

        # Ensure current item is visible
        if 0 <= self.current_index < len(self.files):
            current_row = self._file_index_to_row.get(self.current_index, 0)
            self.listbox.see(current_row)
            self.listbox.selection_clear(0, tk.END)
            self.listbox.selection_set(current_row)

    def _update_progress(self) -> None:
        """Update the progress label."""
        completed = len(self.completed)
        total = len(self.files)

        if self.is_multi_grid and len(self._grid_file_indices) > 1:
            # Show per-grid progress
            parts = []
            for grid_name in sorted(self._grid_file_indices.keys(), key=lambda x: x or ""):
                indices = self._grid_file_indices[grid_name]
                grid_completed = sum(1 for idx in indices if idx in self.completed)
                grid_total = len(indices)
                display_name = grid_name if grid_name else "Files"
                parts.append(f"{grid_completed}/{grid_total} {display_name}")

            parts.append(f"{completed}/{total} total")
            progress_text = " | ".join(parts)
        else:
            progress_text = f"{completed}/{total} completed"

        self.progress_label.config(text=progress_text)

    def _on_listbox_select(self, event) -> None:
        """Handle listbox selection."""
        selection = self.listbox.curselection()
        if selection:
            row = selection[0]

            # Skip header rows in multi-grid mode
            if row in self._header_rows:
                # Restore previous selection
                if self.current_index in self._file_index_to_row:
                    prev_row = self._file_index_to_row[self.current_index]
                    self.listbox.selection_clear(0, tk.END)
                    self.listbox.selection_set(prev_row)
                return

            # Map row to file index
            file_index = self._row_to_file_index.get(row)
            if file_index is not None and file_index != self.current_index:
                self.on_navigate('goto', file_index)

    def _on_close(self) -> None:
        """Handle window close button."""
        self.on_navigate('quit', None)

    def _on_quit(self) -> None:
        """Handle quit button."""
        self.on_navigate('quit', None)

    def go_prev(self) -> None:
        """Navigate to previous file."""
        if self.current_index > 0:
            self.on_navigate('prev', None)

    def go_next(self) -> None:
        """Navigate to next file."""
        if self.current_index < len(self.files) - 1:
            self.on_navigate('next', None)

    def set_current(self, index: int) -> None:
        """Set the current file index.

        Args:
            index: New current index
        """
        if 0 <= index < len(self.files):
            self.current_index = index
            self._refresh_list()

    def mark_completed(self, index: int) -> None:
        """Mark a file as completed.

        Args:
            index: Index of completed file
        """
        if 0 <= index < len(self.files):
            self.completed.add(index)
            self.in_progress.discard(index)
            self._refresh_list()

    def mark_in_progress(self, index: int) -> None:
        """Mark a file as in-progress (partially labeled).

        Args:
            index: Index of in-progress file
        """
        if 0 <= index < len(self.files):
            self.in_progress.add(index)
            # Remove from completed if it was there
            self.completed.discard(index)
            self._refresh_list()

    def destroy(self) -> None:
        """Destroy the navigation window."""
        try:
            self.root.destroy()
        except tk.TclError:
            pass  # Window already destroyed
