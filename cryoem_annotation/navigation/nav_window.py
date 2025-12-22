"""Tkinter navigation window for file selection."""

import tkinter as tk
from tkinter import ttk
from pathlib import Path
from typing import List, Callable, Set, Optional


class NavigationWindow:
    """Tkinter navigation panel for file selection.

    Provides a file list with:
    - Checkmarks for completed files
    - Highlighting for current file
    - Prev/Next buttons
    - Progress counter
    - Click-to-jump functionality
    """

    def __init__(
        self,
        files: List[Path],
        on_navigate: Callable[[str, Optional[int]], None],
        title: str = "Navigation"
    ):
        """Initialize the navigation window.

        Args:
            files: List of file paths to display
            on_navigate: Callback function(action, target_index)
                         action is 'next', 'prev', 'goto', or 'quit'
            title: Window title
        """
        self.files = files
        self.on_navigate = on_navigate
        self.current_index = 0
        self.completed: Set[int] = set()

        # Create window as Toplevel (uses existing Tk instance from matplotlib)
        self.root = tk.Toplevel()
        self.root.title(title)
        self.root.geometry("280x400")
        self.root.resizable(True, True)

        # Prevent closing window from quitting app
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._setup_ui()
        self._refresh_list()

    def _setup_ui(self) -> None:
        """Set up the UI components."""
        # Main frame with padding
        main_frame = ttk.Frame(self.root, padding="5")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title label
        title_label = ttk.Label(
            main_frame,
            text="Micrographs",
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

        for i, file_path in enumerate(self.files):
            # Build display text with status indicator
            if i in self.completed:
                prefix = "[x]"
            elif i == self.current_index:
                prefix = " > "
            else:
                prefix = "[ ]"

            display_text = f"{prefix} {file_path.name}"
            self.listbox.insert(tk.END, display_text)

            # Color coding
            if i == self.current_index:
                self.listbox.itemconfig(i, fg="#0066cc", selectbackground="#0066cc")
            elif i in self.completed:
                self.listbox.itemconfig(i, fg="#228b22")  # Forest green

        # Ensure current item is visible
        if 0 <= self.current_index < len(self.files):
            self.listbox.see(self.current_index)
            self.listbox.selection_clear(0, tk.END)
            self.listbox.selection_set(self.current_index)

        # Update progress
        self._update_progress()

        # Update button states
        self.prev_btn.config(state=tk.NORMAL if self.current_index > 0 else tk.DISABLED)
        self.next_btn.config(
            state=tk.NORMAL if self.current_index < len(self.files) - 1 else tk.DISABLED
        )

    def _update_progress(self) -> None:
        """Update the progress label."""
        completed = len(self.completed)
        total = len(self.files)
        self.progress_label.config(text=f"{completed}/{total} completed")

    def _on_listbox_select(self, event) -> None:
        """Handle listbox selection."""
        selection = self.listbox.curselection()
        if selection:
            index = selection[0]
            if index != self.current_index:
                self.on_navigate('goto', index)

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
            self._refresh_list()

    def destroy(self) -> None:
        """Destroy the navigation window."""
        try:
            self.root.destroy()
        except tk.TclError:
            pass  # Window already destroyed
