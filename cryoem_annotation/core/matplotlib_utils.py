"""Matplotlib backend initialization and configuration utilities.

This module provides shared matplotlib backend setup for interactive
visualization tools in the cryoem-annotation package.
"""

import matplotlib

from cryoem_annotation.core.logging_utils import get_logger


# Backend options to try in order of preference
_BACKEND_OPTIONS = ['Qt5Agg', 'MacOSX', 'TkAgg']

# Track which backend was selected
_selected_backend = None


def setup_interactive_backend() -> str | None:
    """Set up an interactive matplotlib backend.

    Attempts to configure matplotlib with an interactive backend,
    trying each option in order of preference: Qt5Agg, MacOSX, TkAgg.
    Each backend is tested by creating and closing a figure.

    Returns:
        The name of the selected backend, or None if using default.
    """
    global _selected_backend
    logger = get_logger()

    for backend_name in _BACKEND_OPTIONS:
        try:
            matplotlib.use(backend_name)
            import matplotlib.pyplot as plt
            test_fig = plt.figure()
            plt.close(test_fig)
            _selected_backend = backend_name
            logger.success(f"Matplotlib backend: {backend_name}")
            return backend_name
        except Exception:
            continue

    logger.warning("Could not set interactive backend, using default")
    _selected_backend = None
    return None


def get_active_backend() -> str:
    """Get the name of the currently active matplotlib backend.

    Returns:
        The name of the active backend.
    """
    return matplotlib.get_backend()


# Set up interactive backend on module import
setup_interactive_backend()

# Import and configure pyplot in interactive mode
import matplotlib.pyplot as plt
plt.ion()


def get_screen_aware_figsize(
    target_size: float = 12.0,
    screen_fraction: float = 0.90,
    fallback_size: float = 8.0,
    dpi: float = 100.0
) -> tuple[float, float]:
    """Calculate figure size that fits within screen bounds.

    This prevents figure window expansion issues in TurboVNC and other
    environments where matplotlib remembers requested sizes even after
    the window manager constrains them.

    Args:
        target_size: Desired figure size in inches (default 12.0)
        screen_fraction: Fraction of screen to use (default 0.90)
        fallback_size: Size if screen detection fails (default 8.0)
        dpi: Matplotlib DPI setting (default 100.0)

    Returns:
        Tuple of (width, height) in inches, square aspect ratio.
    """
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        root.destroy()

        available_pixels = min(screen_width, screen_height) * screen_fraction
        max_size_inches = available_pixels / dpi
        final_size = min(target_size, max_size_inches)
        return (final_size, final_size)
    except Exception:
        return (fallback_size, fallback_size)
