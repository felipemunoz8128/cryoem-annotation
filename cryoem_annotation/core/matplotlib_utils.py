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
