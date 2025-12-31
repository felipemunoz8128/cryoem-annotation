"""Project file management for cryo-EM annotation workflow.

This module handles the .cryoem-project.json file that stores project configuration
in the results folder, allowing users to avoid repeating paths across commands.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any


PROJECT_FILE = ".cryoem-project.json"
PROJECT_VERSION = 1


def find_project_file(start_path: Optional[Path] = None) -> Optional[Path]:
    """Find .cryoem-project.json in start_path or current directory.

    Args:
        start_path: Directory to search in. If None, uses current working directory.

    Returns:
        Path to project file if found, None otherwise.
    """
    search_path = Path(start_path) if start_path else Path.cwd()

    if not search_path.exists():
        return None

    project_file = search_path / PROJECT_FILE
    if project_file.exists():
        return project_file

    return None


def load_project(project_file: Path) -> Dict[str, Any]:
    """Load project config from file.

    Args:
        project_file: Path to .cryoem-project.json file.

    Returns:
        Dictionary with project configuration.

    Raises:
        FileNotFoundError: If project file doesn't exist.
        json.JSONDecodeError: If project file is invalid JSON.
    """
    with open(project_file, 'r') as f:
        data = json.load(f)

    # Convert path strings back to Path objects
    if 'micrographs' in data and data['micrographs']:
        data['micrographs'] = Path(data['micrographs'])
    if 'checkpoint' in data and data['checkpoint']:
        data['checkpoint'] = Path(data['checkpoint'])

    return data


def save_project(
    results_folder: Path,
    micrographs: Path,
    checkpoint: Optional[Path] = None
) -> Path:
    """Create/update .cryoem-project.json in results folder.

    Args:
        results_folder: Path to the results/output folder.
        micrographs: Path to the micrographs folder.
        checkpoint: Optional path to SAM checkpoint file.

    Returns:
        Path to the created/updated project file.
    """
    project_file = Path(results_folder) / PROJECT_FILE

    # Load existing data if file exists (to preserve created timestamp)
    existing_data = {}
    if project_file.exists():
        try:
            with open(project_file, 'r') as f:
                existing_data = json.load(f)
        except Exception:
            pass  # Start fresh if file is corrupted

    now = datetime.now().isoformat()

    data = {
        'version': PROJECT_VERSION,
        'micrographs': str(Path(micrographs).resolve()),
        'checkpoint': str(Path(checkpoint).resolve()) if checkpoint else None,
        'created': existing_data.get('created', now),
        'updated': now,
    }

    # Ensure results folder exists
    Path(results_folder).mkdir(parents=True, exist_ok=True)

    with open(project_file, 'w') as f:
        json.dump(data, f, indent=2)
        f.write('\n')  # Trailing newline

    return project_file


def resolve_paths(
    results: Optional[Path] = None,
    micrographs: Optional[Path] = None,
    checkpoint: Optional[Path] = None,
    require_micrographs: bool = False,
) -> Dict[str, Any]:
    """Resolve paths from CLI args + project file. CLI args take precedence.

    Auto-detection logic:
    - For results: CLI provided -> use it; else check CWD for project file -> use CWD
    - For micrographs: CLI provided -> use it; else load from project file
    - For checkpoint: CLI provided -> use it; else load from project file

    Args:
        results: Results folder path from CLI (may be None).
        micrographs: Micrographs folder path from CLI (may be None).
        checkpoint: Checkpoint path from CLI (may be None).
        require_micrographs: If True, raise error if micrographs can't be resolved.

    Returns:
        Dictionary with resolved paths:
        - 'results': Path to results folder
        - 'micrographs': Path to micrographs folder (may be None)
        - 'checkpoint': Path to checkpoint file (may be None)
        - 'project_file': Path to project file if found (may be None)

    Raises:
        click.ClickException: If required paths can't be resolved.
    """
    import click

    resolved = {
        'results': None,
        'micrographs': micrographs,
        'checkpoint': checkpoint,
        'project_file': None,
    }

    # Resolve results folder
    if results:
        resolved['results'] = Path(results)
    else:
        # Check CWD for project file
        project_file = find_project_file(Path.cwd())
        if project_file:
            resolved['results'] = Path.cwd()
            resolved['project_file'] = project_file
        else:
            raise click.ClickException(
                "No results folder specified and no project found in current directory.\n"
                "Either:\n"
                "  1. Provide --results path, or\n"
                "  2. Run from a directory containing .cryoem-project.json"
            )

    # Try to load project file from results folder
    if resolved['project_file'] is None:
        project_file = find_project_file(resolved['results'])
        if project_file:
            resolved['project_file'] = project_file

    # Load project data and resolve remaining paths
    if resolved['project_file']:
        try:
            project_data = load_project(resolved['project_file'])

            # Micrographs: CLI takes precedence, then project file
            if resolved['micrographs'] is None and project_data.get('micrographs'):
                resolved['micrographs'] = project_data['micrographs']

            # Checkpoint: CLI takes precedence, then project file
            if resolved['checkpoint'] is None and project_data.get('checkpoint'):
                resolved['checkpoint'] = project_data['checkpoint']

        except Exception as e:
            # Warn but don't fail - project file may be corrupted
            click.echo(f"[WARNING] Could not load project file: {e}", err=True)

    # Validate required paths
    if require_micrographs and resolved['micrographs'] is None:
        raise click.ClickException(
            "Micrographs path is required but not found.\n"
            "Either:\n"
            "  1. Provide --micrographs path, or\n"
            "  2. Ensure the project file contains the micrographs path"
        )

    # Validate that resolved paths exist
    if resolved['micrographs'] and not Path(resolved['micrographs']).exists():
        raise click.ClickException(
            f"Micrographs folder not found: {resolved['micrographs']}"
        )

    if resolved['checkpoint'] and not Path(resolved['checkpoint']).exists():
        # Just warn for checkpoint - it's not always needed
        click.echo(f"[WARNING] Checkpoint file not found: {resolved['checkpoint']}", err=True)
        resolved['checkpoint'] = None

    return resolved


def get_completion_state(
    project_file: Path,
    workflow: str,
    micrograph_key: str
) -> Optional[str]:
    """Get the completion state for a micrograph in a workflow.

    Args:
        project_file: Path to .cryoem-project.json file.
        workflow: Workflow name ("annotation" or "labeling").
        micrograph_key: Relative path key like "grid1/micro1" or just "micro1".

    Returns:
        Completion state ("completed", "in_progress") or None if not set.
    """
    if not project_file.exists():
        return None

    try:
        with open(project_file, 'r') as f:
            data = json.load(f)
    except Exception:
        return None

    completion_state = data.get('completion_state', {})
    workflow_state = completion_state.get(workflow, {})
    return workflow_state.get(micrograph_key)


def set_completion_state(
    project_file: Path,
    workflow: str,
    micrograph_key: str,
    status: str
) -> None:
    """Update the completion state for a micrograph in a workflow.

    Args:
        project_file: Path to .cryoem-project.json file.
        workflow: Workflow name ("annotation" or "labeling").
        micrograph_key: Relative path key like "grid1/micro1" or just "micro1".
        status: Completion status ("completed", "in_progress").

    Note:
        Creates the completion_state structure if it doesn't exist.
        Preserves all other data in the project file.
    """
    # Load existing data
    data = {}
    if project_file.exists():
        try:
            with open(project_file, 'r') as f:
                data = json.load(f)
        except Exception:
            pass  # Start fresh if file is corrupted

    # Ensure completion_state structure exists
    if 'completion_state' not in data:
        data['completion_state'] = {}
    if workflow not in data['completion_state']:
        data['completion_state'][workflow] = {}

    # Update the state
    data['completion_state'][workflow][micrograph_key] = status

    # Update timestamp
    data['updated'] = datetime.now().isoformat()

    # Write back
    with open(project_file, 'w') as f:
        json.dump(data, f, indent=2)
        f.write('\n')  # Trailing newline


def get_all_completion_states(
    project_file: Path,
    workflow: str
) -> Dict[str, str]:
    """Get all completion states for a workflow.

    Args:
        project_file: Path to .cryoem-project.json file.
        workflow: Workflow name ("annotation" or "labeling").

    Returns:
        Dictionary mapping micrograph_key to status, or empty dict if no states exist.
    """
    if not project_file.exists():
        return {}

    try:
        with open(project_file, 'r') as f:
            data = json.load(f)
    except Exception:
        return {}

    completion_state = data.get('completion_state', {})
    return completion_state.get(workflow, {})
