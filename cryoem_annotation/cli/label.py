"""CLI for labeling tool."""

import click
from pathlib import Path
from typing import Optional

from cryoem_annotation.config import load_config


@click.command()
@click.option('--results', '-r', type=click.Path(exists=True, path_type=Path),
              help='Path to annotation results folder (auto-detected if in project directory)')
@click.option('--micrographs', '-m', type=click.Path(exists=True, path_type=Path),
              help='Path to micrograph folder (auto-loaded from project file if not specified)')
@click.option('--config', type=click.Path(exists=True, path_type=Path),
              help='Path to config file')
def main(results: Optional[Path], micrographs: Optional[Path], config: Optional[Path]):
    """
    Interactive labeling tool for assigning labels to segmentations.

    This tool loads previously created segmentations and allows you to assign
    categorical labels (mature, immature, etc.) by clicking on the objects.

    Supports both single-folder and multi-grid input structures:
    - Single folder: micrographs/*.mrc
    - Multi-grid: micrographs/{Grid1,Grid2,...}/*.mrc

    Default keyboard shortcuts: 1=mature, 2=immature, 3=indeterminate, 4=other, 5=empty

    If run from a directory containing .cryoem-project.json (created by cryoem-annotate),
    the --results and --micrographs paths are auto-detected.
    """
    # Load config
    cfg = load_config(config)

    # Lazy import to speed up CLI startup
    from cryoem_annotation.core.grid_dataset import GridDataset
    from cryoem_annotation.core.project import resolve_paths
    from cryoem_annotation.labeling.labeler import label_segmentations
    from cryoem_annotation.labeling.categories import LabelCategories

    # Resolve paths from CLI args and/or project file
    # CLI args take precedence; micrographs can also come from config
    micrograph_from_config = Path(cfg.get('micrograph_folder')) if cfg.get('micrograph_folder') else None
    resolved = resolve_paths(
        results=results,
        micrographs=micrographs or micrograph_from_config,
        require_micrographs=True,
    )

    results_folder = resolved['results']
    micrograph_folder = resolved['micrographs']

    # Report if using auto-detected paths
    if resolved['project_file']:
        click.echo(f"[OK] Using project file: {resolved['project_file']}")

    # Create GridDataset - auto-detects multi-grid vs single-folder
    dataset = GridDataset(micrograph_folder)

    # Print detected structure
    if dataset.is_multi_grid:
        click.echo(f"Detected multi-grid structure ({len(dataset.grid_names)} grids)")
        for grid_name in dataset.grid_names:
            count = dataset.get_micrograph_count(grid_name)
            click.echo(f"  {grid_name}: {count} files")
    else:
        click.echo(f"Single folder mode ({dataset.total_micrographs} files)")

    # Load label categories from config (or use defaults)
    categories_config = cfg.get('labeling.categories')
    categories = LabelCategories(categories_config) if categories_config else None

    # Run labeling
    label_segmentations(
        results_folder=results_folder,
        dataset=dataset,
        categories=categories,
    )


if __name__ == '__main__':
    main()

