"""CLI for labeling tool."""

import click
from pathlib import Path
from typing import Optional

from cryoem_annotation.labeling.labeler import label_segmentations
from cryoem_annotation.config import load_config


@click.command()
@click.option('--results', '-r', type=click.Path(exists=True, path_type=Path),
              required=True, help='Path to annotation results folder')
@click.option('--micrographs', '-m', type=click.Path(exists=True, path_type=Path),
              help='Path to micrograph folder')
@click.option('--config', type=click.Path(exists=True, path_type=Path),
              help='Path to config file')
def main(results: Path, micrographs: Optional[Path], config: Optional[Path]):
    """
    Interactive labeling tool for assigning labels to segmentations.
    
    This tool loads previously created segmentations and allows you to assign
    labels (0-9) to them by clicking on the objects.
    """
    # Load config
    cfg = load_config(config)
    
    # Get micrograph folder from CLI or config
    micrograph_folder = micrographs or Path(cfg.get('micrograph_folder'))
    
    if micrograph_folder is None:
        click.echo("Error: --micrographs is required or set in config file", err=True)
        return
    
    # Run labeling
    label_segmentations(
        results_folder=results,
        micrograph_folder=micrograph_folder,
    )


if __name__ == '__main__':
    main()

