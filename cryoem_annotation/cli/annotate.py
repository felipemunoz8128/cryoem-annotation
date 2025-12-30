"""CLI for annotation tool."""

import click
from pathlib import Path
from typing import Optional

from cryoem_annotation.config import load_config


@click.command()
@click.option('--micrographs', '-m', type=click.Path(exists=True, path_type=Path),
              help='Path to micrograph folder (single folder or multi-grid root)')
@click.option('--checkpoint', '-c', type=click.Path(exists=True, path_type=Path),
              help='Path to SAM checkpoint file')
@click.option('--model-type', type=click.Choice(['vit_b', 'vit_l', 'vit_h']),
              default='vit_b', help='SAM model type')
@click.option('--output', '-o', type=click.Path(path_type=Path),
              default='annotation_results', help='Output folder')
@click.option('--config', type=click.Path(exists=True, path_type=Path),
              help='Path to config file')
@click.option('--device', type=click.Choice(['cuda', 'cpu', 'auto']),
              default='auto', help='Device to use')
def main(micrographs: Optional[Path], checkpoint: Optional[Path],
         model_type: str, output: Path, config: Optional[Path], device: str):
    """
    Interactive annotation tool for cryo-EM micrographs using SAM.

    This tool allows you to click on objects in micrographs and automatically
    segment them using Segment Anything Model (SAM).

    Supports both single-folder and multi-grid input structures:
    - Single folder: micrographs/*.mrc
    - Multi-grid: micrographs/{Grid1,Grid2,...}/*.mrc
    """
    # Load config
    cfg = load_config(config)

    # Get values from CLI args or config
    micrograph_folder = micrographs or Path(cfg.get('micrograph_folder'))
    checkpoint_path = checkpoint or Path(cfg.get('sam_model.checkpoint_path'))

    if micrograph_folder is None:
        click.echo("Error: --micrographs is required or set in config file", err=True)
        return

    if checkpoint_path is None or not checkpoint_path.exists():
        click.echo(f"Error: Checkpoint not found: {checkpoint_path}", err=True)
        click.echo("Please download a SAM checkpoint and specify with --checkpoint", err=True)
        return

    # Handle device
    device_str = None if device == 'auto' else device

    # Lazy import to speed up CLI startup
    from cryoem_annotation.core.grid_dataset import GridDataset
    from cryoem_annotation.core.project import save_project
    from cryoem_annotation.annotation.annotator import annotate_micrographs

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

    # Create output folder and save project file
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    project_file = save_project(
        results_folder=output_path,
        micrographs=micrograph_folder,
        checkpoint=checkpoint_path,
    )
    click.echo(f"[OK] Project file created: {project_file}")

    # Run annotation
    annotate_micrographs(
        dataset=dataset,
        checkpoint_path=checkpoint_path,
        output_folder=output,
        model_type=model_type,
        device=device_str,
    )


if __name__ == '__main__':
    main()
