"""CLI for annotation tool."""

import click
from pathlib import Path
from typing import Optional

from cryoem_annotation.annotation.annotator import annotate_micrographs
from cryoem_annotation.config import load_config


@click.command()
@click.option('--micrographs', '-m', type=click.Path(exists=True, path_type=Path),
              help='Path to micrograph folder')
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
@click.option('--pixel-size', '-p', type=float, default=None,
              help='Pixel size in nm/pixel (overrides MRC header value)')
def main(micrographs: Optional[Path], checkpoint: Optional[Path],
         model_type: str, output: Path, config: Optional[Path], device: str,
         pixel_size: Optional[float]):
    """
    Interactive annotation tool for cryo-EM micrographs using SAM.
    
    This tool allows you to click on objects in micrographs and automatically
    segment them using Segment Anything Model (SAM).
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
    
    # Run annotation
    annotate_micrographs(
        micrograph_folder=micrograph_folder,
        checkpoint_path=checkpoint_path,
        output_folder=output,
        model_type=model_type,
        device=device_str,
        pixel_size_nm=pixel_size,
    )


if __name__ == '__main__':
    main()

