"""CLI for extraction tool."""

import click
from pathlib import Path
from typing import Optional


@click.command()
@click.option('--results', '-r', type=click.Path(exists=True, path_type=Path),
              help='Path to annotation results folder (auto-detected if in project directory)')
@click.option('--micrographs', '-m', type=click.Path(exists=True, path_type=Path),
              help='Path to micrograph folder (auto-loaded from project file if not specified)')
@click.option('--output', '-o', type=click.Path(path_type=Path),
              help='Output file base path (default: extraction in results folder)')
@click.option('--format', '-f', type=click.Choice(['csv', 'json', 'both']),
              default='csv', help='Output format')
@click.option('--pixel-size', '-p', type=float, default=None,
              help='Pixel size in nm/pixel (overrides metadata values)')
def main(results: Optional[Path], micrographs: Optional[Path], output: Optional[Path],
         format: str, pixel_size: Optional[float]):
    """
    Extract labels and areas from annotation results.

    This tool reads all metadata.json files from annotation results and extracts
    the labels and mask areas for each segmented object.

    Output is split into two CSV files:
      - *_metadata.csv: Contains segmentation_id, micrograph_name, click_index, etc.
      - *_results.csv: Contains segmentation_id, label, diameter_nm

    If run from a directory containing .cryoem-project.json (created by cryoem-annotate),
    the --results and --micrographs paths are auto-detected.
    """
    # Lazy import to speed up CLI startup
    from cryoem_annotation.core.project import resolve_paths
    from cryoem_annotation.extraction.extractor import extract_results

    # Resolve paths from CLI args and/or project file
    # CLI args take precedence; micrographs is optional for extraction
    resolved = resolve_paths(
        results=results,
        micrographs=micrographs,
        require_micrographs=False,
    )

    results_folder = resolved['results']
    micrograph_folder = resolved['micrographs']

    # Report if using auto-detected paths
    if resolved['project_file']:
        click.echo(f"[OK] Using project file: {resolved['project_file']}")

    extract_results(
        results_folder=results_folder,
        micrograph_folder=micrograph_folder,
        output_path=output,
        output_format=format,
        pixel_size_override=pixel_size,
    )


if __name__ == '__main__':
    main()

