"""CLI for extraction tool."""

import click
from pathlib import Path
from typing import Optional


@click.command()
@click.option('--results', '-r', type=click.Path(exists=True, path_type=Path),
              required=True, help='Path to annotation results folder')
@click.option('--micrographs', '-m', type=click.Path(exists=True, path_type=Path),
              help='Path to micrograph folder (for accurate total count in summary)')
@click.option('--output', '-o', type=click.Path(path_type=Path),
              help='Output file base path (default: extraction in results folder)')
@click.option('--format', '-f', type=click.Choice(['csv', 'json', 'both']),
              default='csv', help='Output format')
@click.option('--pixel-size', '-p', type=float, default=None,
              help='Pixel size in nm/pixel (overrides metadata values)')
def main(results: Path, micrographs: Optional[Path], output: Optional[Path],
         format: str, pixel_size: Optional[float]):
    """
    Extract labels and areas from annotation results.

    This tool reads all metadata.json files from annotation results and extracts
    the labels and mask areas for each segmented object.

    Output is split into two CSV files:
      - *_metadata.csv: Contains segmentation_id, micrograph_name, click_index, etc.
      - *_results.csv: Contains segmentation_id, label, diameter_nm
    """
    # Lazy import to speed up CLI startup
    from cryoem_annotation.extraction.extractor import extract_results

    extract_results(
        results_folder=results,
        micrograph_folder=micrographs,
        output_path=output,
        output_format=format,
        pixel_size_override=pixel_size,
    )


if __name__ == '__main__':
    main()

