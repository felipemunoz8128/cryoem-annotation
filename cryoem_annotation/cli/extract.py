"""CLI for extraction tool."""

import click
from pathlib import Path
from typing import Optional


@click.command()
@click.option('--results', '-r', type=click.Path(exists=True, path_type=Path),
              required=True, help='Path to annotation results folder')
@click.option('--output', '-o', type=click.Path(path_type=Path),
              help='Output file base path (default: results in results folder)')
@click.option('--format', '-f', type=click.Choice(['csv', 'json', 'both']),
              default='csv', help='Output format')
@click.option('--pixel-size', '-p', type=float, default=None,
              help='Pixel size in nm/pixel (overrides metadata values)')
def main(results: Path, output: Optional[Path], format: str,
         pixel_size: Optional[float]):
    """
    Extract labels and areas from annotation results.

    This tool reads all metadata.json files from annotation results and extracts
    the labels and mask areas for each segmented object.

    Output is split into two CSV files:
      - *_metadata.csv: Contains segmentation_id, micrograph_name, click_index, etc.
      - *_results.csv: Contains segmentation_id, label, area_pixels, area_nm2
    """
    # Lazy import to speed up CLI startup
    from cryoem_annotation.extraction.extractor import extract_results

    extract_results(
        results_folder=results,
        output_path=output,
        output_format=format,
        pixel_size_override=pixel_size,
    )


if __name__ == '__main__':
    main()

