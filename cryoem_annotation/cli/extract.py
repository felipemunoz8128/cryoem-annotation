"""CLI for extraction tool."""

import click
from pathlib import Path
from typing import Optional

from cryoem_annotation.extraction.extractor import extract_results


@click.command()
@click.option('--results', '-r', type=click.Path(exists=True, path_type=Path),
              required=True, help='Path to annotation results folder')
@click.option('--output', '-o', type=click.Path(path_type=Path),
              help='Output file path (default: results.csv/json in results folder)')
@click.option('--format', '-f', type=click.Choice(['csv', 'json', 'both']),
              default='csv', help='Output format')
def main(results: Path, output: Optional[Path], format: str):
    """
    Extract labels and areas from annotation results.
    
    This tool reads all metadata.json files from annotation results and extracts
    the labels and mask areas for each segmented object, saving them to CSV or JSON.
    """
    extract_results(
        results_folder=results,
        output_path=output,
        output_format=format,
    )


if __name__ == '__main__':
    main()

