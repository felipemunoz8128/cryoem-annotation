"""Setup script for cryoem-annotation package."""

from setuptools import setup, find_packages

setup(
    name="cryoem-annotation",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "matplotlib>=3.5.0",
        "opencv-python>=4.5.0",
        "torch>=1.7.0",
        "torchvision>=0.8.0",
        "mrcfile>=1.4.0",
        "pyyaml>=6.0",
        "click>=8.0",
    ],
    entry_points={
        "console_scripts": [
            "cryoem-annotate=cryoem_annotation.cli.annotate:main",
            "cryoem-label=cryoem_annotation.cli.label:main",
            "cryoem-extract=cryoem_annotation.cli.extract:main",
        ],
    },
)

