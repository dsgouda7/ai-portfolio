"""Setup script to install shared utilities as a package."""

from setuptools import setup, find_packages

setup(
    name="rag-shared",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pyyaml>=6.0",
    ],
    python_requires=">=3.9",
)
