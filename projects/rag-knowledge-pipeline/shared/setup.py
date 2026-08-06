"""Setup script to install shared utilities as a package."""

from setuptools import setup

setup(
    name="rag-shared",
    version="0.1.0",
    packages=["shared"],
    package_dir={"shared": "."},
    package_data={"shared": ["contracts/**/*.json", "contracts/**/*.md"]},
    install_requires=[
        "jsonschema[format]>=4.23.0,<5.0.0",
        "pyyaml>=6.0",
        "referencing>=0.35.0,<1.0.0",
    ],
    python_requires=">=3.9",
)
