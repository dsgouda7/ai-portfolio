"""Public model-bundle API.

The implementation lives separately so callers get one stable import surface while bundle layout
and serialization details can evolve behind it.
"""

from roadid.training.packaging_impl import load_tiny_bundle, package_bundle, verify_bundle

__all__ = ["load_tiny_bundle", "package_bundle", "verify_bundle"]
