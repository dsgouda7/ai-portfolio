from __future__ import annotations


class ArtifactValidationError(Exception):
    """Base class for failures that must keep a release unavailable."""


class ManifestValidationError(ArtifactValidationError):
    """The release manifest does not satisfy the frozen v1 contract."""


class ArtifactResolutionError(ArtifactValidationError):
    """A manifest URI cannot be resolved to an approved local artifact."""


class DigestMismatchError(ArtifactValidationError):
    """A resolved artifact does not match its declared digest."""


class CompatibilityError(ArtifactValidationError):
    """The release is incompatible with the configured serving runtime."""


class LifecycleError(RuntimeError):
    """A serving lifecycle transition was attempted out of order."""


class RequestValidationError(ValueError):
    """A chat request is malformed or exceeds a serving bound."""


class ServiceNotReadyError(RuntimeError):
    """The scoring service has not completed verification and warm-up."""


class OverloadedError(RuntimeError):
    """The service is already handling its bounded concurrent workload."""
