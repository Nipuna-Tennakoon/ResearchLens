"""Errors that carry a message meant to be shown to the user as-is."""


class ResearchLensError(Exception):
    """Base class for expected failures reported to the user without a traceback."""


class ConfigError(ResearchLensError):
    """Configuration is missing or invalid."""


class IngestionError(ResearchLensError):
    """A document could not be read, parsed or embedded."""


class VectorStoreError(ResearchLensError):
    """Milvus is unreachable or rejected an operation."""
