"""Project-level exception types."""


class DiffClsError(Exception):
    """Base class for every error this package raises on purpose."""


class ConfigError(DiffClsError):
    """A configuration value is invalid or inconsistent with another value."""


class CheckpointError(DiffClsError):
    """A checkpoint file is missing or does not hold the expected content."""


class ResultsError(DiffClsError):
    """A results file or metric cannot be computed or read."""


class TrainingError(DiffClsError):
    """Training reached a state that makes the rest of the run useless (for example a NaN loss)."""
