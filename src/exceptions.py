"""Project-wide exception hierarchy.

Every module raises one of these types so callers can catch domain errors
without relying on generic exceptions, and so error messages always name the
offending component (IMPLEMENTATION_PLAN.md §5.2).
"""

from __future__ import annotations


class ConflictForecastError(Exception):
    """Base class for all errors raised by this project."""


class ConfigurationError(ConflictForecastError):
    """Raised when the project configuration is invalid."""


class DataLoadError(ConflictForecastError):
    """Raised when raw data cannot be discovered, read, or canonicalized."""


class DataValidationError(ConflictForecastError):
    """Raised when a data-quality rule is violated."""


class FeatureEngineeringError(ConflictForecastError):
    """Raised when feature construction fails (used from M5)."""


class LabelEngineeringError(ConflictForecastError):
    """Raised when label construction or validation fails (used from M6)."""


class SplitError(ConflictForecastError):
    """Raised when the chronological train/val/test split is invalid (used from M7)."""


class ModelError(ConflictForecastError):
    """Raised when a model cannot be trained, saved, or loaded (used from M8)."""


class EvaluationError(ConflictForecastError):
    """Raised when evaluation or report generation fails (used from M10)."""


class ExplainabilityError(ConflictForecastError):
    """Raised when SHAP computation or explanation reports fail (used from M10)."""


class VisualizationError(ConflictForecastError):
    """Raised when a plot or interactive artifact cannot be produced (used from M12)."""


class ForecastError(ConflictForecastError):
    """Raised when the live 14-day forecast cannot be produced (used from --stage forecast)."""
