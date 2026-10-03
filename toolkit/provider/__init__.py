"""Credential and endpoint resolution for AI providers. Imports nothing."""

from .models import ApiStyle, Provider, SetupError
from .resolve import (
    BASE_VAR,
    DEFAULT_ENV_FILE,
    KEY_VAR,
    MODEL_VAR,
    SHORTCUTS,
    STYLE_VAR,
    Diagnosis,
    available,
    diagnose,
    load_settings,
    resolve,
)

__all__ = [
    "ApiStyle",
    "Provider",
    "SetupError",
    "Diagnosis",
    "resolve",
    "diagnose",
    "available",
    "load_settings",
    "SHORTCUTS",
    "KEY_VAR",
    "STYLE_VAR",
    "BASE_VAR",
    "MODEL_VAR",
    "DEFAULT_ENV_FILE",
]
