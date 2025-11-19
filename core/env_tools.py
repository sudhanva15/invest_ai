"""Compatibility layer exposing environment helpers at core.env_tools.

This module re-exports the helpers from core.utils.env_tools so callers can
import using the simplified path required by hosting environments.
"""

from core.utils.env_tools import (  # noqa: F401
    load_env_once,
    env_flag,
    is_demo_mode,
    is_production_env,
    load_config,
    ensure_dirs,
)

__all__ = [
    "load_env_once",
    "env_flag",
    "is_demo_mode",
    "is_production_env",
    "load_config",
    "ensure_dirs",
]
