from __future__ import annotations
from pathlib import Path
import pandas as pd

_DEFAULT_DIR = Path("data") / "cache"
_DEFAULT_DIR.mkdir(parents=True, exist_ok=True)

def _to_path(key: str | Path) -> Path:
    """
    Accept either a filename or a full path. If `key` has no parent,
    write it under data/cache/. Ensures .csv extension.
    """
    p = Path(key)
    if p.suffix.lower() != ".csv":
        p = p.with_suffix(".csv")
    if p.parent == Path("."):
        p = _DEFAULT_DIR / p
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

def read_cache(key: str | Path) -> pd.DataFrame | None:
    """Return DataFrame if cached file exists and is readable; else None."""
    try:
        p = _to_path(key)
        if not p.exists():
            return None
        return pd.read_csv(p)
    except Exception:
        return None

def write_cache(key: str | Path, df: pd.DataFrame) -> None:
    """Persist a DataFrame to CSV (best-effort; silent on failure)."""
    try:
        p = _to_path(key)
        df.to_csv(p, index=False)
    except Exception:
        pass

class _CacheFrame:
    """Small wrapper to match provider_registry's .read/.write usage."""
    def read(self, key: str | Path) -> pd.DataFrame | None:
        return read_cache(key)
    def write(self, key: str | Path, df: pd.DataFrame) -> None:
        write_cache(key, df)

# What provider_registry imports:
cache_frame = _CacheFrame()
