
from dotenv import dotenv_values
import os
from pathlib import Path
from functools import lru_cache

def load_env_once(dotenv_path: str | None = None):
    """
    Load .env without relying on find_dotenv() to avoid assertion errors in -c / REPL contexts.
    Call this early (e.g., in app entrypoint).
    """
    dp = dotenv_path or ".env"
    if not os.environ.get("_ENV_LOADED", ""):
        if Path(dp).exists():
            env = dotenv_values(dp)
            for k, v in env.items():
                if v is not None and k not in os.environ:
                    os.environ[k] = str(v)
        os.environ["_ENV_LOADED"] = "1"

# === [WIRING PATCH] Safe config + path bootstrap ============================
from pathlib import Path
import os

try:
    import yaml
except ImportError:
    yaml = None

def load_config(config_path: str | Path = "config.yaml") -> dict:
    """Load YAML config safely with sane defaults if missing keys."""
    p = Path(config_path)
    if yaml is None or not p.exists():
        return {
            "paths": {"raw_dir": "data/raw", "processed_dir": "data/processed", "outputs_dir": "data/outputs"},
            "data": {"default_universe": ["SPY","AGG","GLD","QQQ"], "use_yfinance_fallback": True},
            "app": {"rebalance_freq": "monthly"},
            "risk": {"optimizer": "hrp", "min_weight": 0.0, "max_weight": 0.3, "risk_free_rate": 0.015},
        }
    with p.open("r") as f:
        cfg = yaml.safe_load(f) or {}
    # backfill minimal keys if cfg is partial
    cfg.setdefault("paths", {})
    cfg["paths"].setdefault("raw_dir", "data/raw")
    cfg["paths"].setdefault("processed_dir", "data/processed")
    cfg["paths"].setdefault("outputs_dir", "data/outputs")
    cfg.setdefault("data", {})
    cfg["data"].setdefault("default_universe", ["SPY","AGG","GLD","QQQ"])
    cfg["data"].setdefault("use_yfinance_fallback", True)
    cfg.setdefault("app", {})
    cfg["app"].setdefault("rebalance_freq", "monthly")
    cfg.setdefault("risk", {})
    cfg["risk"].setdefault("optimizer", "hrp")
    cfg["risk"].setdefault("min_weight", 0.0)
    cfg["risk"].setdefault("max_weight", 0.3)
    cfg["risk"].setdefault("risk_free_rate", 0.015)
    # Universe validation thresholds (optional)
    cfg.setdefault("universe", {})
    cfg["universe"].setdefault("core_min_years", 10.0)
    cfg["universe"].setdefault("sat_min_years", 7.0)
    cfg["universe"].setdefault("max_missing_pct", 10.0)
    return cfg

def ensure_dirs(cfg: dict) -> None:
    """Create directories declared in config if they don't exist."""
    for k in ("raw_dir", "processed_dir", "outputs_dir"):
        d = Path(cfg["paths"].get(k, "")).expanduser()
        if d:
            d.mkdir(parents=True, exist_ok=True)
# =========================================================================== 

def _truthy(value, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def env_flag(name: str, default: bool | str = False) -> bool:
    """Return boolean interpretation of an environment flag (loads .env once)."""
    load_env_once()
    val = os.getenv(name)
    if val is None:
        return _truthy(default, default=False)
    return _truthy(val, default=False)


@lru_cache(maxsize=None)
def is_demo_mode() -> bool:
    """True when INVEST_AI_DEMO is truthy (1/true/yes)."""
    return env_flag("INVEST_AI_DEMO", False)


@lru_cache(maxsize=None)
def is_production_env() -> bool:
    """True when INVEST_AI_ENV is 'production'."""
    load_env_once()
    return os.getenv("INVEST_AI_ENV", "").strip().lower() == "production"