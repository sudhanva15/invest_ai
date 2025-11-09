
from dotenv import dotenv_values
import os
from pathlib import Path

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
    return cfg

def ensure_dirs(cfg: dict) -> None:
    """Create directories declared in config if they don't exist."""
    for k in ("raw_dir", "processed_dir", "outputs_dir"):
        d = Path(cfg["paths"].get(k, "")).expanduser()
        if d:
            d.mkdir(parents=True, exist_ok=True)
# =========================================================================== 