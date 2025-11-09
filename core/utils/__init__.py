# --- compat shims for older imports -----------------------------------------
from __future__ import annotations
from pathlib import Path
import json
import logging
import os

# Small logger so modules can do: from core.utils import log
def get_logger(name: str = "invest_ai"):
    lvl = os.getenv("LOG_LEVEL", "INFO").upper()
    logger = logging.getLogger(name)
    if not logger.handlers:
        logging.basicConfig(level=getattr(logging, lvl, logging.INFO),
                            format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return logger

log = get_logger()

def _find_project_root() -> Path:
    # start at this file and walk up looking for a 'config' directory
    here = Path(__file__).resolve()
    for parent in [here.parent] + list(here.parents):
        if (parent / "config" / "config.yaml").exists():
            return parent
    # fallback: assume two levels up (project root)
    return here.parents[2]

# Some modules import:  from core.utils import load_config, load_json
def load_config(path: str | None = None):
    try:
        import yaml  # lazy import
    except Exception as e:
        raise RuntimeError("PyYAML is required to load config.yaml") from e
    if path:
        cfg_path = Path(path)
    else:
        root = _find_project_root()
        cfg_path = root / "config" / "config.yaml"
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)

def load_json(path: str):
    pth = Path(path)
    if not pth.is_absolute():
        root = _find_project_root()
        pth = root / pth
    with open(pth, "r") as f:
        return json.load(f)
# -----------------------------------------------------------------------------
