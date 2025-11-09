import os, json, yaml, logging, logging.config
from pathlib import Path
from dotenv import load_dotenv

def load_logger():
    cfg = Path("config/logging_config.yaml")
    if cfg.exists():
        with open(cfg, "r") as f:
            logging.config.dictConfig(yaml.safe_load(f))
    else:
        logging.basicConfig(level=logging.INFO)
    return logging.getLogger("invest_ai")

log = load_logger()

def load_config():
    with open("config/config.yaml","r") as f:
        return yaml.safe_load(f)

def load_env():
    load_dotenv("config/credentials.env")
    return {k:v for k,v in os.environ.items() if k.endswith("_API_KEY")}


def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)
