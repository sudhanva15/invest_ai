from importlib.metadata import version, PackageNotFoundError
__all__ = ["data_ingestion","preprocessing","risk_metrics","portfolio_engine","backtesting","recommendation_engine","simulation_runner","utils"]
try:
    __version__ = version("invest_ai")
except PackageNotFoundError:
    __version__ = "0.0.1"

# Re-export common utilities for convenience and backward-compatibility
try:
    from .utils import load_config, load_json  # type: ignore
    __all__ += ["load_config", "load_json"]
except Exception:
    # utils may not define these in early bootstrap; keep import optional
    pass
