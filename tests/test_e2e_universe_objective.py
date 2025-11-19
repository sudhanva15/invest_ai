from __future__ import annotations

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from core.objective_mapper import load_objectives_config
from core.recommendation_engine import ObjectiveConfig, build_recommendations
from core.risk_profile import RiskProfileResult
from core.universe_yaml import load_universe_from_yaml
from core.utils.env_tools import load_config


def _synthetic_returns(symbols: list[str], periods: int = 504) -> pd.DataFrame:
    rng = np.random.default_rng(123)
    dates = pd.date_range("2015-01-01", periods=periods, freq="B")
    data = rng.normal(loc=0.0003, scale=0.01, size=(len(dates), len(symbols)))
    return pd.DataFrame(data, index=dates, columns=symbols)


def test_build_recommendations_balanced_offline():
    universe = load_universe_from_yaml()
    assert not universe.empty, "universe.yaml should define at least one asset"

    objectives = load_objectives_config()
    assert "BALANCED" in objectives, "BALANCED objective missing from objectives.yaml"
    balanced_v4 = objectives["BALANCED"]

    symbols = universe["symbol"].dropna().astype(str).unique().tolist()
    symbols = [s for s in symbols if s]
    assert symbols, "No symbols available from universe.yaml"
    focus_symbols = symbols[:12]

    returns = _synthetic_returns(focus_symbols)

    catalog_path = Path("config/assets_catalog.json")
    catalog = json.loads(catalog_path.read_text())

    cfg = load_config()

    profile = RiskProfileResult(
        questionnaire_score=50,
        facts_score=50,
        combined_score=50,
        slider_score=50,
        true_risk=50,
        vol_min=0.10,
        vol_target=0.13,
        vol_max=0.16,
        cagr_min=0.05,
        cagr_target=0.07,
        label="Balanced",
        horizon_years=10,
        objective="BALANCED",
    )

    objective_cfg = ObjectiveConfig(
        name="Balanced",
        universe_filter=None,
        bounds={"core_min": 0.65, "sat_max_total": 0.35, "sat_max_single": 0.07},
        optimizer="hrp",
        notes="Balanced offline objective",
    )

    result = build_recommendations(
        returns=returns,
        catalog=catalog,
        cfg=cfg,
        risk_profile=profile,
        objective_cfg=objective_cfg,
        n_candidates=4,
        seed=42,
        objective_config=balanced_v4,
    )

    recommended = result.get("recommended", [])
    assert recommended, "Expected at least one offline recommendation"