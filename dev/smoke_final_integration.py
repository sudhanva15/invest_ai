#!/usr/bin/env python3
"""Offline smoke test for universe/objective integration."""

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
    rng = np.random.default_rng(7)
    dates = pd.date_range("2014-01-01", periods=periods, freq="B")
    data = rng.normal(loc=0.0004, scale=0.009, size=(len(dates), len(symbols)))
    return pd.DataFrame(data, index=dates, columns=symbols)


def main() -> None:
    universe = load_universe_from_yaml()
    asset_counts = universe["asset_class"].value_counts().to_dict()
    print("Universe asset counts by class:")
    for asset_class, count in asset_counts.items():
        print(f"  - {asset_class}: {count}")

    objectives = load_objectives_config()
    print("\nAvailable objectives:")
    for name in objectives.keys():
        print(f"  - {name}")

    symbols = universe["symbol"].dropna().astype(str).unique().tolist()
    if not symbols:
        raise RuntimeError("Universe contains no symbols; cannot run smoke test")
    focus_symbols = symbols[:15]

    returns = _synthetic_returns(focus_symbols)

    catalog = json.loads(Path("config/assets_catalog.json").read_text())
    cfg = load_config()

    profile = RiskProfileResult(
        questionnaire_score=60,
        facts_score=60,
        combined_score=60,
        slider_score=60,
        true_risk=60,
        vol_min=0.12,
        vol_target=0.16,
        vol_max=0.20,
        cagr_min=0.06,
        cagr_target=0.09,
        label="Growth",
        horizon_years=12,
        objective="GROWTH",
    )

    growth_v4 = objectives["GROWTH"]
    objective_cfg = ObjectiveConfig(
        name="Growth",
        universe_filter=None,
        bounds={"core_min": 0.60, "sat_max_total": 0.40, "sat_max_single": 0.08},
        optimizer="hrp",
        notes="Offline growth smoke objective",
    )

    result = build_recommendations(
        returns=returns,
        catalog=catalog,
        cfg=cfg,
        risk_profile=profile,
        objective_cfg=objective_cfg,
        n_candidates=5,
        seed=99,
        objective_config=growth_v4,
    )

    recommended = result.get("recommended", [])
    stats = result.get("stats") or {}
    print("\nOffline recommendation summary:")
    print(f"  Portfolios returned: {len(recommended)}")
    if stats:
        print(
            f"  Stats → total={stats.get('total_candidates', 0)}, strict={stats.get('strict_passes', 0)}, "
            f"recommended={stats.get('recommended', 0)}, fallback={stats.get('fallback_count', 0)}, "
            f"hard={stats.get('hard_fallback_count', 0)}"
        )
    if recommended:
        print(f"  Top portfolio: {recommended[0].get('name', 'unknown')}")


if __name__ == "__main__":
    main()
