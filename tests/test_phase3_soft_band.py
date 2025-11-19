from typing import Optional

import pandas as pd
import numpy as np
from core.recommendation_engine import ObjectiveConfig
from core.multifactor import portfolio_metrics, compute_risk_contributions, portfolio_passes_filters, compute_composite_score
from core.utils.env_tools import load_config

CFG = load_config()

# Synthetic helpers --------------------------------------------------------

class DummyRiskProfile:
    def __init__(
        self,
        band_min_vol: float,
        band_max_vol: float,
        true_risk: float = 50.0,
        cagr_min: Optional[float] = None,
    ):
        from core.risk_profile import map_true_risk_to_cagr_band

        self.band_min_vol = band_min_vol
        self.band_max_vol = band_max_vol
        self.true_risk = true_risk

        if cagr_min is None:
            cagr_band_min, cagr_band_target = map_true_risk_to_cagr_band(true_risk)
            self.cagr_min = cagr_band_min
            self.cagr_target = cagr_band_target
        else:
            self.cagr_min = cagr_min
            self.cagr_target = cagr_min


def _synthetic_returns(n_symbols=6, n_days=1000, annual_vol_target=0.14, seed=123):
    rng = np.random.default_rng(seed)
    daily_vol = annual_vol_target / np.sqrt(252)
    dates = pd.date_range("2020-01-01", periods=n_days, freq="B")
    data = {}
    for i in range(n_symbols):
        sym = f"SYM{i+1}"
        data[sym] = rng.normal(loc=0.0002, scale=daily_vol, size=n_days)  # small positive drift
    return pd.DataFrame(data, index=dates)


def test_soft_band_passes_and_penalizes():
    # Use explicit dummy profile for deterministic soft-band scenario
    rp = DummyRiskProfile(band_min_vol=0.07, band_max_vol=0.30)
    soft_lower = rp.band_min_vol * 0.6  # 0.042
    target_annual_vol = 0.15  # Asset vol; diversified portfolio ends up ~6% (<7%) so in soft band zone
    rets = _synthetic_returns(annual_vol_target=target_annual_vol)

    obj_cfg = ObjectiveConfig(
        name="Balanced",
        universe_filter=None,
        bounds={"core_min": 0.65, "sat_max_total": 0.35, "sat_max_single": 0.07},
        optimizer="hrp",
    )

    # Directly evaluate equal-weight portfolio to force soft-band scenario deterministically
    import pandas as pd
    weights = pd.Series({s: 1.0 / len(rets.columns) for s in rets.columns})
    pm = portfolio_metrics(weights, rets, risk_free_rate=CFG.get("optimization", {}).get("risk_free_rate", 0.015))
    cov = rets.cov()
    rc = compute_risk_contributions(weights, cov)
    passed, reason = portfolio_passes_filters(pm, rc, CFG, rp)
    assert passed, f"Equal-weight portfolio should pass soft-band filters (reason={reason})"
    assert pm.get("soft_band") is True, "Expected soft_band flag to be True when volatility in soft zone"

    # Composite score penalty check
    sharpe = pm["sharpe"]
    maxdd = pm["max_drawdown"]
    base_score = sharpe - float(CFG.get("multifactor", {}).get("drawdown_penalty_lambda", 0.2)) * abs(maxdd)
    score = compute_composite_score(
        sharpe=sharpe,
        max_drawdown=maxdd,
        lambda_penalty=float(CFG.get("multifactor", {}).get("drawdown_penalty_lambda", 0.2)),
        volatility=pm.get("volatility"),
        risk_profile=rp,
        soft_band=True,
        soft_vol_penalty_lambda=float(CFG.get("multifactor", {}).get("soft_vol_penalty_lambda", 0.5)),
    )
    assert score <= base_score + 1e-6, "Soft-band penalty should reduce or keep score <= base score"

