from typing import Optional

import pandas as pd
import numpy as np
from core.multifactor import portfolio_metrics, compute_risk_contributions, portfolio_passes_filters
from core.recommendation_engine import build_recommendations, ObjectiveConfig
from core.utils.env_tools import load_config

CFG = load_config()

class DummyRiskProfile:
    """Lightweight stand-in for RiskProfileResult used in filter tests."""

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
            # Simple fallback: assume target equals min when only min provided
            self.cagr_target = cagr_min

# Synthetic returns with two clearly distinct asset classes (equity vs bond)
# Equity higher vol, bond lower vol – ensures diversification ratio > 1.0 typically

def _make_returns(days=400, seed=11):
    rng = np.random.default_rng(seed)
    dates = pd.date_range('2022-01-03', periods=days, freq='B')
    # annual vol targets
    eq_vol = 0.18
    bond_vol = 0.07
    eq_daily = eq_vol / np.sqrt(252)
    bond_daily = bond_vol / np.sqrt(252)
    data = {
        'EQ1': rng.normal(loc=0.0003, scale=eq_daily, size=days),
        'EQ2': rng.normal(loc=0.0003, scale=eq_daily, size=days),
        'BOND1': rng.normal(loc=0.00010, scale=bond_daily, size=days),
        'BOND2': rng.normal(loc=0.00010, scale=bond_daily, size=days),
    }
    return pd.DataFrame(data, index=dates)


def test_diversification_ratio_and_asset_classes_mapping():
    rets = _make_returns()
    # Catalog with explicit asset classes
    catalog = {
        'assets': [
            {'symbol': 'EQ1', 'asset_class': 'equity', 'core_or_satellite': 'core'},
            {'symbol': 'EQ2', 'asset_class': 'equity', 'core_or_satellite': 'core'},
            {'symbol': 'BOND1', 'asset_class': 'bond', 'core_or_satellite': 'core'},
            {'symbol': 'BOND2', 'asset_class': 'bond', 'core_or_satellite': 'core'},
        ]
    }

    # Build relaxed cfg inline to avoid mutating global when key absent in test environment
    relaxed_cfg = dict(CFG or {})
    mf = dict(relaxed_cfg.get('multifactor', {}))
    mf.update({
        'min_portfolio_sharpe': -1.0,
        'max_portfolio_drawdown': -0.90,
        'min_diversification_ratio': 1.0,
        'min_holdings': 2,
    })
    relaxed_cfg['multifactor'] = mf
    uni = dict(relaxed_cfg.get('universe', {}))
    uni.update({'core_min_years': 0.1, 'sat_min_years': 0.1})
    relaxed_cfg['universe'] = uni
    rp = DummyRiskProfile(band_min_vol=0.10, band_max_vol=0.30)

    obj_cfg = ObjectiveConfig(name='Balanced', universe_filter=None,
                              bounds={'core_min': 0.65, 'sat_max_total': 0.35, 'sat_max_single': 0.07}, optimizer='hrp')

    result = build_recommendations(
        returns=rets,
        catalog=catalog,
        cfg=relaxed_cfg,
        risk_profile=rp,
        objective_cfg=obj_cfg,
        n_candidates=4,
        seed=42,
    )
    recommended = result.get('recommended', [])
    assert recommended, 'Expected at least one recommended portfolio'

    # Check diversification ratio > 1.0 for at least one portfolio (indicative of diversification benefit)
    assert any(c.get('metrics', {}).get('diversification_ratio', 0) >= 1.0 for c in recommended), 'No portfolio shows diversification benefit'

    # Ensure asset_classes mapping injected and includes >1 distinct class
    first = recommended[0]
    ac_map = first.get('asset_classes')
    assert isinstance(ac_map, dict) and ac_map, 'asset_classes mapping missing'
    distinct = set(ac_map.values()) - {'unknown'}
    assert len(distinct) >= 2, f'Expected >=2 distinct asset classes, got {distinct}'

    # Direct portfolio metrics path: compute metrics & risk contributions, then pass filters
    w = pd.Series(first['weights'])
    pm = portfolio_metrics(w, rets)
    assert pm.get('valid'), 'Portfolio metrics invalid'
    cov = rets.cov()
    rc = compute_risk_contributions(w, cov)
    passed, reason = portfolio_passes_filters(pm, rc, CFG, rp)
    assert passed, f'Portfolio failed filters unexpectedly: {reason}'

