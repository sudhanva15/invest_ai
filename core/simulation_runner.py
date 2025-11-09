import pandas as pd
def simulate_dca_calendar(curve: pd.Series, plans: dict[str, dict]):
    """Return a mapping plan_name -> {'invested': Series, 'value': Series}.

    Parameters
    ----------
    curve : pd.Series
        Portfolio price index (e.g., normalized to 1.0 at start), indexed by trading dates.
    plans : dict[str, dict]
        Plan spec: {'PlanName': {'lump': 5000, 'monthly': 750}, ...}

    Notes
    -----
    - Monthly contributions are scheduled on calendar month-end, executed on the
      nearest previous trading day (pad). A single contribution per month.
    - 'invested' is the cumulative cash invested over time (lump + monthly DCA).
    - 'value' is the market value of the DCA strategy over time given `curve`.
    """
    out = {}
    if curve is None or len(curve) == 0:
        return out
    curve = curve.dropna().sort_index()
    if curve.empty:
        return out

    # Precompute month-end trading dates aligned to curve index
    month_end_idx = pd.Series(1, index=curve.index).resample("ME").last().index
    month_trade_dates = []
    used = set()
    for d in month_end_idx:
        pos = curve.index.get_indexer([d], method="pad")
        if pos[0] == -1:
            continue
        ts = curve.index[pos[0]]
        if ts in used:
            continue
        used.add(ts)
        month_trade_dates.append(ts)

    for name, spec in plans.items():
        lump = float(spec.get("lump", 0.0))
        monthly = float(spec.get("monthly", 0.0))

        # Market value via shared engine
        value = simulate_dca_calendar_series(curve, monthly=monthly, lump=lump)

        # Build invested (cumulative cash) series
        invested = pd.Series(0.0, index=curve.index)
        # Lump at first date if any
        if lump > 0.0:
            invested.iloc[0] += lump
        # Monthly contributions at month-end trading dates
        if monthly > 0.0:
            for ts in month_trade_dates:
                invested.loc[ts] += monthly
        invested = invested.cumsum()
        invested.name = f"INVESTED_${int(monthly)}_Lump_${int(lump)}"

        out[name] = {"invested": invested, "value": value}

    return out

def simulate_contributions(curve: pd.Series, monthly: float) -> float:
    """
    DCA-only final value using the provided normalized price index (curve).
    Buys on calendar month-end, executed on the nearest previous trading day.
    Returns the final market value of accumulated DCA shares.
    """
    if curve is None or len(curve) == 0 or not monthly or monthly <= 0:
        return 0.0
    s = curve.dropna().sort_index()
    if s.empty:
        return 0.0

    # Month-end trading dates (pad to prior trading day)
    month_end_idx = pd.Series(1, index=s.index).resample("ME").last().index
    used = set()
    buy_dates = []
    for d in month_end_idx:
        pos = s.index.get_indexer([d], method="pad")
        if pos[0] == -1:
            continue
        ts = s.index[pos[0]]
        if ts in used:
            continue
        used.add(ts)
        buy_dates.append(ts)

    # Accumulate shares at each buy date
    shares = 0.0
    for ts in buy_dates:
        px = float(s.loc[ts])
        if px > 0:
            shares += (monthly / px)

    # Final value at last index point
    return float(shares * float(s.iloc[-1]))

def simulate_dca_calendar_series(curve: pd.Series, monthly: float, lump: float = 0.0) -> pd.Series:
    """
    DCA-only market value series using the provided normalized price index (curve).
    Buys on calendar month-end, executed on the nearest previous trading day.
    Returns a Series of market value of accumulated DCA shares over time.
    """

    if curve is None or len(curve) == 0 or (not monthly and not lump):
        return pd.Series(dtype="float64")
    s = curve.dropna().sort_index()
    if s.empty:
        return pd.Series(dtype="float64")

    # Month-end trading dates (pad to prior trading day)
    month_end_idx = pd.Series(1, index=s.index).resample("ME").last().index
    used = set()
    buy_dates = []
    for d in month_end_idx:
        pos = s.index.get_indexer([d], method="pad")
        if pos[0] == -1:
            continue
        ts = s.index[pos[0]]
        if ts in used:
            continue
        used.add(ts)
        buy_dates.append(ts)

    # Accumulate shares at each buy date
    shares = 0.0
    value = pd.Series(0.0, index=s.index)
    for ts in s.index:
        # Buy on this date if it's a scheduled buy date
        if ts in buy_dates:
            px = float(s.loc[ts])
            if monthly > 0.0 and px > 0.0:
                shares += (monthly / px)
        # Lump sum at first date
        if ts == s.index[0] and lump > 0.0:
            px = float(s.loc[ts])
            if px > 0.0:
                shares += (lump / px)
        # Update market value
        value.loc[ts] = shares * float(s.loc[ts])

    value.name = f"DCA_Value_Monthly_${int(monthly)}_Lump_${int(lump)}"
    return value
def compute_xirr(cash_flows: pd.Series) -> float:
    """
    Compute the annualized internal rate of return (XIRR) for irregular cash flows.

    Parameters
    ----------
    cash_flows : pd.Series
        Series of cash flows indexed by dates (datetime-like). Outflows should be negative,
        inflows positive.

    Returns
    -------
    float
        Annualized rate as a decimal (e.g., 0.05 for 5%), or NaN if it cannot be computed.
    """
    import numpy as np

    if cash_flows is None or len(cash_flows) == 0:
        return float("nan")
    s = cash_flows.dropna()
    if s.empty:
        return float("nan")

    try:
        idx = pd.to_datetime(s.index)
    except Exception:
        return float("nan")
    s.index = idx

    dates = s.index.to_pydatetime()
    t0 = dates[0]
    times = np.array([(d - t0).days for d in dates], dtype=float) / 365.0
    amounts = s.values.astype(float)

    def npv(rate):
        return float(np.sum(amounts / (1.0 + rate) ** times))

    # Try Newton-Raphson starting from 10%
    rate = 0.10
    tol = 1e-7
    maxiter = 100
    for _ in range(maxiter):
        denom = (1.0 + rate) ** times
        if np.any(denom == 0):
            break
        f = np.sum(amounts / denom)
        df = np.sum(-times * amounts / (denom * (1.0 + rate)))
        if abs(f) < tol:
            return float(rate)
        if df == 0 or not np.isfinite(df):
            break
        rate -= f / df
        if not np.isfinite(rate) or rate <= -0.99999999:
            break

    # Fallback: bisection search over a wide interval
    low, high = -0.9999, 10.0
    f_low, f_high = npv(low), npv(high)
    if f_low * f_high > 0:
        return float("nan")
    for _ in range(200):
        mid = (low + high) / 2.0
        f_mid = npv(mid)
        if abs(f_mid) < tol:
            return float(mid)
        if f_low * f_mid < 0:
            high = mid
            f_high = f_mid
        else:
            low = mid
            f_low = f_mid
    return float(mid)

__all__ = ["simulate_contributions", "simulate_dca_calendar_series", "compute_xirr", "simulate_dca_calendar"]
