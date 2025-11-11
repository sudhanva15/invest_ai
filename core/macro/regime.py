"""
Macro regime labeling for portfolio evaluation.

Classifies market environments into distinct regimes (Risk-on, Disinflation, 
Tightening, Recessionary) based on interest rates, inflation, and unemployment.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional


def load_macro_data() -> pd.DataFrame:
    """
    Load macro indicators from data/macro directory.
    
    Returns:
        DataFrame with date index and columns: DGS10, CPI, UNRATE, etc.
    """
    try:
        # Determine repo root
        root = Path(__file__).resolve().parents[2]
        macro_dir = root / "data" / "macro"
        
        # Load key indicators
        indicators = {}
        
        for series_name in ["DGS10", "CPIAUCSL", "UNRATE", "FEDFUNDS"]:
            csv_path = macro_dir / f"{series_name}.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                if "date" in df.columns:
                    df["date"] = pd.to_datetime(df["date"])
                    df = df.set_index("date")
                elif df.index.name == "DATE" or "DATE" in df.columns:
                    if "DATE" in df.columns:
                        df["DATE"] = pd.to_datetime(df["DATE"])
                        df = df.set_index("DATE")
                    else:
                        df.index = pd.to_datetime(df.index)
                
                # Get value column (usually named after the series)
                val_col = [c for c in df.columns if c in [series_name, "value", "VALUE"]]
                if val_col:
                    indicators[series_name] = pd.to_numeric(df[val_col[0]], errors="coerce")
    except Exception:
        pass
    
    if not indicators:
        return pd.DataFrame()
    
    # Combine into single DataFrame
    macro_df = pd.DataFrame(indicators)
    return macro_df.sort_index()


def regime_features(macro_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Compute regime features from macro indicators.
    
    Args:
        macro_df: DataFrame with date index and columns: DGS10, CPIAUCSL, UNRATE, FEDFUNDS
                  If None, loads from data/macro directory
    
    Returns:
        DataFrame with z-scored features:
            - dgs10_level: 10-year treasury yield level
            - dgs10_6m_chg: 6-month change in 10-year yield
            - cpi_yoy: CPI year-over-year change
            - unrate_6m_chg: 6-month change in unemployment rate
    
    Notes:
        Features are z-scored (mean=0, std=1) for use in clustering.
    """
    if macro_df is None:
        macro_df = load_macro_data()
    
    if macro_df.empty:
        return pd.DataFrame()
    
    features = pd.DataFrame(index=macro_df.index)
    
    # DGS10: Level and 6-month change
    if "DGS10" in macro_df.columns:
        features["dgs10_level"] = macro_df["DGS10"]
        features["dgs10_6m_chg"] = macro_df["DGS10"].diff(126)  # ~6 months of trading days
    
    # CPI: Year-over-year change
    if "CPIAUCSL" in macro_df.columns:
        features["cpi_yoy"] = macro_df["CPIAUCSL"].pct_change(252)  # ~1 year
    
    # UNRATE: 6-month change
    if "UNRATE" in macro_df.columns:
        features["unrate_6m_chg"] = macro_df["UNRATE"].diff(126)
    
    # Fed Funds: Level (optional)
    if "FEDFUNDS" in macro_df.columns:
        features["fedfunds_level"] = macro_df["FEDFUNDS"]
    
    # Drop NaN and z-score
    features = features.dropna()
    
    if len(features) > 0:
        # Z-score normalization
        for col in features.columns:
            mean = features[col].mean()
            std = features[col].std()
            if std > 0:
                features[col] = (features[col] - mean) / std
    
    return features


def label_regimes(
    features: pd.DataFrame,
    method: str = "rule_based",
    k: int = 4
) -> pd.Series:
    """
    Label market regimes based on macro features.
    
    Args:
        features: DataFrame from regime_features() with z-scored indicators
        method: "rule_based" (simple thresholds) or "kmeans" (clustering)
        k: Number of clusters for KMeans (default: 4)
    
    Returns:
        Series with regime labels: "Risk-on", "Disinflation", "Tightening", "Recessionary"
    
    Rule-based logic:
        - Tightening: Rising rates (dgs10_6m_chg > 0.5) + high inflation (cpi_yoy > 0.5)
        - Recessionary: Rising unemployment (unrate_6m_chg > 0.5) + falling rates
        - Disinflation: Falling inflation (cpi_yoy < -0.3) + stable/falling rates
        - Risk-on: Everything else (default)
    """
    if features.empty:
        return pd.Series([], dtype=str)
    
    labels = pd.Series("Risk-on", index=features.index)
    
    if method == "rule_based":
        # Extract features (handle missing columns gracefully)
        dgs10_chg = features.get("dgs10_6m_chg", pd.Series(0, index=features.index))
        cpi_yoy = features.get("cpi_yoy", pd.Series(0, index=features.index))
        unrate_chg = features.get("unrate_6m_chg", pd.Series(0, index=features.index))
        
        # Tightening: Rising rates + high inflation
        tightening = (dgs10_chg > 0.5) & (cpi_yoy > 0.5)
        
        # Recessionary: Rising unemployment + falling rates
        recessionary = (unrate_chg > 0.5) & (dgs10_chg < -0.3)
        
        # Disinflation: Falling inflation + stable/falling rates
        disinflation = (cpi_yoy < -0.3) & (dgs10_chg < 0.3)
        
        # Apply labels (priority order)
        labels[recessionary] = "Recessionary"
        labels[tightening] = "Tightening"
        labels[disinflation] = "Disinflation"
        
    elif method == "kmeans":
        try:
            from sklearn.cluster import KMeans
            
            # Fit KMeans
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_ids = kmeans.fit_predict(features.values)
            
            # Map cluster IDs to interpretable names based on centroids
            centroids = kmeans.cluster_centers_
            
            # Simple heuristic: map clusters to regimes based on feature values
            regime_names = []
            for i in range(k):
                centroid = centroids[i]
                
                # Get feature positions
                feat_cols = list(features.columns)
                dgs10_idx = feat_cols.index("dgs10_6m_chg") if "dgs10_6m_chg" in feat_cols else -1
                cpi_idx = feat_cols.index("cpi_yoy") if "cpi_yoy" in feat_cols else -1
                unrate_idx = feat_cols.index("unrate_6m_chg") if "unrate_6m_chg" in feat_cols else -1
                
                dgs10_val = centroid[dgs10_idx] if dgs10_idx >= 0 else 0
                cpi_val = centroid[cpi_idx] if cpi_idx >= 0 else 0
                unrate_val = centroid[unrate_idx] if unrate_idx >= 0 else 0
                
                # Assign regime name
                if unrate_val > 0.3:
                    regime_names.append("Recessionary")
                elif dgs10_val > 0.3 and cpi_val > 0.3:
                    regime_names.append("Tightening")
                elif cpi_val < -0.3:
                    regime_names.append("Disinflation")
                else:
                    regime_names.append("Risk-on")
            
            # Map cluster IDs to names
            labels = pd.Series([regime_names[cid] for cid in cluster_ids], index=features.index)
            
        except ImportError:
            # Fallback to rule-based if sklearn not available
            return label_regimes(features, method="rule_based", k=k)
    
    return labels


def regime_performance(
    returns_by_portfolio: dict[str, pd.Series],
    regime_labels: pd.Series
) -> pd.DataFrame:
    """
    Compute performance statistics per regime for each portfolio candidate.
    
    Args:
        returns_by_portfolio: Dict mapping portfolio name to daily returns Series
        regime_labels: Series with regime labels (from label_regimes)
    
    Returns:
        DataFrame with rows=portfolios, columns=[regime, CAGR, Sharpe, Count]
        Multi-index: (portfolio_name, regime)
    """
    from core.utils.metrics import annualized_metrics
    
    results = []
    
    for port_name, port_returns in returns_by_portfolio.items():
        # Align returns with regime labels
        aligned = pd.DataFrame({
            "returns": port_returns,
            "regime": regime_labels
        }).dropna()
        
        if aligned.empty:
            continue
        
        # Compute metrics per regime
        for regime in aligned["regime"].unique():
            regime_rets = aligned[aligned["regime"] == regime]["returns"]
            
            if len(regime_rets) < 20:  # Skip if too few observations
                continue
            
            metrics = annualized_metrics(regime_rets)
            
            results.append({
                "Portfolio": port_name,
                "Regime": regime,
                "CAGR": metrics["CAGR"],
                "Sharpe": metrics["Sharpe"],
                "N": metrics["N"],
            })
    
    if not results:
        return pd.DataFrame()
    
    df = pd.DataFrame(results)
    return df.set_index(["Portfolio", "Regime"])


def current_regime(
    features: Optional[pd.DataFrame] = None,
    labels: Optional[pd.Series] = None,
    lookback_days: int = 30
) -> str:
    """
    Determine the current market regime.
    
    Args:
        features: Regime features DataFrame (if None, will compute)
        labels: Regime labels Series (if None, will compute)
        lookback_days: Number of recent days to use for mode (default: 30)
    
    Returns:
        Most common regime label in the recent period (mode)
    """
    if features is None:
        features = regime_features()
    
    if labels is None:
        labels = label_regimes(features)
    
    if labels.empty:
        return "Unknown"
    
    # Get most recent labels
    recent = labels.tail(lookback_days)
    
    if len(recent) == 0:
        return "Unknown"
    
    # Return mode (most common)
    return recent.mode().iloc[0] if len(recent.mode()) > 0 else recent.iloc[-1]


__all__ = [
    "load_macro_data",
    "regime_features",
    "label_regimes",
    "regime_performance",
    "current_regime",
]
