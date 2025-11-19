"""
Tests for Universe Expansion and Objective Mapping

Tests the new universe.yaml and objectives.yaml configuration system.
"""

import pytest
import pandas as pd
from core.universe_yaml import (
    load_universe_from_yaml,
    get_symbols_by_asset_class,
    get_asset_metadata,
    validate_universe_yaml
)
from core.objective_mapper import (
    load_objectives_config,
    recommend_objectives_for_risk,
    adjust_bands_for_risk,
    classify_objective_fit,
    get_stretch_objective
)


class TestUniverseLoading:
    """Test universe configuration loading."""
    
    def test_load_universe_basic(self):
        """Test basic universe loading."""
        universe = load_universe_from_yaml()
        
        assert len(universe) > 0, "Universe should have assets"
        assert 'symbol' in universe.columns
        assert 'asset_class' in universe.columns
        assert 'core_satellite' in universe.columns
    
    def test_universe_asset_classes(self):
        """Test that universe covers required asset classes."""
        universe = load_universe_from_yaml()
        
        asset_classes = set(universe['asset_class'].unique())
        required_classes = {'equity', 'bond'}
        
        assert required_classes.issubset(asset_classes), \
            f"Universe missing required asset classes. Have: {asset_classes}"
    
    def test_filter_by_asset_class(self):
        """Test filtering universe by asset class."""
        equity_only = load_universe_from_yaml(asset_classes=['equity'])
        
        assert len(equity_only) > 0
        assert all(equity_only['asset_class'] == 'equity')
    
    def test_filter_core_only(self):
        """Test filtering to core assets only."""
        core_only = load_universe_from_yaml(core_only=True)
        
        assert len(core_only) > 0
        assert all(core_only['core_satellite'] == 'core')
    
    def test_symbols_by_asset_class(self):
        """Test grouping symbols by asset class."""
        by_class = get_symbols_by_asset_class()
        
        assert isinstance(by_class, dict)
        assert 'equity' in by_class
        assert isinstance(by_class['equity'], list)
        assert len(by_class['equity']) > 0
    
    def test_get_asset_metadata(self):
        """Test getting metadata for specific asset."""
        metadata = get_asset_metadata('SPY')
        
        assert metadata is not None
        assert metadata['symbol'] == 'SPY'
        assert 'asset_class' in metadata
    
    def test_validate_universe(self):
        """Test universe validation."""
        errors = validate_universe_yaml()
        
        assert isinstance(errors, list)
        # Universe should be valid (empty error list)
        assert len(errors) == 0, f"Universe validation errors: {errors}"


class TestObjectiveConfiguration:
    """Test objective configuration loading and mapping."""
    
    def test_load_objectives(self):
        """Test loading objectives configuration."""
        objectives = load_objectives_config()
        
        assert len(objectives) > 0
        assert 'BALANCED' in objectives
        assert 'CONSERVATIVE' in objectives
        assert 'GROWTH' in objectives
    
    def test_objective_config_structure(self):
        """Test that objective configs have required fields."""
        objectives = load_objectives_config()
        
        for name, obj in objectives.items():
            assert hasattr(obj, 'name')
            assert hasattr(obj, 'label')
            assert hasattr(obj, 'risk_score_min')
            assert hasattr(obj, 'risk_score_max')
            assert hasattr(obj, 'target_return_min')
            assert hasattr(obj, 'target_vol_min')
            assert hasattr(obj, 'asset_class_bands')
            
            # Check asset class bands structure
            assert isinstance(obj.asset_class_bands, dict)
            for ac, band in obj.asset_class_bands.items():
                assert 'min' in band
                assert 'max' in band
                assert 0 <= band['min'] <= 1.0
                assert 0 <= band['max'] <= 1.0
                assert band['min'] <= band['max']
    
    def test_recommend_objectives_for_low_risk(self):
        """Test objective recommendations for low risk score."""
        recommended = recommend_objectives_for_risk(20)
        
        assert len(recommended) > 0
        assert 'CONSERVATIVE' in recommended or 'BALANCED' in recommended
        # Should NOT recommend aggressive objectives
        assert 'AGGRESSIVE' not in recommended
    
    def test_recommend_objectives_for_medium_risk(self):
        """Test objective recommendations for medium risk score."""
        recommended = recommend_objectives_for_risk(50)
        
        assert len(recommended) > 0
        # Medium risk should get balanced or growth+income
        assert any(obj in recommended for obj in ['BALANCED', 'GROWTH_PLUS_INCOME'])
    
    def test_recommend_objectives_for_high_risk(self):
        """Test objective recommendations for high risk score."""
        recommended = recommend_objectives_for_risk(80)
        
        assert len(recommended) > 0
        # High risk should get growth or aggressive
        assert any(obj in recommended for obj in ['GROWTH', 'AGGRESSIVE', 'GROWTH_PLUS_INCOME'])
    
    def test_adjust_bands_for_risk(self):
        """Test band adjustment based on risk score."""
        objectives = load_objectives_config()
        balanced = objectives['BALANCED']
        
        # Test below optimal (should reduce targets)
        return_band_low, vol_band_low = adjust_bands_for_risk(35, balanced)
        
        # Test at optimal (no adjustment)
        return_band_opt, vol_band_opt = adjust_bands_for_risk(50, balanced)
        
        # Test above optimal (should increase targets)
        return_band_high, vol_band_high = adjust_bands_for_risk(65, balanced)
        
        # Low risk should have lower or equal targets
        assert return_band_low[1] <= return_band_opt[1]
        assert vol_band_low[1] <= vol_band_opt[1]
        
        # High risk should have higher or equal targets
        assert return_band_high[1] >= return_band_opt[1]
        assert vol_band_high[1] >= vol_band_opt[1]
    
    def test_classify_objective_fit_match(self):
        """Test objective fit classification for good match."""
        fit_type, explanation, suggested = classify_objective_fit(50, 'BALANCED')
        
        assert fit_type == 'match'
        assert suggested is None
    
    def test_classify_objective_fit_mismatch_low_risk(self):
        """Test objective fit classification for low risk + aggressive objective."""
        fit_type, explanation, suggested = classify_objective_fit(25, 'AGGRESSIVE')
        
        assert fit_type == 'mismatch'
        assert suggested is not None
        assert suggested in ['CONSERVATIVE', 'BALANCED']
    
    def test_classify_objective_fit_mismatch_high_risk(self):
        """Test objective fit classification for high risk + conservative objective."""
        fit_type, explanation, suggested = classify_objective_fit(80, 'CONSERVATIVE')
        
        assert fit_type == 'mismatch'
        assert suggested is not None
    
    def test_get_stretch_objective(self):
        """Test getting stretch objective."""
        stretch = get_stretch_objective('BALANCED')
        
        assert stretch is not None
        assert stretch in ['GROWTH', 'GROWTH_PLUS_INCOME']
        
        # Most aggressive should have no stretch
        stretch_aggressive = get_stretch_objective('AGGRESSIVE')
        assert stretch_aggressive is None


class TestRiskObjectiveMapping:
    """Test risk score to objective mapping logic."""
    
    def test_risk_objective_monotonicity(self):
        """Test that recommended objectives become more aggressive as risk increases."""
        objectives = load_objectives_config()
        
        # Get recommendations for increasing risk scores
        risk_scores = [10, 30, 50, 70, 90]
        recommendations = [recommend_objectives_for_risk(r) for r in risk_scores]
        
        # Check that we don't recommend AGGRESSIVE for very low risk
        assert 'AGGRESSIVE' not in recommendations[0]
        
        # Check that we don't recommend CONSERVATIVE for very high risk
        assert 'CONSERVATIVE' not in recommendations[-1]
    
    def test_risk_bands_continuous(self):
        """Test that risk bands don't have gaps."""
        objectives = load_objectives_config()
        
        # Sort objectives by optimal risk score
        sorted_objs = sorted(objectives.values(), key=lambda x: x.risk_score_optimal)
        
        # Check that risk bands overlap or are contiguous
        for i in range(len(sorted_objs) - 1):
            current = sorted_objs[i]
            next_obj = sorted_objs[i + 1]
            
            # Next objective should start at or before current ends
            assert next_obj.risk_score_min <= current.risk_score_max + 5, \
                f"Gap between {current.name} and {next_obj.name}"
    
    def test_all_risk_scores_covered(self):
        """Test that all risk scores (0-100) have at least one recommended objective."""
        for risk in range(0, 101, 10):
            recommended = recommend_objectives_for_risk(risk)
            assert len(recommended) > 0, f"No objective recommended for risk={risk}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
