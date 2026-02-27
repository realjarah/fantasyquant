"""Tests for the blending and projection utilities."""

import numpy as np
import pandas as pd
import pytest

from fantasyquant.config import PredictionConfig
from fantasyquant.prediction.blender import (
    blend_multipliers,
    project_raw_weekly,
    volume_anchor,
)


class TestBlendMultipliers:
    def test_equal_weight(self):
        h = pd.Series({"A": 1.2, "B": 0.8}, name="w_hist")
        v = pd.Series({"A": 1.0, "B": 1.0}, name="w_vegas")
        result = blend_multipliers(h, v)
        assert result["A"] == pytest.approx(1.1)
        assert result["B"] == pytest.approx(0.9)

    def test_missing_in_one_source(self):
        h = pd.Series({"A": 1.2}, name="w_hist")
        v = pd.Series({"A": 1.0, "C": 0.9}, name="w_vegas")
        result = blend_multipliers(h, v)
        assert result["C"] == pytest.approx(0.9)  # only in vegas

    def test_custom_weights(self):
        h = pd.Series({"X": 1.0}, name="w_hist")
        v = pd.Series({"X": 2.0}, name="w_vegas")
        cfg = PredictionConfig(historical_weight=0.3, vegas_weight=0.7)
        result = blend_multipliers(h, v, cfg)
        assert result["X"] == pytest.approx(0.3 * 1.0 + 0.7 * 2.0)


class TestProjectRawWeekly:
    def test_basic_projection(self):
        skill = pd.Series({"P1": 10.0, "P2": 20.0})
        w_final = pd.Series({"A": 1.0, "B": 1.5})
        matchup_grid = {
            "T1": {1: "A", 2: "B"},
            "T2": {1: "B", 2: "A"},
        }
        player_teams = pd.Series({"P1": "T1", "P2": "T2"})

        result = project_raw_weekly(skill, w_final, matchup_grid, player_teams, weeks=2)

        assert result.at["P1", 1] == pytest.approx(10.0)   # 10 × 1.0
        assert result.at["P1", 2] == pytest.approx(15.0)   # 10 × 1.5
        assert result.at["P2", 1] == pytest.approx(30.0)   # 20 × 1.5
        assert result.at["P2", 2] == pytest.approx(20.0)   # 20 × 1.0

    def test_bye_week_is_zero(self):
        skill = pd.Series({"P1": 10.0})
        w_final = pd.Series({"A": 1.2})
        matchup_grid = {"T1": {1: "A"}}  # week 2 is bye
        player_teams = pd.Series({"P1": "T1"})

        result = project_raw_weekly(skill, w_final, matchup_grid, player_teams, weeks=2)
        assert result.at["P1", 2] == pytest.approx(0.0)


class TestVolumeAnchor:
    def test_scales_to_target(self):
        raw = pd.DataFrame(
            {"1": [10.0, 5.0], "2": [10.0, 5.0]},
            index=["P1", "P2"],
        )
        raw.columns = [1, 2]
        props = {"P1": 30.0}  # raw total = 20, so scale by 1.5

        scaled = volume_anchor(raw, props)
        assert scaled.loc["P1"].sum() == pytest.approx(30.0)
        # P2 has no prop → unchanged
        assert scaled.loc["P2"].sum() == pytest.approx(10.0)

    def test_no_props_leaves_unchanged(self):
        raw = pd.DataFrame({1: [10.0], 2: [10.0]}, index=["P1"])
        scaled = volume_anchor(raw, {})
        assert scaled.loc["P1"].sum() == pytest.approx(20.0)
