"""Tests for Vegas multiplier calculations."""

import pytest

from fantasyquant.prediction.vegas_multipliers import (
    compute_vegas_multipliers,
    implied_pa_to_multiplier,
    power_ratings_to_implied_pa,
    win_totals_to_power_ratings,
)


class TestWinTotalsToPowerRatings:
    def test_baseline_team(self):
        """An 8.5-win team should have a power rating of 0."""
        pr = win_totals_to_power_ratings({"AVG": 8.5})
        assert pr["AVG"] == pytest.approx(0.0)

    def test_good_team(self):
        pr = win_totals_to_power_ratings({"KC": 11.5})
        assert pr["KC"] == pytest.approx(5.25)

    def test_bad_team(self):
        pr = win_totals_to_power_ratings({"NE": 4.5})
        assert pr["NE"] == pytest.approx(-7.0)


class TestPowerRatingsToImpliedPA:
    def test_neutral(self):
        pa = power_ratings_to_implied_pa({"AVG": 0.0})
        assert pa["AVG"] == pytest.approx(22.5)

    def test_good_defense(self):
        pa = power_ratings_to_implied_pa({"KC": 5.25})
        assert pa["KC"] == pytest.approx(17.25)

    def test_bad_defense(self):
        pa = power_ratings_to_implied_pa({"NE": -7.0})
        assert pa["NE"] == pytest.approx(29.5)


class TestImpliedPAToMultiplier:
    def test_neutral(self):
        m = implied_pa_to_multiplier({"AVG": 22.5})
        assert m["AVG"] == pytest.approx(1.0)

    def test_favorable_matchup(self):
        m = implied_pa_to_multiplier({"BAD": 29.5})
        assert m["BAD"] > 1.0

    def test_tough_matchup(self):
        m = implied_pa_to_multiplier({"GOOD": 17.25})
        assert m["GOOD"] < 1.0


class TestEndToEnd:
    def test_compute_vegas_multipliers(self):
        wt = {"KC": 11.5, "NE": 4.5, "AVG": 8.5}
        mults = compute_vegas_multipliers(wt)

        # KC is a good team → good defense → opponents score less → mult < 1
        assert mults["KC"] < 1.0
        # NE is a bad team → bad defense → opponents score more → mult > 1
        assert mults["NE"] > 1.0
        # AVG → 1.0
        assert mults["AVG"] == pytest.approx(1.0)

    def test_all_teams_present(self):
        wt = {"A": 10.0, "B": 7.0, "C": 8.5}
        mults = compute_vegas_multipliers(wt)
        assert set(mults.index) == {"A", "B", "C"}
