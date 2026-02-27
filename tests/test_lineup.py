"""Tests for the position-constrained weekly lineup optimizer."""

import pandas as pd
import pytest

from fantasyquant.config import EngineConfig, RosterSettings
from fantasyquant.optimization.lineup import set_lineup, project_lineup_score


def _roster_config() -> EngineConfig:
    return EngineConfig(
        roster=RosterSettings(teams=10, rounds=15, qb=1, rb=2, wr=2, te=1, flex=1, bench=6, dst=0, k=0),
    )


class TestSetLineup:
    def test_fills_all_starter_slots(self):
        """Should start exactly qb+rb+wr+te+flex players."""
        config = _roster_config()
        roster = [f"P{i}" for i in range(12)]
        positions = {
            "P0": "QB", "P1": "QB",
            "P2": "RB", "P3": "RB", "P4": "RB",
            "P5": "WR", "P6": "WR", "P7": "WR",
            "P8": "TE", "P9": "TE",
            "P10": "RB", "P11": "WR",
        }
        scores = {f"P{i}": 10.0 + i for i in range(12)}

        starters, total = set_lineup(roster, scores, positions, config)

        # 1 QB + 2 RB + 2 WR + 1 TE + 1 FLEX = 7 starters
        assert len(starters) == config.roster.starters

    def test_picks_highest_scorers_per_position(self):
        """Best QB should start, not the worst one."""
        config = _roster_config()
        roster = ["QB1", "QB2", "RB1", "RB2", "RB3", "WR1", "WR2", "WR3", "TE1"]
        positions = {
            "QB1": "QB", "QB2": "QB",
            "RB1": "RB", "RB2": "RB", "RB3": "RB",
            "WR1": "WR", "WR2": "WR", "WR3": "WR",
            "TE1": "TE",
        }
        scores = {
            "QB1": 20.0, "QB2": 5.0,
            "RB1": 15.0, "RB2": 12.0, "RB3": 8.0,
            "WR1": 18.0, "WR2": 14.0, "WR3": 6.0,
            "TE1": 10.0,
        }

        starters, total = set_lineup(roster, scores, positions, config)

        assert "QB1" in starters
        assert "QB2" not in starters
        assert "RB1" in starters
        assert "RB2" in starters
        assert "WR1" in starters
        assert "WR2" in starters
        assert "TE1" in starters

    def test_flex_goes_to_best_remaining(self):
        """FLEX should be filled by the best unused RB/WR/TE."""
        config = _roster_config()
        roster = ["QB1", "RB1", "RB2", "RB3", "WR1", "WR2", "TE1"]
        positions = {
            "QB1": "QB",
            "RB1": "RB", "RB2": "RB", "RB3": "RB",
            "WR1": "WR", "WR2": "WR",
            "TE1": "TE",
        }
        scores = {
            "QB1": 20.0,
            "RB1": 15.0, "RB2": 12.0, "RB3": 25.0,  # RB3 is best
            "WR1": 18.0, "WR2": 14.0,
            "TE1": 10.0,
        }

        starters, total = set_lineup(roster, scores, positions, config)

        # RB3 and RB1 are top 2 RBs → start as RB.
        # RB2 is next best remaining → FLEX.
        # Actually RB3 (25) and RB1 (15) start at RB.
        # FLEX: WR2 (14) vs RB2 (12) → WR2 should be WR starter.
        # Wait, WR1 (18) and WR2 (14) start at WR.  FLEX = RB2 (12) vs TE... only TE1 needed.
        # After required slots: QB1, RB3+RB1, WR1+WR2, TE1 → FLEX = RB2 (12)
        assert "RB2" in starters  # Should be the FLEX

    def test_empty_roster(self):
        """Empty roster produces no starters."""
        config = _roster_config()
        starters, total = set_lineup([], {}, {}, config)
        assert starters == []
        assert total == 0.0

    def test_bye_week_zeros(self):
        """Players with 0 points (bye week) are still eligible but scored 0."""
        config = _roster_config()
        roster = ["QB1", "RB1", "RB2", "WR1", "WR2", "TE1", "RB3"]
        positions = {
            "QB1": "QB", "RB1": "RB", "RB2": "RB", "RB3": "RB",
            "WR1": "WR", "WR2": "WR", "TE1": "TE",
        }
        scores = {
            "QB1": 0.0,  # bye week
            "RB1": 10.0, "RB2": 8.0, "RB3": 5.0,
            "WR1": 12.0, "WR2": 7.0,
            "TE1": 6.0,
        }

        starters, total = set_lineup(roster, scores, positions, config)
        assert "QB1" in starters  # Only QB available
        assert total == pytest.approx(0 + 10 + 8 + 12 + 7 + 6 + 5)


class TestProjectLineupScore:
    def test_uses_projections_matrix(self):
        config = _roster_config()
        proj = pd.DataFrame(
            {1: [20.0, 15.0, 12.0, 18.0, 14.0, 10.0, 8.0],
             2: [22.0, 14.0, 11.0, 19.0, 13.0, 9.0, 7.0]},
            index=["QB1", "RB1", "RB2", "WR1", "WR2", "TE1", "RB3"],
        )
        positions = {
            "QB1": "QB", "RB1": "RB", "RB2": "RB", "RB3": "RB",
            "WR1": "WR", "WR2": "WR", "TE1": "TE",
        }
        roster = list(proj.index)

        score = project_lineup_score(roster, proj, positions, week=1, config=config)
        # Best lineup week 1: QB1(20)+RB1(15)+RB2(12)+WR1(18)+WR2(14)+TE1(10)+FLEX=RB3(8)
        assert score == pytest.approx(97.0)
