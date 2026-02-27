"""Tests for the Alternating Minimization decomposition."""

import numpy as np
import pandas as pd
import pytest

from fantasyquant.config import PredictionConfig
from fantasyquant.prediction.alternating_min import AlternatingMinimization


def _make_synthetic_data(
    n_players: int = 20,
    n_weeks: int = 10,
    n_teams: int = 6,
    seed: int = 42,
):
    """Create synthetic stats = skill × defense_mult + noise."""
    rng = np.random.default_rng(seed)

    skills = rng.uniform(5, 25, size=n_players)
    defense_mults = rng.uniform(0.7, 1.3, size=n_teams)
    team_names = [f"T{i}" for i in range(n_teams)]

    stats = np.zeros((n_players, n_weeks))
    opponents = pd.DataFrame(index=range(n_players), columns=range(n_weeks))

    for i in range(n_players):
        for j in range(n_weeks):
            d = rng.integers(0, n_teams)
            opponents.iat[i, j] = team_names[d]
            stats[i, j] = skills[i] * defense_mults[d] + rng.normal(0, 1)

    stats = np.maximum(stats, 0)  # no negative fantasy points
    stats_df = pd.DataFrame(stats)
    player_ids = pd.Series([f"P{i}" for i in range(n_players)])
    teams = pd.Series([team_names[i % n_teams] for i in range(n_players)])

    return stats_df, opponents, player_ids, teams, skills, defense_mults, team_names


class TestAlternatingMinimization:
    def test_converges(self):
        stats_df, opponents, pids, teams, _, _, _ = _make_synthetic_data()
        am = AlternatingMinimization(PredictionConfig(alt_min_max_iterations=100))
        result = am.fit(stats_df, opponents, pids, teams)

        assert result.iterations <= 100
        assert result.residual < np.inf

    def test_recovers_skill_ordering(self):
        """The decomposition should roughly preserve the ordering of
        player skills (highest-skilled player should have highest u)."""
        stats_df, opponents, pids, teams, true_skills, _, _ = _make_synthetic_data()
        am = AlternatingMinimization(PredictionConfig(alt_min_max_iterations=100))
        result = am.fit(stats_df, opponents, pids, teams)

        # Rank correlation between true and estimated skills.
        estimated = result.player_skill.values
        true_order = np.argsort(-true_skills)
        est_order = np.argsort(-estimated)

        # Top 5 players should overlap significantly.
        top5_true = set(true_order[:5])
        top5_est = set(est_order[:5])
        overlap = len(top5_true & top5_est)
        assert overlap >= 3, f"Only {overlap}/5 top players matched"

    def test_defense_multipliers_centered(self):
        """Defense multipliers should be roughly centred around 1.0."""
        stats_df, opponents, pids, teams, _, _, _ = _make_synthetic_data()
        am = AlternatingMinimization()
        result = am.fit(stats_df, opponents, pids, teams)

        mean_w = result.defense_multipliers.mean()
        assert 0.7 < mean_w < 1.3, f"Mean defense mult = {mean_w}"

    def test_empty_input(self):
        """Gracefully handle an empty stats matrix."""
        stats_df = pd.DataFrame()
        opponents = pd.DataFrame()
        pids = pd.Series(dtype=str)
        teams = pd.Series(dtype=str)
        am = AlternatingMinimization()
        result = am.fit(stats_df, opponents, pids, teams)

        assert len(result.player_skill) == 0
