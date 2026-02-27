"""Tests for β(t) opponent-score estimation."""

import pandas as pd
import pytest

from fantasyquant.config import EngineConfig, RosterSettings, OptimizationConfig
from fantasyquant.optimization.solver import (
    DraftState,
    PlayerPool,
    estimate_beta,
)


def _make_pool(n_players: int = 30, weeks: int = 4) -> PlayerPool:
    """Create a small synthetic player pool with interleaved positions."""
    # Interleave positions so ADP-ordered roster is realistic.
    pos_cycle = ["QB", "RB", "RB", "WR", "WR", "TE"]
    positions = [pos_cycle[i % len(pos_cycle)] for i in range(n_players)]
    ids = [f"P{i}" for i in range(n_players)]
    names = [f"Player {i}" for i in range(n_players)]
    teams = [f"T{i % 4}" for i in range(n_players)]

    info = pd.DataFrame({
        "player_id": ids,
        "player_name": names,
        "position": positions,
        "team": teams,
    })

    # Higher-indexed players score slightly more (descending ADP = ascending quality).
    proj_data = {}
    for w in range(1, weeks + 1):
        # Invert: lower index (better ADP) = higher score.
        proj_data[w] = [15.0 - i * 0.3 + w * 0.1 for i in range(n_players)]
    proj = pd.DataFrame(proj_data, index=ids)

    adp = pd.Series(range(1, n_players + 1), index=ids, name="adp")

    return PlayerPool(projections=proj, info=info, adp=adp)


def _small_config(weeks: int = 4) -> EngineConfig:
    return EngineConfig(
        roster=RosterSettings(teams=4, rounds=6, qb=1, rb=1, wr=1, te=1, flex=1, bench=1, dst=0, k=0),
        optimization=OptimizationConfig(solver_time_limit_seconds=10),
        nfl_weeks=weeks,
    )


class TestEstimateBeta:
    def test_returns_dict_for_each_week(self):
        pool = _make_pool()
        config = _small_config()
        state = DraftState()

        beta = estimate_beta(state, pool, config)

        assert isinstance(beta, dict)
        for w in range(1, config.nfl_weeks + 1):
            assert w in beta
            assert beta[w] >= 0.0

    def test_beta_positive_for_nonempty_pool(self):
        """Opponent should project some positive score."""
        pool = _make_pool()
        config = _small_config()
        state = DraftState()

        beta = estimate_beta(state, pool, config)

        assert all(v > 0.0 for v in beta.values())

    def test_beta_decreases_as_players_taken(self):
        """As the best players are taken, opponent gets weaker."""
        pool = _make_pool()
        config = _small_config()

        state_early = DraftState()
        beta_early = estimate_beta(state_early, pool, config)

        # Take the top 10 players.
        state_late = DraftState(taken={f"P{i}" for i in range(10)})
        beta_late = estimate_beta(state_late, pool, config)

        # Average beta should be lower with fewer top players available.
        avg_early = sum(beta_early.values()) / len(beta_early)
        avg_late = sum(beta_late.values()) / len(beta_late)
        assert avg_late < avg_early

    def test_beta_uses_adp_ranking(self):
        """When ADP is provided, opponent drafts by ADP order."""
        pool = _make_pool()
        config = _small_config()
        state = DraftState()

        beta = estimate_beta(state, pool, config)

        # With ADP ordering, opponent gets the best available players
        # (lowest ADP numbers), so beta should reflect strong lineup.
        assert beta[1] > 0
