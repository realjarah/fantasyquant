"""Tests for the MIP draft solver."""

import pandas as pd
import pytest

from fantasyquant.config import EngineConfig, RosterSettings, OptimizationConfig
from fantasyquant.optimization.solver import (
    DraftSolver,
    DraftState,
    PlayerPool,
)


def _make_pool(n_players: int = 30, weeks: int = 4) -> PlayerPool:
    """Create a small synthetic player pool for testing."""
    positions = (["QB"] * 5 + ["RB"] * 10 + ["WR"] * 10 + ["TE"] * 5)[:n_players]
    ids = [f"P{i}" for i in range(n_players)]
    names = [f"Player {i}" for i in range(n_players)]
    teams = [f"T{i % 4}" for i in range(n_players)]

    info = pd.DataFrame({
        "player_id": ids,
        "player_name": names,
        "position": positions,
        "team": teams,
    })

    # Projections: higher-indexed players are slightly better.
    proj_data = {}
    for w in range(1, weeks + 1):
        proj_data[w] = [5.0 + i * 0.5 + w * 0.1 for i in range(n_players)]
    proj = pd.DataFrame(proj_data, index=ids)
    proj.columns.name = "week"

    adp = pd.Series(range(1, n_players + 1), index=ids, name="adp")

    return PlayerPool(projections=proj, info=info, adp=adp)


def _small_config(weeks: int = 4) -> EngineConfig:
    return EngineConfig(
        roster=RosterSettings(teams=4, rounds=6, qb=1, rb=1, wr=1, te=1, flex=1, bench=1, dst=0, k=0),
        optimization=OptimizationConfig(solver_time_limit_seconds=10),
        nfl_weeks=weeks,
    )


class TestDraftSolver:
    def test_returns_recommendation(self):
        pool = _make_pool()
        config = _small_config()
        solver = DraftSolver(pool, config)
        state = DraftState()

        rec = solver.solve(state)
        assert rec is not None
        assert rec.player_id in pool.projections.index
        assert rec.position in ("QB", "RB", "WR", "TE")

    def test_does_not_pick_taken_player(self):
        pool = _make_pool()
        config = _small_config()
        solver = DraftSolver(pool, config)
        state = DraftState(taken={"P29"})  # best player taken

        rec = solver.solve(state)
        assert rec is not None
        assert rec.player_id != "P29"

    def test_builds_roster_incrementally(self):
        pool = _make_pool()
        config = _small_config()
        solver = DraftSolver(pool, config)
        state = DraftState()

        picks = []
        for _ in range(3):
            rec = solver.solve(state)
            assert rec is not None
            state.my_picks.append(rec.player_id)
            state.taken.add(rec.player_id)
            state.current_pick += 1
            picks.append(rec.player_id)

        # All picks should be unique.
        assert len(set(picks)) == 3

    def test_respects_positional_limits(self):
        """After filling all QB slots, solver should not pick another QB."""
        pool = _make_pool(n_players=30, weeks=4)
        config = _small_config()  # qb=1, bench=1 → max 2 QBs
        solver = DraftSolver(pool, config)

        # Pre-fill with 2 QBs (max allowed).
        state = DraftState(
            my_picks=["P0", "P1"],  # both QBs
            taken={"P0", "P1"},
            current_pick=3,
        )

        rec = solver.solve(state)
        assert rec is not None
        # Should not pick another QB.
        assert rec.position != "QB", f"Picked QB {rec.player_name} when QB slots full"
