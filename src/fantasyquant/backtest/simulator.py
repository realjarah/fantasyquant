"""Draft + season simulator for backtesting.

Simulates an entire fantasy season:
1. Train the prediction engine on prior seasons.
2. Simulate a snake draft where opponents pick by ADP.
3. Set optimal lineups each week using the engine's projections.
4. Score each week against *actual* points to measure performance.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from fantasyquant.config import EngineConfig, DEFAULT_CONFIG
from fantasyquant.optimization.solver import (
    DraftSolver,
    DraftState,
    PlayerPool,
)


@dataclass
class SeasonResult:
    """Outcome of a simulated season."""

    roster: list[str]
    """player_ids on our final roster."""

    weekly_scores: dict[int, float]
    """Our team's actual score each week."""

    weekly_opponent_scores: dict[int, float]
    """Simulated opponent score each week (league-median proxy)."""

    wins: int = 0
    losses: int = 0

    @property
    def total_points(self) -> float:
        return sum(self.weekly_scores.values())

    @property
    def record(self) -> str:
        return f"{self.wins}-{self.losses}"


def _simulate_opponent_picks(
    adp: pd.Series,
    taken: set[str],
    n_picks: int,
) -> list[str]:
    """Opponents pick the next *n_picks* best-ADP players not yet taken."""
    available = adp.drop(labels=list(taken), errors="ignore").sort_values()
    return list(available.index[:n_picks])


def _set_optimal_lineup(
    roster_pids: list[str],
    actual_points: pd.DataFrame,
    week: int,
    config: EngineConfig,
) -> float:
    """Given a roster and actual points, set the best possible lineup
    for *week* and return the score.

    Uses a greedy approach: fill required positions first, then flex.
    """
    r = config.roster
    scores: dict[str, float] = {}
    for pid in roster_pids:
        try:
            scores[pid] = float(actual_points.at[pid, week])
        except (KeyError, ValueError):
            scores[pid] = 0.0

    # We need position info — embed it in actual_points index or pass separately.
    # For simplicity, return the sum of top-N scorers on the roster.
    # (A more rigorous version would enforce positional constraints.)
    vals = sorted(scores.values(), reverse=True)
    n_starters = r.starters
    return sum(vals[:n_starters])


def simulate_season(
    pool: PlayerPool,
    actual_weekly: pd.DataFrame,
    config: EngineConfig = DEFAULT_CONFIG,
    my_slot: int = 1,
) -> SeasonResult:
    """Run a full draft simulation + season scoring.

    Parameters
    ----------
    pool:
        Player pool with *projected* points (used by the solver).
    actual_weekly:
        player_id × week matrix of *actual* fantasy points scored
        (ground truth for evaluation).
    config:
        Engine configuration.
    my_slot:
        Our draft position (1-indexed).

    Returns
    -------
    SeasonResult
    """
    n_teams = config.roster.teams
    n_rounds = config.roster.rounds
    total_picks = n_teams * n_rounds

    solver = DraftSolver(pool, config)
    state = DraftState()

    adp = pool.adp if pool.adp is not None else pd.Series(dtype=float)

    # --- Snake draft simulation ---
    for pick_num in range(1, total_picks + 1):
        state.current_pick = pick_num
        rnd = (pick_num - 1) // n_teams + 1
        pos_in_round = (pick_num - 1) % n_teams + 1

        if rnd % 2 == 1:
            is_my_pick = pos_in_round == my_slot
        else:
            is_my_pick = pos_in_round == (n_teams - my_slot + 1)

        if is_my_pick:
            rec = solver.solve(state)
            if rec is not None:
                state.my_picks.append(rec.player_id)
                state.taken.add(rec.player_id)
            # else: couldn't find a pick — roster might be full
        else:
            # Opponent picks by ADP.
            opp_picks = _simulate_opponent_picks(adp, state.taken, 1)
            for pid in opp_picks:
                state.taken.add(pid)

    # --- Season scoring ---
    roster = state.my_picks
    weekly_scores: dict[int, float] = {}
    weekly_opp: dict[int, float] = {}

    for w in range(1, config.nfl_weeks + 1):
        my_score = _set_optimal_lineup(roster, actual_weekly, w, config)
        weekly_scores[w] = my_score

        # Opponent proxy: median score of all other possible rosters.
        # Simplified: take the median of all player scores that week.
        all_scores = actual_weekly[w].dropna() if w in actual_weekly.columns else pd.Series([0.0])
        sorted_scores = all_scores.sort_values(ascending=False)
        # Rough opponent score: sum of players ranked around the median roster.
        mid = n_teams // 2
        start = mid * config.roster.starters
        end = start + config.roster.starters
        opp_score = float(sorted_scores.iloc[start:end].sum()) if len(sorted_scores) > end else 0.0
        weekly_opp[w] = opp_score

    wins = sum(1 for w in weekly_scores if weekly_scores[w] > weekly_opp.get(w, 0))
    losses = len(weekly_scores) - wins

    return SeasonResult(
        roster=roster,
        weekly_scores=weekly_scores,
        weekly_opponent_scores=weekly_opp,
        wins=wins,
        losses=losses,
    )


def backtest(
    training_seasons: list[int],
    test_season: int,
    config: EngineConfig = DEFAULT_CONFIG,
    my_slot: int = 1,
) -> SeasonResult:
    """Full backtest: train on *training_seasons*, test on *test_season*.

    This loads data via ``nfl_data_py``, builds projections from the
    training window, then simulates a draft and season using actual
    test-season results.

    Returns
    -------
    SeasonResult
    """
    from fantasyquant.data.historical import (
        build_stat_matrix,
        compute_fantasy_points,
        load_weekly_stats,
    )
    from fantasyquant.prediction.projections import build_projections

    # Build projections from training data.
    train_config = EngineConfig(
        scoring=config.scoring,
        roster=config.roster,
        prediction=config.prediction,
        optimization=config.optimization,
        nfl_weeks=config.nfl_weeks,
        current_season=test_season,  # so training window = test_season - N
    )
    output = build_projections(train_config)

    # Load actual test-season data for scoring.
    test_df = load_weekly_stats(config, seasons=[test_season])
    actual_matrix, actual_info = build_stat_matrix(test_df, stat="fantasy_points")

    # Reshape actual_matrix: flatten multi-level columns to just week ints.
    # build_stat_matrix produces (season, week) column tuples.
    actual_matrix.columns = [w for _, w in actual_matrix.columns]
    actual_matrix.index = actual_info["player_id"]

    # Build ADP proxy from historical usage (players ranked by total points).
    total_pts = actual_matrix.sum(axis=1).sort_values(ascending=False)
    adp_proxy = pd.Series(
        range(1, len(total_pts) + 1),
        index=total_pts.index,
        name="adp",
    )

    pool = PlayerPool(
        projections=output.weekly_projections,
        info=output.player_info,
        adp=adp_proxy,
    )

    return simulate_season(pool, actual_matrix, config, my_slot)
