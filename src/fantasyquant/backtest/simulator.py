"""Draft + season simulator for backtesting.

Simulates an entire fantasy season per Becker & Sun (2013) Section 6:
1. Train the prediction engine on prior seasons.
2. Simulate a snake draft where opponents pick by ADP.
3. Set optimal lineups each week using position-constrained optimizer (Section 4.4).
4. Score each week against *actual* points to measure performance.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from fantasyquant.config import EngineConfig, DEFAULT_CONFIG
from fantasyquant.optimization.lineup import set_lineup
from fantasyquant.optimization.solver import (
    DraftSolver,
    DraftState,
    PlayerPool,
    estimate_beta,
)


@dataclass
class SeasonResult:
    """Outcome of a simulated season."""

    roster: list[str]
    """player_ids on our final roster."""

    weekly_scores: dict[int, float]
    """Our team's actual score each week."""

    weekly_opponent_scores: dict[int, float]
    """Simulated opponent score each week."""

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


def simulate_season(
    pool: PlayerPool,
    actual_weekly: pd.DataFrame,
    config: EngineConfig = DEFAULT_CONFIG,
    my_slot: int = 1,
    position_lookup: dict[str, str] | None = None,
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
    position_lookup:
        {player_id: position} mapping. If None, built from pool.info.

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

    # Build position lookup.
    if position_lookup is None:
        position_lookup = dict(
            zip(pool.info["player_id"], pool.info["position"]),
        )

    # --- Snake draft simulation ---
    beta_cache: dict[int, float] | None = None

    for pick_num in range(1, total_picks + 1):
        state.current_pick = pick_num
        rnd = (pick_num - 1) // n_teams + 1
        pos_in_round = (pick_num - 1) % n_teams + 1

        if rnd % 2 == 1:
            is_my_pick = pos_in_round == my_slot
        else:
            is_my_pick = pos_in_round == (n_teams - my_slot + 1)

        if is_my_pick:
            # Re-estimate β(t) at each of our picks (per paper).
            beta_cache = estimate_beta(state, pool, config)
            rec = solver.solve(state, beta=beta_cache)
            if rec is not None:
                state.my_picks.append(rec.player_id)
                state.taken.add(rec.player_id)
        else:
            # Opponent picks by ADP.
            opp_picks = _simulate_opponent_picks(adp, state.taken, 1)
            for pid in opp_picks:
                state.taken.add(pid)

    # --- Season scoring with position-constrained lineups (Section 4.4) ---
    roster = state.my_picks
    weekly_scores: dict[int, float] = {}
    weekly_opp: dict[int, float] = {}

    for w in range(1, config.nfl_weeks + 1):
        # Our score: position-constrained optimal lineup using actuals.
        my_scores = {}
        for pid in roster:
            try:
                my_scores[pid] = float(actual_weekly.at[pid, w])
            except (KeyError, ValueError):
                my_scores[pid] = 0.0

        _, my_total = set_lineup(roster, my_scores, position_lookup, config)
        weekly_scores[w] = my_total

        # Opponent proxy: build a "median team" from non-roster players,
        # set their optimal lineup, use that as the opponent score.
        all_player_scores = (
            actual_weekly[w].dropna()
            if w in actual_weekly.columns
            else pd.Series([0.0])
        )
        non_roster = [
            pid for pid in all_player_scores.index if pid not in set(roster)
        ]
        # Sort by score and take the "5th team" worth of players as opponent.
        sorted_non_roster = sorted(
            non_roster, key=lambda p: all_player_scores.get(p, 0.0), reverse=True,
        )
        mid = n_teams // 2
        start = mid * config.roster.starters
        end = start + config.roster.starters
        opp_roster_slice = sorted_non_roster[start:end] if len(sorted_non_roster) > end else sorted_non_roster[:config.roster.starters]

        opp_scores_map = {
            pid: float(all_player_scores.get(pid, 0.0))
            for pid in opp_roster_slice
        }
        opp_positions = {pid: position_lookup.get(pid, "RB") for pid in opp_roster_slice}
        _, opp_total = set_lineup(
            opp_roster_slice, opp_scores_map, opp_positions, config,
        )
        weekly_opp[w] = opp_total

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
    actual_matrix.columns = [w for _, w in actual_matrix.columns]
    actual_matrix.index = actual_info["player_id"]

    # Build position lookup from actual info.
    position_lookup = dict(
        zip(actual_info["player_id"], actual_info["position"]),
    )
    # Also add from projections info for players only in training data.
    for _, row in output.player_info.iterrows():
        pid = row["player_id"]
        if pid not in position_lookup:
            position_lookup[pid] = row["position"]

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

    return simulate_season(
        pool, actual_matrix, config, my_slot,
        position_lookup=position_lookup,
    )
