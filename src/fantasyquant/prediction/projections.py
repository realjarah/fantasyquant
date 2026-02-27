"""Orchestrate the full prediction pipeline to produce the f(i,t) matrix.

This is the primary entry point that chains data loading → alternating
minimisation → Vegas multipliers → blending → volume anchoring.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from fantasyquant.config import EngineConfig, DEFAULT_CONFIG
from fantasyquant.data.historical import (
    build_opponent_matrix,
    build_stat_matrix,
    load_weekly_stats,
)
from fantasyquant.data.schedule import build_matchup_grid, load_schedule
from fantasyquant.data.vegas import load_player_props, load_win_totals
from fantasyquant.prediction.alternating_min import (
    AlternatingMinimization,
    DecompositionResult,
)
from fantasyquant.prediction.blender import (
    blend_multipliers,
    project_raw_weekly,
    volume_anchor,
)
from fantasyquant.prediction.vegas_multipliers import compute_vegas_multipliers


@dataclass
class ProjectionOutput:
    """Everything produced by the prediction pipeline."""

    weekly_projections: pd.DataFrame
    """f(i,t): player_id × week matrix of projected fantasy points."""

    player_info: pd.DataFrame
    """Metadata for each player (name, position, team)."""

    decomposition: DecompositionResult
    """Raw alternating-minimisation results."""

    w_final: pd.Series
    """Blended defensive multipliers."""


def build_projections(
    config: EngineConfig = DEFAULT_CONFIG,
    *,
    win_totals_source: str | None = None,
    player_props_source: str | None = None,
    odds_api_key: str | None = None,
    historical_df: pd.DataFrame | None = None,
) -> ProjectionOutput:
    """End-to-end projection builder.

    Parameters
    ----------
    config:
        Full engine configuration.
    win_totals_source:
        Path to a JSON/CSV file with team win totals. Falls back to
        built-in defaults if not provided.
    player_props_source:
        Path to a JSON/CSV file with player season total props.
    odds_api_key:
        API key for The Odds API (optional).
    historical_df:
        Pre-loaded weekly stats DataFrame. If *None*, loads via
        ``nfl_data_py``.

    Returns
    -------
    ProjectionOutput
    """
    # ---- 1. Data ingestion ----
    if historical_df is None:
        historical_df = load_weekly_stats(config)

    stat_matrix, player_info = build_stat_matrix(historical_df, stat="fantasy_points")
    opp_matrix = build_opponent_matrix(historical_df)

    schedule = load_schedule(config=config)
    matchup_grid = build_matchup_grid(schedule)

    win_totals = load_win_totals(source=win_totals_source, api_key=odds_api_key, config=config)
    player_props = load_player_props(source=player_props_source)

    # ---- 2. Alternating Minimisation ----
    altmin = AlternatingMinimization(config.prediction)
    decomp = altmin.fit(
        stats=stat_matrix,
        opponents=opp_matrix,
        player_ids=player_info["player_id"],
        teams=player_info["team"],
    )

    # ---- 3. Vegas Multipliers ----
    w_vegas = compute_vegas_multipliers(win_totals)

    # ---- 4. Blend ----
    w_final = blend_multipliers(
        decomp.defense_multipliers, w_vegas, config.prediction,
    )

    # ---- 5. Raw weekly projections ----
    # Players who changed teams across seasons appear multiple times in
    # player_info.  Keep only the most recent team per player_id.
    _team_df = player_info[["player_id", "team"]].drop_duplicates(
        subset="player_id", keep="last",
    )
    player_teams = pd.Series(
        _team_df["team"].values, index=_team_df["player_id"].values,
    )
    raw = project_raw_weekly(
        decomp.player_skill, w_final, matchup_grid, player_teams,
        weeks=config.nfl_weeks,
    )

    # ---- 6. Volume anchoring ----
    final = volume_anchor(raw, player_props)

    return ProjectionOutput(
        weekly_projections=final,
        player_info=player_info,
        decomposition=decomp,
        w_final=w_final,
    )
