"""Blend historical and Vegas defensive multipliers, then apply volume
anchoring to produce final weekly projections.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fantasyquant.config import PredictionConfig


def blend_multipliers(
    w_historical: pd.Series,
    w_vegas: pd.Series,
    config: PredictionConfig | None = None,
) -> pd.Series:
    """Weighted average of backward- and forward-looking multipliers.

    Teams present in only one source get the other's value. Teams in
    neither are set to 1.0 (neutral).
    """
    config = config or PredictionConfig()
    all_teams = sorted(set(w_historical.index) | set(w_vegas.index))
    blended: dict[str, float] = {}
    for team in all_teams:
        h = w_historical.get(team, np.nan)
        v = w_vegas.get(team, np.nan)
        if np.isnan(h) and np.isnan(v):
            blended[team] = 1.0
        elif np.isnan(h):
            blended[team] = v
        elif np.isnan(v):
            blended[team] = h
        else:
            blended[team] = config.historical_weight * h + config.vegas_weight * v
    return pd.Series(blended, name="w_final").sort_index()


def project_raw_weekly(
    player_skill: pd.Series,
    w_final: pd.Series,
    matchup_grid: dict[str, dict[int, str]],
    player_teams: pd.Series,
    weeks: int = 17,
) -> pd.DataFrame:
    """Compute raw weekly projections: ``P_raw(i, t) = u_i × W_final(opp_t)``.

    Parameters
    ----------
    player_skill:
        Series indexed by player_id with skill values (u_i).
    w_final:
        Series indexed by team abbreviation with blended multipliers.
    matchup_grid:
        ``{team: {week: opponent_team}}`` mapping.
    player_teams:
        Series indexed by player_id → team abbreviation.
    weeks:
        Number of weeks to project.

    Returns
    -------
    pd.DataFrame
        Index = player_id, columns = week numbers (1..weeks).
    """
    # Deduplicate players who appear multiple times (e.g., team changes).
    # Keep the highest skill estimate.
    player_skill = player_skill.groupby(player_skill.index).max()

    records: list[dict] = []
    for pid, skill in player_skill.items():
        team = player_teams.get(pid)
        if team is None or not isinstance(team, str) or team not in matchup_grid:
            continue
        row: dict = {"player_id": pid}
        for w in range(1, weeks + 1):
            opp = matchup_grid.get(team, {}).get(w)
            if opp is None:
                row[w] = 0.0  # bye week
            else:
                mult = w_final.get(opp, 1.0)
                row[w] = skill * mult
        records.append(row)

    df = pd.DataFrame(records).set_index("player_id")
    df.columns.name = "week"
    return df


def volume_anchor(
    raw_weekly: pd.DataFrame,
    player_props: dict[str, float],
) -> pd.DataFrame:
    """Scale weekly projections so each player's season total matches
    their Vegas season prop (g(i)).

    Players without a prop line are left unscaled.
    """
    scaled = raw_weekly.copy()
    for pid in scaled.index:
        if pid not in player_props:
            continue
        raw_total = scaled.loc[pid].sum()
        if raw_total <= 0:
            continue
        target = player_props[pid]
        factor = target / raw_total
        scaled.loc[pid] = scaled.loc[pid] * factor
    return scaled
