"""Derive forward-looking defensive multipliers from Vegas win totals.

Pipeline:
  Win Total → Power Rating → Implied Points Allowed → Defensive Multiplier
"""

from __future__ import annotations

import pandas as pd


# League-average baseline (roughly 22.5 points per game in a 17-game season).
_LEAGUE_AVG_PA = 22.5  # points allowed per game, league average
_WIN_TOTAL_BASELINE = 8.5
_POWER_RATING_SCALE = 1.75  # points per win above/below .500


def win_totals_to_power_ratings(
    win_totals: dict[str, float],
    baseline: float = _WIN_TOTAL_BASELINE,
    scale: float = _POWER_RATING_SCALE,
) -> dict[str, float]:
    """Convert win totals to relative power ratings.

    A team with win total 11.5 gets:  (11.5 - 8.5) * 1.75 = +5.25
    """
    return {team: (wt - baseline) * scale for team, wt in win_totals.items()}


def power_ratings_to_implied_pa(
    power_ratings: dict[str, float],
    league_avg_pa: float = _LEAGUE_AVG_PA,
) -> dict[str, float]:
    """Derive implied points-allowed from power ratings.

    A positive power rating means a *good* team, which implies a *good*
    defense, hence they allow **fewer** points.

    ``Implied PA = league_avg - power_rating``

    So a +5.25 team allows roughly 17.25 points/game.
    """
    return {team: league_avg_pa - pr for team, pr in power_ratings.items()}


def implied_pa_to_multiplier(
    implied_pa: dict[str, float],
    league_avg_pa: float = _LEAGUE_AVG_PA,
) -> dict[str, float]:
    """Express implied PA as a multiplier relative to league average.

    ``W_vegas = implied_pa / league_avg``

    A team allowing 10 % more points → 1.10.
    """
    return {team: pa / league_avg_pa for team, pa in implied_pa.items()}


def compute_vegas_multipliers(
    win_totals: dict[str, float],
) -> pd.Series:
    """End-to-end: win totals → defensive multiplier series.

    Returns
    -------
    pd.Series
        Indexed by team abbreviation, values are W_vegas.
    """
    pr = win_totals_to_power_ratings(win_totals)
    pa = power_ratings_to_implied_pa(pr)
    mult = implied_pa_to_multiplier(pa)
    return pd.Series(mult, name="w_vegas").sort_index()
