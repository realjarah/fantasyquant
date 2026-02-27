"""NFL schedule grid loader.

Produces a week × team mapping so we know who each team plays in every
week of the target season.
"""

from __future__ import annotations

import pandas as pd

from fantasyquant.config import EngineConfig, DEFAULT_CONFIG


def load_schedule(
    season: int | None = None,
    config: EngineConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """Load the schedule for *season* (defaults to ``config.current_season``).

    Returns
    -------
    pd.DataFrame
        Columns: season, week, home_team, away_team.
    """
    import nfl_data_py as nfl

    season = season or config.current_season
    sched = nfl.import_schedules([season])
    sched = sched[sched["week"] <= config.nfl_weeks]
    return sched[["season", "week", "home_team", "away_team"]].reset_index(drop=True)


def build_matchup_grid(schedule: pd.DataFrame) -> dict[str, dict[int, str]]:
    """Convert a schedule DataFrame into ``{team: {week: opponent}}``.

    Bye weeks will simply be absent from the inner dict.
    """
    grid: dict[str, dict[int, str]] = {}

    for _, row in schedule.iterrows():
        week = int(row["week"])
        home, away = row["home_team"], row["away_team"]
        grid.setdefault(home, {})[week] = away
        grid.setdefault(away, {})[week] = home

    return grid
