"""NFL schedule grid loader.

Produces a week × team mapping so we know who each team plays in every
week of the target season.
"""

from __future__ import annotations

import logging

import pandas as pd

from fantasyquant.config import EngineConfig, DEFAULT_CONFIG

logger = logging.getLogger(__name__)


def load_schedule(
    season: int | None = None,
    config: EngineConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """Load the schedule for *season* (defaults to ``config.current_season``).

    If the upcoming season's schedule isn't published yet, falls back to
    the most recent available year.

    Returns
    -------
    pd.DataFrame
        Columns: season, week, home_team, away_team.
    """
    import nfl_data_py as nfl

    target = season or config.current_season

    for yr in [target, target - 1, target - 2]:
        try:
            sched = nfl.import_schedules([yr])
            if sched is not None and not sched.empty:
                if yr != target:
                    logger.info(
                        "Schedule for %d not available — using %d as proxy",
                        target, yr,
                    )
                sched = sched[sched["week"] <= config.nfl_weeks]
                return sched[["season", "week", "home_team", "away_team"]].reset_index(drop=True)
        except Exception:
            continue

    raise RuntimeError(f"No NFL schedule available for {target} through {target - 2}")


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
