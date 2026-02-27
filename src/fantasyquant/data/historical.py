"""Ingest historical weekly player stats via *nfl_data_py*.

The main entry point is :func:`load_weekly_stats`, which returns a tidy
DataFrame with one row per player-week and computed fantasy points.
"""

from __future__ import annotations

import pandas as pd

from fantasyquant.config import EngineConfig, DEFAULT_CONFIG, ScoringSettings

# Positions we care about for fantasy purposes.
FANTASY_POSITIONS = {"QB", "RB", "WR", "TE"}

# Column mapping from nfl_data_py weekly data to our internal names.
_COLUMN_MAP = {
    "player_id": "player_id",
    "player_name": "player_name",
    "player_display_name": "player_name",
    "recent_team": "team",
    "opponent_team": "opponent",
    "position": "position",
    "season": "season",
    "week": "week",
    "passing_yards": "passing_yards",
    "passing_tds": "passing_tds",
    "interceptions": "interceptions",
    "rushing_yards": "rushing_yards",
    "rushing_tds": "rushing_tds",
    "receptions": "receptions",
    "receiving_yards": "receiving_yards",
    "receiving_tds": "receiving_tds",
    "sack_fumbles_lost": "fumbles_lost",
}


def compute_fantasy_points(df: pd.DataFrame, scoring: ScoringSettings) -> pd.Series:
    """Vectorised fantasy-point calculation.

    Handles position-dependent scoring (TE premium) when a ``position``
    column is present.
    """
    pts = (
        df["passing_yards"].fillna(0) * scoring.passing_yards
        + df["passing_tds"].fillna(0) * scoring.passing_tds
        + df["interceptions"].fillna(0) * scoring.interceptions
        + df["rushing_yards"].fillna(0) * scoring.rushing_yards
        + df["rushing_tds"].fillna(0) * scoring.rushing_tds
        + df["receptions"].fillna(0) * scoring.receptions
        + df["receiving_yards"].fillna(0) * scoring.receiving_yards
        + df["receiving_tds"].fillna(0) * scoring.receiving_tds
        + df["fumbles_lost"].fillna(0) * scoring.fumbles_lost
    )

    # TE premium: extra points per TE reception.
    if scoring.te_reception_bonus != 0.0 and "position" in df.columns:
        te_mask = df["position"] == "TE"
        pts = pts + te_mask * df["receptions"].fillna(0) * scoring.te_reception_bonus

    return pts


def load_weekly_stats(
    config: EngineConfig = DEFAULT_CONFIG,
    seasons: list[int] | None = None,
) -> pd.DataFrame:
    """Load and clean weekly player stats for the training window.

    Parameters
    ----------
    config:
        Engine configuration (uses ``current_season`` and
        ``prediction.training_seasons``).
    seasons:
        Explicit list of seasons to load.  If *None*, derived from config.

    Returns
    -------
    pd.DataFrame
        Columns: player_id, player_name, team, opponent, position, season,
        week, passing_yards, passing_tds, interceptions, rushing_yards,
        rushing_tds, receptions, receiving_yards, receiving_tds,
        fumbles_lost, fantasy_points.
    """
    import nfl_data_py as nfl  # lazy import — heavy dependency

    if seasons is None:
        end = config.current_season - 1  # don't include current (incomplete)
        start = end - config.prediction.training_seasons + 1
        seasons = list(range(start, end + 1))

    raw = nfl.import_weekly_data(seasons)

    # Keep only fantasy-relevant positions and regular season (weeks 1-17/18).
    raw = raw[raw["position"].isin(FANTASY_POSITIONS)]
    raw = raw[raw["week"] <= config.nfl_weeks]

    # Rename and subset columns.  Prefer player_display_name over
    # player_name to avoid duplicates when both exist.
    if "player_display_name" in raw.columns and "player_name" in raw.columns:
        raw = raw.drop(columns=["player_name"])
    available = {k: v for k, v in _COLUMN_MAP.items() if k in raw.columns}
    # Deduplicate target column names while preserving order.
    seen: set[str] = set()
    target_cols: list[str] = []
    for v in available.values():
        if v not in seen:
            target_cols.append(v)
            seen.add(v)
    df = raw.rename(columns=available)[target_cols].copy()

    # Ensure numeric stat columns exist even if source is missing one.
    for col in (
        "passing_yards", "passing_tds", "interceptions",
        "rushing_yards", "rushing_tds", "receptions",
        "receiving_yards", "receiving_tds", "fumbles_lost",
    ):
        if col not in df.columns:
            df[col] = 0.0

    df["fantasy_points"] = compute_fantasy_points(df, config.scoring)

    return df.reset_index(drop=True)


def build_stat_matrix(
    df: pd.DataFrame,
    stat: str = "fantasy_points",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pivot weekly stats into a player × week matrix.

    Returns
    -------
    (player_week_matrix, player_info)
        *player_week_matrix* has players as rows, (season, week) as columns.
        *player_info* maps row index → (player_id, player_name, position, team).
    """
    pivot = df.pivot_table(
        index=["player_id", "player_name", "position", "team"],
        columns=["season", "week"],
        values=stat,
        aggfunc="first",
    ).fillna(0.0)

    info = pivot.index.to_frame(index=False)
    pivot = pivot.reset_index(drop=True)
    return pivot, info


def build_opponent_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Build a player × week matrix of opponent team names.

    Same shape as :func:`build_stat_matrix` output but cells hold the
    opposing team abbreviation (or NaN if no game).
    """
    pivot = df.pivot_table(
        index=["player_id", "player_name", "position", "team"],
        columns=["season", "week"],
        values="opponent",
        aggfunc="first",
    )
    return pivot.reset_index(drop=True)
