"""Average Draft Position data loading.

ADP is the market signal — it tells the solver what other drafters are
likely to do, which drives constraint (1b) and β(t) estimation.

Sources (in priority order):
  1. User-provided CSV/JSON file
  2. nfl_data_py draft picks (ECR-based, updated weekly during preseason)
  3. Built-in fallback from projection rankings

Expected output shape: ``pd.Series`` indexed by ``player_id`` with
values representing overall ADP rank (1 = first pick).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from fantasyquant.config import EngineConfig, DEFAULT_CONFIG


def load_adp(
    source: str | Path | None = None,
    *,
    config: EngineConfig = DEFAULT_CONFIG,
    platform: str | None = None,
    projections: pd.DataFrame | None = None,
) -> pd.Series | None:
    """Load ADP data from a file, nfl_data_py, or projection fallback.

    Parameters
    ----------
    source:
        Path to a CSV (columns: ``player_id``, ``adp``) or JSON
        (``{"player_id": adp, ...}``).
    config:
        Engine configuration (uses ``current_season``, ``platform``).
    platform:
        Override platform for ADP population (e.g. "espn", "sleeper").
        Falls back to ``config.platform``.
    projections:
        If provided and no other source available, derive ADP from
        total projected points (highest projected = ADP 1).

    Returns
    -------
    pd.Series | None
        Indexed by player_id, values are ADP rank (1-indexed float).
        Returns None only if no data source is available.
    """
    # 1. User-provided file.
    if source is not None:
        return _load_from_file(Path(source))

    # 2. nfl_data_py draft data (ECR / ADP aggregates).
    adp = _load_from_nfl_data_py(config, platform)
    if adp is not None and len(adp) > 0:
        return adp

    # 3. Fallback: derive from projections.
    if projections is not None and len(projections) > 0:
        return _derive_from_projections(projections)

    return None


def _load_from_file(path: Path) -> pd.Series:
    """Load ADP from a local CSV or JSON file."""
    if path.suffix == ".json":
        data = json.loads(path.read_text())
        return pd.Series(data, dtype=float, name="adp")

    # CSV: expect columns player_id, adp.
    df = pd.read_csv(path)
    if "player_id" in df.columns and "adp" in df.columns:
        return pd.Series(df["adp"].values, index=df["player_id"].values, name="adp", dtype=float)

    # Alternative column names.
    id_col = _find_column(df, ["player_id", "id", "sleeper_id", "gsis_id"])
    adp_col = _find_column(df, ["adp", "avg_pick", "average_pick", "overall"])
    if id_col and adp_col:
        return pd.Series(df[adp_col].values, index=df[id_col].values, name="adp", dtype=float)

    raise ValueError(f"Cannot parse ADP from {path}: expected columns 'player_id' and 'adp'")


def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Find the first matching column name (case-insensitive)."""
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def _load_from_nfl_data_py(
    config: EngineConfig,
    platform: str | None = None,
) -> pd.Series | None:
    """Fetch ADP from nfl_data_py's draft data.

    nfl_data_py provides Expert Consensus Rankings (ECR) and
    platform-specific ADP aggregated from FantasyPros.
    """
    try:
        import nfl_data_py as nfl
    except ImportError:
        return None

    try:
        # import_seasonal_data includes 'fantasy_points_ppr' and
        # season-level data.  For ADP, we use draft picks or rosters.
        # nfl_data_py.import_draft_picks gives actual draft capital, not
        # fantasy ADP.  The best proxy is ECR / seasonal projections rank.
        #
        # import_seasonal_data gives season-level stats for completed seasons.
        # For current-season ADP, we rank by expert projected points.
        season = config.current_season

        # Try loading weekly projections for ranking.
        # nfl_data_py doesn't have a direct ADP endpoint, but we can
        # rank players by their projected season totals.
        weekly = nfl.import_weekly_data([season - 1])
        if weekly is None or weekly.empty:
            return None

        # Use last completed season's actual performance as ADP proxy.
        # This is better than nothing — real ADP data would come from
        # a user-uploaded file or a dedicated API.
        fantasy_positions = {"QB", "RB", "WR", "TE"}
        weekly = weekly[weekly["position"].isin(fantasy_positions)]
        weekly = weekly[weekly["week"] <= config.nfl_weeks]

        # Sum fantasy points per player.
        scoring = config.scoring
        pts = (
            weekly["passing_yards"].fillna(0) * scoring.passing_yards
            + weekly["passing_tds"].fillna(0) * scoring.passing_tds
            + weekly["interceptions"].fillna(0) * scoring.interceptions
            + weekly["rushing_yards"].fillna(0) * scoring.rushing_yards
            + weekly["rushing_tds"].fillna(0) * scoring.rushing_tds
            + weekly["receptions"].fillna(0) * scoring.receptions
            + weekly["receiving_yards"].fillna(0) * scoring.receiving_yards
            + weekly["receiving_tds"].fillna(0) * scoring.receiving_tds
            + weekly.get("sack_fumbles_lost", pd.Series(0, index=weekly.index)).fillna(0) * scoring.fumbles_lost
        )
        weekly = weekly.copy()
        weekly["fpts"] = pts
        season_totals = weekly.groupby("player_id")["fpts"].sum()
        season_totals = season_totals.sort_values(ascending=False)

        # ADP = rank position (1 = highest scorer last season).
        adp = pd.Series(
            range(1, len(season_totals) + 1),
            index=season_totals.index,
            name="adp",
            dtype=float,
        )
        return adp

    except Exception:
        return None


def _derive_from_projections(projections: pd.DataFrame) -> pd.Series:
    """Derive ADP ranking from total projected points (highest = rank 1)."""
    totals = projections.sum(axis=1).sort_values(ascending=False)
    adp = pd.Series(
        range(1, len(totals) + 1),
        index=totals.index,
        name="adp",
        dtype=float,
    )
    return adp
