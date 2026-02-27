"""Player prop season totals loader.

Player props (season fantasy point totals from Vegas) feed into
``volume_anchor()`` to calibrate weekly projections so each player's
projected season total matches the market consensus.

Sources (in priority order):
  1. User-provided CSV/JSON file
  2. nfl_data_py historical season totals (prior season actuals as proxy)
  3. Empty dict (volume anchoring becomes a no-op)

Expected output shape: ``dict[str, float]`` mapping player_id → season
total fantasy points.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from fantasyquant.config import EngineConfig, DEFAULT_CONFIG


def load_player_props(
    source: str | Path | None = None,
    *,
    config: EngineConfig = DEFAULT_CONFIG,
) -> dict[str, float]:
    """Load player season total props.

    Parameters
    ----------
    source:
        Path to a CSV (columns: ``player_id``, ``season_total``) or JSON
        (``{"player_id": total, ...}``).
    config:
        Engine configuration (uses ``current_season``, ``scoring``).

    Returns
    -------
    dict[str, float]
        Mapping of player_id → projected season fantasy points.
        Empty dict if no source available.
    """
    # 1. User-provided file.
    if source is not None:
        return _load_from_file(Path(source))

    # 2. Derive from prior season actuals via nfl_data_py.
    props = _derive_from_historical(config)
    if props:
        return props

    # 3. Empty — volume_anchor becomes a no-op.
    return {}


def _load_from_file(path: Path) -> dict[str, float]:
    """Load props from a local CSV or JSON file."""
    if path.suffix == ".json":
        return json.loads(path.read_text())

    # CSV: expect columns player_id, season_total.
    df = pd.read_csv(path)
    id_col = _find_column(df, ["player_id", "id", "sleeper_id", "gsis_id"])
    total_col = _find_column(df, ["season_total", "total", "fpts", "fantasy_points", "projected_total"])
    if id_col and total_col:
        return dict(zip(df[id_col].astype(str), df[total_col].astype(float)))

    raise ValueError(
        f"Cannot parse props from {path}: "
        "expected columns 'player_id' and 'season_total'"
    )


def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Find the first matching column name (case-insensitive)."""
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def _derive_from_historical(config: EngineConfig) -> dict[str, float]:
    """Use prior season actuals from nfl_data_py as prop proxies.

    This gives the system real per-player season totals to anchor
    projections against, rather than leaving volume_anchor as a no-op.
    """
    try:
        import nfl_data_py as nfl
    except ImportError:
        return {}

    try:
        season = config.current_season - 1
        weekly = nfl.import_weekly_data([season])
        if weekly is None or weekly.empty:
            return {}

        fantasy_positions = {"QB", "RB", "WR", "TE"}
        weekly = weekly[weekly["position"].isin(fantasy_positions)]
        weekly = weekly[weekly["week"] <= config.nfl_weeks]

        # Compute fantasy points using league scoring.
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

        # TE premium.
        if scoring.te_reception_bonus != 0.0 and "position" in weekly.columns:
            te_mask = weekly["position"] == "TE"
            pts = pts + te_mask * weekly["receptions"].fillna(0) * scoring.te_reception_bonus

        weekly = weekly.copy()
        weekly["fpts"] = pts
        season_totals = weekly.groupby("player_id")["fpts"].sum()

        # Only include players with meaningful production (> 20 pts/season).
        season_totals = season_totals[season_totals > 20.0]

        return season_totals.to_dict()

    except Exception:
        return {}
