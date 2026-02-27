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

    Uses the canonical ``compute_fantasy_points()`` from historical.py
    so scoring is consistent everywhere (PPR, TE premium, etc).
    """
    try:
        from fantasyquant.data.historical import (
            FANTASY_POSITIONS,
            _COLUMN_MAP,
            compute_fantasy_points,
        )
        import nfl_data_py as nfl
    except ImportError:
        return {}

    try:
        season = config.current_season - 1
        raw = nfl.import_weekly_data([season])
        if raw is None or raw.empty:
            return {}

        raw = raw[raw["position"].isin(FANTASY_POSITIONS)]
        raw = raw[raw["week"] <= config.nfl_weeks]

        # Rename columns to match compute_fantasy_points expectations.
        if "player_display_name" in raw.columns and "player_name" in raw.columns:
            raw = raw.drop(columns=["player_name"])
        available = {k: v for k, v in _COLUMN_MAP.items() if k in raw.columns}
        seen: set[str] = set()
        target_cols: list[str] = []
        for v in available.values():
            if v not in seen:
                target_cols.append(v)
                seen.add(v)
        df = raw.rename(columns=available)[target_cols].copy()

        for col in (
            "passing_yards", "passing_tds", "interceptions",
            "rushing_yards", "rushing_tds", "receptions",
            "receiving_yards", "receiving_tds", "fumbles_lost",
        ):
            if col not in df.columns:
                df[col] = 0.0

        df["fpts"] = compute_fantasy_points(df, config.scoring)
        season_totals = df.groupby("player_id")["fpts"].sum()

        # Only include players with meaningful production (> 20 pts/season).
        season_totals = season_totals[season_totals > 20.0]

        return season_totals.to_dict()

    except Exception:
        return {}
