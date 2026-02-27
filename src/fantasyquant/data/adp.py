"""Average Draft Position data loading.

ADP is the market signal — it tells the solver what other drafters are
likely to do, which drives constraint (1b) and β(t) estimation.

Sources (in priority order):
  1. User-provided CSV/JSON file
  2. Scoring-format-aware VOR ranking derived from prior season
     actuals via nfl_data_py (automatic, no user input needed)
  3. Projection-derived fallback ranking

The default path (source 2) produces ADP that is **specific to the
user's exact scoring and roster settings**: PPR weight, TE premium,
passing TD points, superflex, team count — all flow into a Value Over
Replacement (VOR) calculation that shifts positional rankings to match
how each format changes draft capital.
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
    """Load ADP data, automatically adjusted for scoring format.

    Parameters
    ----------
    source:
        Path to a CSV (columns: ``player_id``, ``adp``) or JSON
        (``{"player_id": adp, ...}``).  When provided, used as-is.
    config:
        Engine configuration.  The scoring and roster settings drive the
        automatic VOR-based ADP derivation.
    platform:
        Override platform for ADP population (not used in auto mode).
    projections:
        If provided and no other source available, derive ADP from
        total projected points (highest projected = ADP 1).

    Returns
    -------
    pd.Series | None
        Indexed by player_id, values are ADP rank (1-indexed float).
        Returns None only if no data source is available.
    """
    # 1. User-provided file — use exactly as given.
    if source is not None:
        return _load_from_file(Path(source))

    # 2. Scoring-format-aware VOR ranking from nfl_data_py.
    adp = _derive_vor_adp(config)
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


# ---------------------------------------------------------------------------
# VOR-based ADP derivation (scoring + roster aware)
# ---------------------------------------------------------------------------

def _replacement_level(config: EngineConfig) -> dict[str, float]:
    """Compute the replacement-level rank for each position.

    The "replacement player" at a position is the last player that would
    be a weekly starter across all teams in the league.  This naturally
    handles:
      - Superflex/2QB: QB replacement rank jumps from ~12 to ~22
      - Deep leagues (14+ teams): replacement levels drop for all positions
      - Flex-heavy leagues: RB/WR replacement levels shift

    Returns ``{position: rank}`` where rank is a 1-indexed float.
    """
    r = config.roster
    n = r.teams

    # Flex slots are shared among RB/WR/TE.  Empirically:
    #   ~50% of flex spots go to RB, ~40% WR, ~10% TE.
    # In superflex, ~70% of teams start a QB there.
    flex_rb = 0.50 * r.flex
    flex_wr = 0.40 * r.flex
    flex_te = 0.10 * r.flex

    sflex_qb = 0.70 * r.superflex
    sflex_rb = 0.10 * r.superflex
    sflex_wr = 0.10 * r.superflex
    sflex_te = 0.10 * r.superflex

    return {
        "QB": n * (r.qb + sflex_qb) + 1,
        "RB": n * (r.rb + flex_rb + sflex_rb) + 1,
        "WR": n * (r.wr + flex_wr + sflex_wr) + 1,
        "TE": n * (r.te + flex_te + sflex_te) + 1,
    }


def _derive_vor_adp(config: EngineConfig) -> pd.Series | None:
    """Compute VOR-based ADP from prior season actuals.

    Steps:
      1. Load prior season weekly stats via nfl_data_py
      2. Score each week using the user's exact scoring settings
      3. Sum season totals per player
      4. Compute replacement level per position from roster settings
      5. VOR = season_total - replacement_level(position)
      6. Rank by VOR descending -> ADP
    """
    try:
        from fantasyquant.data.historical import (
            FANTASY_POSITIONS,
            _COLUMN_MAP,
            compute_fantasy_points,
        )
        import nfl_data_py as nfl
    except ImportError:
        return None

    try:
        season = config.current_season - 1
        raw = nfl.import_weekly_data([season])
        if raw is None or raw.empty:
            return None

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

        # Ensure stat columns exist.
        for col in (
            "passing_yards", "passing_tds", "interceptions",
            "rushing_yards", "rushing_tds", "receptions",
            "receiving_yards", "receiving_tds", "fumbles_lost",
        ):
            if col not in df.columns:
                df[col] = 0.0

        # Score with user's exact settings (PPR, TE premium, 6pt pass, etc).
        df["fpts"] = compute_fantasy_points(df, config.scoring)

        # Season totals per player.
        season_totals = df.groupby("player_id")["fpts"].sum()
        player_positions = df.groupby("player_id")["position"].first()

        # Only keep players with meaningful production.
        season_totals = season_totals[season_totals > 20.0]

        # Compute replacement level for each position.
        repl = _replacement_level(config)

        # Find actual replacement-level score for each position.
        repl_scores: dict[str, float] = {}
        for pos, rank in repl.items():
            pos_players = season_totals[player_positions == pos].sort_values(ascending=False)
            idx = int(rank) - 1
            if idx < len(pos_players):
                repl_scores[pos] = float(pos_players.iloc[idx])
            elif len(pos_players) > 0:
                repl_scores[pos] = float(pos_players.iloc[-1])
            else:
                repl_scores[pos] = 0.0

        # VOR = total - replacement_level(position).
        vor = pd.Series(dtype=float, name="vor")
        for pid, total in season_totals.items():
            pos = player_positions.get(pid, "")
            baseline = repl_scores.get(pos, 0.0)
            vor[pid] = total - baseline

        # Rank by VOR descending -> ADP (rank 1 = highest VOR).
        vor = vor.sort_values(ascending=False)
        adp = pd.Series(
            range(1, len(vor) + 1),
            index=vor.index,
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
