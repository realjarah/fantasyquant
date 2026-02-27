"""Average Draft Position data loading.

ADP is the market signal — it tells the solver what other drafters are
likely to do, which drives constraint (1b) and β(t) estimation in the
Becker & Sun (2013) formulation.

The paper uses **live ADP / Expert Consensus Rankings** — a
forward-looking market signal — not backward-looking stats.  Our
priority chain reflects this:

  1. User-provided CSV/JSON file (explicit override)
  2. **Live market ADP** from Fantasy Football Calculator's public API,
     matched to the user's scoring format and league size
  3. VOR-derived ranking from prior season actuals (off-season fallback
     when no live ADP is available yet)
  4. Projection-derived ranking (last resort)

The live ADP path (source 2) is format-specific: PPR, half-PPR,
standard, 2QB — each returns different rankings because the drafting
market prices players differently across formats.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path

import pandas as pd
import requests

from fantasyquant.config import EngineConfig, DEFAULT_CONFIG

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ADP cache (one fetch per format per process lifetime, or TTL-based)
# ---------------------------------------------------------------------------

_ADP_CACHE: dict[str, tuple[float, pd.Series]] = {}
_CACHE_TTL_SECONDS = 3600  # 1 hour


def _cache_key(format_str: str, teams: int, year: int) -> str:
    return f"{format_str}:{teams}:{year}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_adp(
    source: str | Path | None = None,
    *,
    config: EngineConfig = DEFAULT_CONFIG,
    platform: str | None = None,
    projections: pd.DataFrame | None = None,
) -> pd.Series | None:
    """Load ADP data — live market rankings by default.

    Parameters
    ----------
    source:
        Path to a CSV (columns: ``player_id``, ``adp``) or JSON
        (``{"player_id": adp, ...}``).  When provided, used as-is.
    config:
        Engine configuration.  Scoring and roster settings determine
        which ADP format to fetch (PPR, half-PPR, standard, 2QB).
    platform:
        Override platform hint (not currently used for API selection).
    projections:
        If provided and no other source available, derive ADP from
        total projected points (highest projected = ADP 1).

    Returns
    -------
    pd.Series | None
        Indexed by player_id, values are ADP rank (1-indexed float).
        Returns None only if no data source is available.
    """
    # 1. User-provided file — explicit override.
    if source is not None:
        return _load_from_file(Path(source))

    # 2. Live market ADP from public API.
    adp = _fetch_live_adp(config)
    if adp is not None and len(adp) > 0:
        return adp

    # 3. VOR-derived from prior season actuals (off-season fallback).
    adp = _derive_vor_adp(config)
    if adp is not None and len(adp) > 0:
        return adp

    # 4. Projection-derived (last resort).
    if projections is not None and len(projections) > 0:
        return _derive_from_projections(projections)

    return None


# ---------------------------------------------------------------------------
# Source 1: User file
# ---------------------------------------------------------------------------

def _load_from_file(path: Path) -> pd.Series:
    """Load ADP from a local CSV or JSON file."""
    if path.suffix == ".json":
        data = json.loads(path.read_text())
        return pd.Series(data, dtype=float, name="adp")

    df = pd.read_csv(path)
    if "player_id" in df.columns and "adp" in df.columns:
        return pd.Series(df["adp"].values, index=df["player_id"].values, name="adp", dtype=float)

    id_col = _find_column(df, ["player_id", "id", "sleeper_id", "gsis_id"])
    adp_col = _find_column(df, ["adp", "avg_pick", "average_pick", "overall"])
    if id_col and adp_col:
        return pd.Series(df[adp_col].values, index=df[id_col].values, name="adp", dtype=float)

    raise ValueError(f"Cannot parse ADP from {path}: expected columns 'player_id' and 'adp'")


def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


# ---------------------------------------------------------------------------
# Source 2: Live market ADP (Fantasy Football Calculator API)
# ---------------------------------------------------------------------------

_FFC_BASE = "https://fantasyfootballcalculator.com/api/v1/adp"


def _scoring_to_ffc_format(config: EngineConfig) -> str:
    """Map our scoring settings to FFC's format string.

    FFC supports: standard, ppr, half-ppr, 2qb, superflex, dynasty,
    idp, rookie.
    """
    roster = config.roster
    scoring = config.scoring

    # Superflex / 2QB take priority — they dominate ADP shape.
    if roster.superflex > 0:
        return "superflex"
    if roster.qb >= 2:
        return "2qb"

    # PPR weight.
    if scoring.receptions >= 0.8:
        return "ppr"
    if scoring.receptions >= 0.3:
        return "half-ppr"
    return "standard"


def _ffc_teams(config: EngineConfig) -> int:
    """Snap to nearest FFC-supported team count (8, 10, 12, 14)."""
    teams = config.roster.teams
    for n in [8, 10, 12, 14]:
        if teams <= n:
            return n
    return 14


def _fetch_ffc_year(fmt: str, teams: int, year: int) -> list[dict] | None:
    """Fetch one year's data from FFC.  Returns the players list or None."""
    try:
        url = f"{_FFC_BASE}/{fmt}"
        params = {"teams": teams, "year": year}
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") == "Error":
            return None
        players = data.get("players", [])
        return players if players else None
    except (requests.RequestException, json.JSONDecodeError, KeyError):
        return None


def _fetch_live_adp(config: EngineConfig) -> pd.Series | None:
    """Fetch live ADP from Fantasy Football Calculator's public API.

    Returns format-specific ADP (PPR, half-PPR, standard, 2QB, superflex)
    matched to the user's league settings.  Results are cached for 1 hour.

    Tries the upcoming season first; if the API doesn't have data yet
    (common in the off-season, Jan-Jul), falls back to the most recent
    available year.

    The API returns player names which we cross-reference against
    nfl_data_py's player roster to map to player_ids.
    """
    fmt = _scoring_to_ffc_format(config)
    teams = _ffc_teams(config)
    target_year = config.current_season

    # Check cache for the target year.
    key = _cache_key(fmt, teams, target_year)
    if key in _ADP_CACHE:
        ts, cached = _ADP_CACHE[key]
        if time.time() - ts < _CACHE_TTL_SECONDS:
            return cached

    # Try target year first, then fall back up to 2 years.
    players = None
    used_year = target_year
    for year in [target_year, target_year - 1, target_year - 2]:
        players = _fetch_ffc_year(fmt, teams, year)
        if players:
            used_year = year
            break

    if not players:
        logger.debug("No live ADP data available from FFC for %d–%d", target_year - 2, target_year)
        return None

    if used_year != target_year:
        logger.info(
            "FFC does not have %d ADP yet — using %d data as the latest available",
            target_year, used_year,
        )

    # Build name → ADP mapping from API response.
    name_adp: dict[str, float] = {}
    for p in players:
        name = p.get("name", "")
        adp_val = p.get("adp")
        if name and adp_val is not None:
            name_adp[name] = float(adp_val)

    if not name_adp:
        return None

    # Cross-reference with nfl_data_py roster to get player_ids.
    adp_series = _match_names_to_ids(name_adp, config)

    if adp_series is not None and len(adp_series) > 0:
        _ADP_CACHE[key] = (time.time(), adp_series)
        logger.info(
            "Loaded live %s ADP for %d-team leagues (%d players, %d data)",
            fmt.upper(), teams, len(adp_series), used_year,
        )
        return adp_series

    return None


def _match_names_to_ids(
    name_adp: dict[str, float],
    config: EngineConfig,
) -> pd.Series | None:
    """Map player display names from FFC to nfl_data_py player_ids.

    Uses ``nfl.import_ids()`` — the canonical cross-platform ID table —
    and fuzzy matching: normalize both sides to lowercase, strip
    suffixes (Jr., III, etc.), and match.
    """
    try:
        import nfl_data_py as nfl
    except ImportError:
        return None

    try:
        ids = nfl.import_ids()
        if ids is None or ids.empty:
            return None

        # Build normalized name → gsis_id lookup.
        id_lookup: dict[str, str] = {}
        for _, row in ids.iterrows():
            pid = row.get("gsis_id", "")
            if not pid or pd.isna(pid):
                continue
            pid = str(pid)
            for col in ["name", "merge_name"]:
                name = row.get(col, "")
                if name and not pd.isna(name):
                    id_lookup[_normalize_name(str(name))] = pid

        # Match FFC names to player_ids.
        matched: dict[str, float] = {}
        for name, adp_val in name_adp.items():
            norm = _normalize_name(name)
            pid = id_lookup.get(norm)
            if pid:
                matched[pid] = adp_val

        if not matched:
            return None

        return pd.Series(matched, dtype=float, name="adp").sort_values()

    except Exception:
        return None


def _normalize_name(name: str) -> str:
    """Normalize a player name for fuzzy matching.

    Strips suffixes, punctuation, and lowercases.
    """
    name = name.lower().strip()
    # Remove common suffixes.
    for suffix in [" jr.", " jr", " sr.", " sr", " iii", " ii", " iv", " v"]:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    # Remove punctuation.
    name = name.replace(".", "").replace("'", "").replace("-", " ")
    # Collapse whitespace.
    return " ".join(name.split())


# ---------------------------------------------------------------------------
# Source 3: VOR-based ADP (off-season / live-data fallback)
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

    Used as an off-season fallback when live market ADP is not yet
    available (e.g., January–June before preseason drafts begin).
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
        player_positions = df.groupby("player_id")["position"].first()

        season_totals = season_totals[season_totals > 20.0]

        repl = _replacement_level(config)

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

        vor = pd.Series(dtype=float, name="vor")
        for pid, total in season_totals.items():
            pos = player_positions.get(pid, "")
            baseline = repl_scores.get(pos, 0.0)
            vor[pid] = total - baseline

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


# ---------------------------------------------------------------------------
# Source 4: Projection-derived (last resort)
# ---------------------------------------------------------------------------

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
