"""Vegas odds data interface.

Provides helpers to load season win totals and player prop season
totals.  Live data requires an API key for The Odds API or similar;
we also support loading from local JSON/CSV files for offline use
and backtesting.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from fantasyquant.config import EngineConfig, DEFAULT_CONFIG

# ---------------------------------------------------------------------------
# Win Totals
# ---------------------------------------------------------------------------

# Fallback/example win totals when no live feed is available.
# Source: approximate consensus lines (update each preseason).
_DEFAULT_WIN_TOTALS: dict[str, float] = {
    "KC": 11.5, "SF": 10.5, "DET": 10.5, "BAL": 10.5,
    "BUF": 10.0, "PHI": 10.0, "DAL": 9.5, "CIN": 9.5,
    "MIA": 9.5, "HOU": 9.5, "NYJ": 9.0, "JAX": 9.0,
    "CLE": 8.5, "GB": 8.5, "PIT": 8.5, "LAR": 8.5,
    "LAC": 8.5, "CHI": 8.0, "SEA": 8.0, "ATL": 8.0,
    "MIN": 8.0, "TB": 8.0, "NO": 7.5, "IND": 7.5,
    "DEN": 7.0, "TEN": 6.5, "LV": 6.5, "NYG": 6.5,
    "ARI": 6.0, "WAS": 6.0, "CAR": 5.5, "NE": 4.5,
}


def load_win_totals(
    source: str | Path | None = None,
    api_key: str | None = None,
    config: EngineConfig = DEFAULT_CONFIG,
) -> dict[str, float]:
    """Return ``{team_abbr: win_total}`` for the target season.

    Resolution order:

    1. *source* is a path to a JSON file ``{"KC": 11.5, ...}``.
    2. *api_key* is provided — fetch from The Odds API.
    3. Fall back to built-in defaults.
    """
    if source is not None:
        path = Path(source)
        if path.suffix == ".json":
            return json.loads(path.read_text())
        if path.suffix == ".csv":
            df = pd.read_csv(path)
            return dict(zip(df["team"], df["win_total"]))

    if api_key:
        return _fetch_win_totals_api(api_key, config)

    return dict(_DEFAULT_WIN_TOTALS)


def _fetch_win_totals_api(api_key: str, config: EngineConfig) -> dict[str, float]:
    """Fetch win totals from The Odds API."""
    url = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds/"
    params = {
        "apiKey": api_key,
        "regions": "us",
        "markets": "totals",
        "oddsFormat": "american",
    }
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    data: list[dict[str, Any]] = resp.json()

    totals: dict[str, float] = {}
    for game in data:
        for bm in game.get("bookmakers", []):
            for market in bm.get("markets", []):
                if market.get("key") == "totals":
                    for outcome in market.get("outcomes", []):
                        team = outcome.get("description", outcome.get("name", ""))
                        totals.setdefault(team, outcome.get("point", 8.5))
    return totals


# ---------------------------------------------------------------------------
# Player Season Props (delegated to data.props for richer fallbacks)
# ---------------------------------------------------------------------------

def load_player_props(
    source: str | Path | None = None,
    *,
    config: EngineConfig = DEFAULT_CONFIG,
) -> dict[str, float]:
    """Return ``{player_id: season_fantasy_points_total}``.

    Delegates to ``data.props`` which supports file loading and
    historical-season derivation as a fallback.
    """
    from fantasyquant.data.props import load_player_props as _load_props
    return _load_props(source, config=config)
