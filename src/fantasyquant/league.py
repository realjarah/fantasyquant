"""League configuration: load from JSON and platform presets.

A user's league rules are the meta-input that calibrates everything:
scoring weights determine how raw stats become fantasy points, roster
slots determine positional scarcity, and the platform identity maps
to ADP populations and scoring defaults.

Usage::

    # From a JSON file
    config = load_league("my_league.json")

    # From a platform preset
    config = load_league(preset="sleeper")

    # Generate a template
    save_league_template("league.json", preset="espn")

Example ``league.json``::

    {
        "platform": "sleeper",
        "season": 2025,
        "scoring": {
            "receptions": 1.0,
            "passing_tds": 4
        },
        "roster": {
            "teams": 12,
            "qb": 1,
            "rb": 2,
            "wr": 2,
            "te": 1,
            "flex": 1,
            "superflex": 1,
            "bench": 6,
            "dst": 1,
            "k": 0
        }
    }
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fantasyquant.config import (
    EngineConfig,
    OptimizationConfig,
    PredictionConfig,
    RosterSettings,
    ScoringSettings,
)


# ---------------------------------------------------------------------------
# Platform presets
# ---------------------------------------------------------------------------

PRESETS: dict[str, dict[str, Any]] = {
    "espn_ppr": {
        "platform": "espn",
        "scoring": {
            "passing_yards": 0.04,
            "passing_tds": 4.0,
            "interceptions": -2.0,
            "rushing_yards": 0.1,
            "rushing_tds": 6.0,
            "receptions": 1.0,
            "receiving_yards": 0.1,
            "receiving_tds": 6.0,
            "fumbles_lost": -2.0,
        },
        "roster": {
            "teams": 10,
            "rounds": 16,
            "qb": 1, "rb": 2, "wr": 2, "te": 1,
            "flex": 1, "superflex": 0,
            "bench": 7, "dst": 1, "k": 1,
        },
    },
    "espn_half_ppr": {
        "platform": "espn",
        "scoring": {
            "passing_yards": 0.04,
            "passing_tds": 4.0,
            "interceptions": -2.0,
            "rushing_yards": 0.1,
            "rushing_tds": 6.0,
            "receptions": 0.5,
            "receiving_yards": 0.1,
            "receiving_tds": 6.0,
            "fumbles_lost": -2.0,
        },
        "roster": {
            "teams": 10,
            "rounds": 16,
            "qb": 1, "rb": 2, "wr": 2, "te": 1,
            "flex": 1, "superflex": 0,
            "bench": 7, "dst": 1, "k": 1,
        },
    },
    "yahoo_half_ppr": {
        "platform": "yahoo",
        "scoring": {
            "passing_yards": 0.04,
            "passing_tds": 4.0,
            "interceptions": -1.0,
            "rushing_yards": 0.1,
            "rushing_tds": 6.0,
            "receptions": 0.5,
            "receiving_yards": 0.1,
            "receiving_tds": 6.0,
            "fumbles_lost": -2.0,
        },
        "roster": {
            "teams": 10,
            "rounds": 15,
            "qb": 1, "rb": 2, "wr": 2, "te": 1,
            "flex": 1, "superflex": 0,
            "bench": 5, "dst": 1, "k": 1,
        },
    },
    "sleeper_ppr": {
        "platform": "sleeper",
        "scoring": {
            "passing_yards": 0.04,
            "passing_tds": 4.0,
            "interceptions": -1.0,
            "rushing_yards": 0.1,
            "rushing_tds": 6.0,
            "receptions": 1.0,
            "receiving_yards": 0.1,
            "receiving_tds": 6.0,
            "fumbles_lost": -2.0,
        },
        "roster": {
            "teams": 12,
            "rounds": 15,
            "qb": 1, "rb": 2, "wr": 2, "te": 1,
            "flex": 1, "superflex": 0,
            "bench": 6, "dst": 1, "k": 0,
        },
    },
    "sleeper_superflex": {
        "platform": "sleeper",
        "scoring": {
            "passing_yards": 0.04,
            "passing_tds": 4.0,
            "interceptions": -1.0,
            "rushing_yards": 0.1,
            "rushing_tds": 6.0,
            "receptions": 1.0,
            "receiving_yards": 0.1,
            "receiving_tds": 6.0,
            "fumbles_lost": -2.0,
        },
        "roster": {
            "teams": 12,
            "rounds": 15,
            "qb": 1, "rb": 2, "wr": 2, "te": 1,
            "flex": 1, "superflex": 1,
            "bench": 5, "dst": 1, "k": 0,
        },
    },
    "nfl_ppr": {
        "platform": "nfl",
        "scoring": {
            "passing_yards": 0.04,
            "passing_tds": 4.0,
            "interceptions": -2.0,
            "rushing_yards": 0.1,
            "rushing_tds": 6.0,
            "receptions": 1.0,
            "receiving_yards": 0.1,
            "receiving_tds": 6.0,
            "fumbles_lost": -2.0,
        },
        "roster": {
            "teams": 10,
            "rounds": 15,
            "qb": 1, "rb": 2, "wr": 2, "te": 1,
            "flex": 1, "superflex": 0,
            "bench": 6, "dst": 1, "k": 1,
        },
    },
    "underdog_bestball": {
        "platform": "underdog",
        "scoring": {
            "passing_yards": 0.04,
            "passing_tds": 4.0,
            "interceptions": -1.0,
            "rushing_yards": 0.1,
            "rushing_tds": 6.0,
            "receptions": 0.5,
            "receiving_yards": 0.1,
            "receiving_tds": 6.0,
            "fumbles_lost": -1.0,
        },
        "roster": {
            "teams": 12,
            "rounds": 18,
            "qb": 1, "rb": 2, "wr": 3, "te": 1,
            "flex": 1, "superflex": 0,
            "bench": 8, "dst": 0, "k": 0,
        },
    },
}


def list_presets() -> list[str]:
    """Return available preset names."""
    return sorted(PRESETS.keys())


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def _build_scoring(data: dict[str, Any]) -> ScoringSettings:
    """Build ScoringSettings from a dict, keeping defaults for missing keys."""
    valid_fields = {f.name for f in ScoringSettings.__dataclass_fields__.values()}
    filtered = {k: v for k, v in data.items() if k in valid_fields}
    return ScoringSettings(**filtered)


def _build_roster(data: dict[str, Any]) -> RosterSettings:
    """Build RosterSettings from a dict, keeping defaults for missing keys."""
    valid_fields = {f.name for f in RosterSettings.__dataclass_fields__.values()}
    filtered = {k: v for k, v in data.items() if k in valid_fields}
    return RosterSettings(**filtered)


def _build_optimization(data: dict[str, Any]) -> OptimizationConfig:
    """Build OptimizationConfig from a dict."""
    valid_fields = {f.name for f in OptimizationConfig.__dataclass_fields__.values()}
    filtered = {}
    for k, v in data.items():
        if k in valid_fields:
            # Handle tuple fields.
            if k == "playoff_weeks" and isinstance(v, list):
                filtered[k] = tuple(v)
            else:
                filtered[k] = v
    return OptimizationConfig(**filtered)


def _build_prediction(data: dict[str, Any]) -> PredictionConfig:
    """Build PredictionConfig from a dict."""
    valid_fields = {f.name for f in PredictionConfig.__dataclass_fields__.values()}
    filtered = {k: v for k, v in data.items() if k in valid_fields}
    return PredictionConfig(**filtered)


def load_league(
    path: str | Path | None = None,
    *,
    preset: str | None = None,
) -> EngineConfig:
    """Load league configuration from a JSON file and/or a platform preset.

    Priority: file values override preset values override defaults.

    Parameters
    ----------
    path:
        Path to a ``league.json`` file.
    preset:
        Name of a platform preset (e.g. ``"espn_ppr"``, ``"sleeper_superflex"``).

    Returns
    -------
    EngineConfig
    """
    # Start with empty or preset base.
    base: dict[str, Any] = {}
    if preset:
        if preset not in PRESETS:
            raise ValueError(
                f"Unknown preset '{preset}'. Available: {list_presets()}"
            )
        base = _deep_copy_dict(PRESETS[preset])

    # Layer file overrides on top.
    if path is not None:
        path = Path(path)
        with open(path) as f:
            file_data = json.load(f)
        # If the file specifies a preset, load that first as base.
        if "preset" in file_data and not preset:
            preset_name = file_data.pop("preset")
            if preset_name in PRESETS:
                base = _deep_copy_dict(PRESETS[preset_name])
        _deep_merge(base, file_data)

    return _dict_to_config(base)


def _dict_to_config(data: dict[str, Any]) -> EngineConfig:
    """Convert a merged dict into EngineConfig."""
    scoring = _build_scoring(data.get("scoring", {}))
    roster = _build_roster(data.get("roster", {}))
    prediction = _build_prediction(data.get("prediction", {}))
    optimization = _build_optimization(data.get("optimization", {}))

    return EngineConfig(
        scoring=scoring,
        roster=roster,
        prediction=prediction,
        optimization=optimization,
        nfl_weeks=data.get("nfl_weeks", 17),
        current_season=data.get("season", 2025),
        platform=data.get("platform", "custom"),
    )


def _deep_copy_dict(d: dict) -> dict:
    """Shallow-safe copy of a nested dict."""
    result = {}
    for k, v in d.items():
        if isinstance(v, dict):
            result[k] = _deep_copy_dict(v)
        elif isinstance(v, list):
            result[k] = v.copy()
        else:
            result[k] = v
    return result


def _deep_merge(base: dict, override: dict) -> None:
    """Merge *override* into *base* in place (override wins)."""
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v


# ---------------------------------------------------------------------------
# Template generation
# ---------------------------------------------------------------------------

def league_to_dict(config: EngineConfig) -> dict[str, Any]:
    """Serialize an EngineConfig to a JSON-friendly dict."""
    return {
        "platform": config.platform,
        "season": config.current_season,
        "scoring": {
            "passing_yards": config.scoring.passing_yards,
            "passing_tds": config.scoring.passing_tds,
            "interceptions": config.scoring.interceptions,
            "rushing_yards": config.scoring.rushing_yards,
            "rushing_tds": config.scoring.rushing_tds,
            "receptions": config.scoring.receptions,
            "receiving_yards": config.scoring.receiving_yards,
            "receiving_tds": config.scoring.receiving_tds,
            "fumbles_lost": config.scoring.fumbles_lost,
        },
        "roster": {
            "teams": config.roster.teams,
            "rounds": config.roster.rounds,
            "qb": config.roster.qb,
            "rb": config.roster.rb,
            "wr": config.roster.wr,
            "te": config.roster.te,
            "flex": config.roster.flex,
            "superflex": config.roster.superflex,
            "bench": config.roster.bench,
            "dst": config.roster.dst,
            "k": config.roster.k,
        },
    }


def save_league_template(
    path: str | Path,
    preset: str | None = None,
) -> Path:
    """Write a league.json template to *path*.

    If *preset* is given, populates with that platform's defaults.
    Otherwise uses the engine defaults.
    """
    path = Path(path)
    if preset:
        config = load_league(preset=preset)
    else:
        config = EngineConfig()

    data = league_to_dict(config)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")

    return path
