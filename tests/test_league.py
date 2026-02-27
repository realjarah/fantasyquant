"""Tests for league configuration loading, presets, and serialization."""

import json
import tempfile
from pathlib import Path

import pytest

from fantasyquant.config import EngineConfig
from fantasyquant.league import (
    PRESETS,
    league_to_dict,
    list_presets,
    load_league,
    save_league_template,
)


class TestPresets:
    def test_all_presets_load(self):
        """Every preset should produce a valid EngineConfig."""
        for name in list_presets():
            config = load_league(preset=name)
            assert isinstance(config, EngineConfig)
            assert config.roster.teams > 0
            assert config.roster.starters > 0

    def test_espn_ppr_scoring(self):
        config = load_league(preset="espn_ppr")
        assert config.scoring.receptions == 1.0
        assert config.scoring.reception_format == "PPR"
        assert config.platform == "espn"

    def test_yahoo_half_ppr_scoring(self):
        config = load_league(preset="yahoo_half_ppr")
        assert config.scoring.receptions == 0.5
        assert config.scoring.reception_format == "Half-PPR"

    def test_sleeper_superflex_has_superflex_slot(self):
        config = load_league(preset="sleeper_superflex")
        assert config.roster.superflex == 1
        assert config.roster.starters > config.roster.qb  # SF adds a starter slot

    def test_underdog_bestball_no_dst_no_k(self):
        config = load_league(preset="underdog_bestball")
        assert config.roster.dst == 0
        assert config.roster.k == 0
        assert config.roster.wr == 3  # best ball plays 3 WR

    def test_unknown_preset_raises(self):
        with pytest.raises(ValueError, match="Unknown preset"):
            load_league(preset="made_up_platform")


class TestLoadFromFile:
    def test_load_minimal_json(self, tmp_path):
        league_file = tmp_path / "league.json"
        league_file.write_text(json.dumps({
            "platform": "sleeper",
            "season": 2026,
            "scoring": {"receptions": 0.5},
            "roster": {"teams": 14},
        }))

        config = load_league(league_file)
        assert config.platform == "sleeper"
        assert config.current_season == 2026
        assert config.scoring.receptions == 0.5
        assert config.roster.teams == 14
        # Unspecified fields keep defaults.
        assert config.scoring.passing_tds == 4.0
        assert config.roster.qb == 1

    def test_file_with_preset_base(self, tmp_path):
        """File can reference a preset, then override specific fields."""
        league_file = tmp_path / "league.json"
        league_file.write_text(json.dumps({
            "preset": "espn_ppr",
            "roster": {"teams": 8},
        }))

        config = load_league(league_file)
        assert config.platform == "espn"
        assert config.scoring.receptions == 1.0  # from preset
        assert config.roster.teams == 8           # overridden

    def test_file_overrides_preset_arg(self, tmp_path):
        """Explicit file values override preset values."""
        league_file = tmp_path / "league.json"
        league_file.write_text(json.dumps({
            "scoring": {"receptions": 0.0},  # standard
        }))

        config = load_league(league_file, preset="espn_ppr")
        assert config.scoring.receptions == 0.0  # file wins

    def test_optimization_from_file(self, tmp_path):
        league_file = tmp_path / "league.json"
        league_file.write_text(json.dumps({
            "optimization": {
                "lambda_wins": 200.0,
                "alpha": 1.2,
                "playoff_weeks": [15, 16, 17],
            },
        }))

        config = load_league(league_file)
        assert config.optimization.lambda_wins == 200.0
        assert config.optimization.alpha == 1.2
        assert config.optimization.playoff_weeks == (15, 16, 17)


class TestSaveTemplate:
    def test_roundtrip(self, tmp_path):
        """Save a template, load it back, verify config matches."""
        out = tmp_path / "league.json"
        save_league_template(out, preset="sleeper_ppr")

        config = load_league(out)
        assert config.platform == "sleeper"
        assert config.scoring.receptions == 1.0
        assert config.roster.teams == 12

    def test_default_template(self, tmp_path):
        out = tmp_path / "default.json"
        save_league_template(out)

        config = load_league(out)
        assert isinstance(config, EngineConfig)


class TestLeagueToDict:
    def test_serializes_all_fields(self):
        config = load_league(preset="sleeper_superflex")
        data = league_to_dict(config)

        assert "platform" in data
        assert "scoring" in data
        assert "roster" in data
        assert data["roster"]["superflex"] == 1
        assert data["scoring"]["receptions"] == 1.0

    def test_json_serializable(self):
        config = EngineConfig()
        data = league_to_dict(config)
        # Should not raise.
        result = json.dumps(data)
        assert isinstance(result, str)
