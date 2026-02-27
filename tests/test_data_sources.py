"""Tests for ADP, player props, and Vegas data source loaders."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from fantasyquant.config import EngineConfig, RosterSettings, ScoringSettings


# -----------------------------------------------------------------------
# ADP loader tests
# -----------------------------------------------------------------------


class TestLoadAdp:
    """Tests for data.adp.load_adp()."""

    def test_load_from_csv(self, tmp_path):
        from fantasyquant.data.adp import load_adp

        csv = tmp_path / "adp.csv"
        csv.write_text("player_id,adp\nA,1.5\nB,5.2\nC,12.0\n")
        result = load_adp(str(csv))
        assert isinstance(result, pd.Series)
        assert result.name == "adp"
        assert len(result) == 3
        assert float(result["A"]) == 1.5
        assert float(result["C"]) == 12.0

    def test_load_from_json(self, tmp_path):
        from fantasyquant.data.adp import load_adp

        j = tmp_path / "adp.json"
        j.write_text(json.dumps({"P1": 2.0, "P2": 8.5}))
        result = load_adp(str(j))
        assert isinstance(result, pd.Series)
        assert float(result["P1"]) == 2.0
        assert float(result["P2"]) == 8.5

    def test_load_from_csv_alternative_columns(self, tmp_path):
        from fantasyquant.data.adp import load_adp

        csv = tmp_path / "adp.csv"
        csv.write_text("id,avg_pick\nX,3.0\nY,7.0\n")
        result = load_adp(str(csv))
        assert len(result) == 2
        assert float(result["X"]) == 3.0

    def test_invalid_csv_raises(self, tmp_path):
        from fantasyquant.data.adp import load_adp

        csv = tmp_path / "bad.csv"
        csv.write_text("name,score\nfoo,10\n")
        with pytest.raises(ValueError, match="Cannot parse ADP"):
            load_adp(str(csv))

    def test_derive_from_projections(self):
        from fantasyquant.data.adp import _derive_from_projections

        proj = pd.DataFrame(
            {"w1": [100, 50, 80], "w2": [110, 60, 70]},
            index=["A", "B", "C"],
        )
        result = _derive_from_projections(proj)
        assert result is not None
        assert len(result) == 3
        # A has highest total (210), so ADP = 1.
        assert float(result["A"]) == 1.0
        # B has lowest total (110), so ADP = 3.
        assert float(result["B"]) == 3.0

    def test_returns_none_when_no_data(self):
        from fantasyquant.data.adp import load_adp

        result = load_adp()
        # Without nfl_data_py and no projections, may return None.
        # (nfl_data_py may succeed in the test environment, so we just
        # check it doesn't crash.)
        assert result is None or isinstance(result, pd.Series)


# -----------------------------------------------------------------------
# Live ADP fetcher tests
# -----------------------------------------------------------------------


class TestScoringToFfcFormat:
    """Tests that scoring/roster settings map to correct FFC API format."""

    def test_ppr(self):
        from fantasyquant.data.adp import _scoring_to_ffc_format

        config = EngineConfig(scoring=ScoringSettings(receptions=1.0))
        assert _scoring_to_ffc_format(config) == "ppr"

    def test_half_ppr(self):
        from fantasyquant.data.adp import _scoring_to_ffc_format

        config = EngineConfig(scoring=ScoringSettings(receptions=0.5))
        assert _scoring_to_ffc_format(config) == "half-ppr"

    def test_standard(self):
        from fantasyquant.data.adp import _scoring_to_ffc_format

        config = EngineConfig(scoring=ScoringSettings(receptions=0.0))
        assert _scoring_to_ffc_format(config) == "standard"

    def test_superflex_overrides_ppr(self):
        from fantasyquant.data.adp import _scoring_to_ffc_format

        config = EngineConfig(
            scoring=ScoringSettings(receptions=1.0),
            roster=RosterSettings(superflex=1),
        )
        assert _scoring_to_ffc_format(config) == "superflex"

    def test_two_qb(self):
        from fantasyquant.data.adp import _scoring_to_ffc_format

        config = EngineConfig(roster=RosterSettings(qb=2))
        assert _scoring_to_ffc_format(config) == "2qb"


class TestFfcTeams:
    """Tests for team count snapping to FFC-supported values."""

    def test_snap_to_nearest(self):
        from fantasyquant.data.adp import _ffc_teams

        assert _ffc_teams(EngineConfig(roster=RosterSettings(teams=8))) == 8
        assert _ffc_teams(EngineConfig(roster=RosterSettings(teams=10))) == 10
        assert _ffc_teams(EngineConfig(roster=RosterSettings(teams=12))) == 12
        assert _ffc_teams(EngineConfig(roster=RosterSettings(teams=14))) == 14

    def test_odd_sizes_snap_up(self):
        from fantasyquant.data.adp import _ffc_teams

        assert _ffc_teams(EngineConfig(roster=RosterSettings(teams=9))) == 10
        assert _ffc_teams(EngineConfig(roster=RosterSettings(teams=11))) == 12
        assert _ffc_teams(EngineConfig(roster=RosterSettings(teams=6))) == 8

    def test_large_leagues_cap_at_14(self):
        from fantasyquant.data.adp import _ffc_teams

        assert _ffc_teams(EngineConfig(roster=RosterSettings(teams=16))) == 14
        assert _ffc_teams(EngineConfig(roster=RosterSettings(teams=20))) == 14


class TestNormalizeName:
    """Tests for player name normalization used in ADP matching."""

    def test_basic(self):
        from fantasyquant.data.adp import _normalize_name

        assert _normalize_name("Patrick Mahomes") == "patrick mahomes"
        assert _normalize_name("Patrick Mahomes II") == "patrick mahomes"
        assert _normalize_name("Travis Kelce") == "travis kelce"

    def test_suffixes_stripped(self):
        from fantasyquant.data.adp import _normalize_name

        assert _normalize_name("Marvin Harrison Jr.") == "marvin harrison"
        assert _normalize_name("Odell Beckham Jr") == "odell beckham"
        assert _normalize_name("Robert Griffin III") == "robert griffin"

    def test_punctuation(self):
        from fantasyquant.data.adp import _normalize_name

        assert _normalize_name("Ja'Marr Chase") == "jamarr chase"
        assert _normalize_name("D.K. Metcalf") == "dk metcalf"
        assert _normalize_name("De'Von Achane") == "devon achane"


class TestLiveAdpFetch:
    """Integration test for live ADP (may fail if API is unreachable)."""

    def test_fetch_returns_series_or_none(self):
        from fantasyquant.data.adp import _fetch_live_adp

        config = EngineConfig(scoring=ScoringSettings(receptions=1.0))
        result = _fetch_live_adp(config)
        # May return None if API is down or no data for current year.
        if result is not None:
            assert isinstance(result, pd.Series)
            assert result.name == "adp"
            assert len(result) > 0

    def test_caching_works(self):
        from fantasyquant.data.adp import _ADP_CACHE, _fetch_live_adp, _cache_key, _scoring_to_ffc_format, _ffc_teams

        config = EngineConfig(scoring=ScoringSettings(receptions=1.0))
        fmt = _scoring_to_ffc_format(config)
        teams = _ffc_teams(config)
        key = _cache_key(fmt, teams, config.current_season)

        # Clear cache.
        _ADP_CACHE.pop(key, None)

        first = _fetch_live_adp(config)
        if first is not None:
            # Second call should use cache.
            assert key in _ADP_CACHE
            second = _fetch_live_adp(config)
            assert second is not None


class TestLoadAdpPriorityChain:
    """Test the full priority chain: file > live > VOR > projections."""

    def test_file_takes_priority(self, tmp_path):
        from fantasyquant.data.adp import load_adp

        csv = tmp_path / "adp.csv"
        csv.write_text("player_id,adp\nA,1.0\nB,2.0\n")
        result = load_adp(str(csv))
        assert len(result) == 2
        assert float(result["A"]) == 1.0

    def test_without_file_returns_something(self):
        from fantasyquant.data.adp import load_adp

        proj = pd.DataFrame(
            {"w1": [100, 50], "w2": [110, 60]},
            index=["X", "Y"],
        )
        result = load_adp(projections=proj)
        # Should get data from live, VOR, or projection fallback.
        assert result is not None
        assert len(result) >= 2


# -----------------------------------------------------------------------
# VOR-based ADP tests (scoring + roster aware)
# -----------------------------------------------------------------------


class TestReplacementLevel:
    """Tests for _replacement_level() — the positional scarcity model."""

    def test_standard_1qb_league(self):
        from fantasyquant.data.adp import _replacement_level

        config = EngineConfig(roster=RosterSettings(
            teams=12, qb=1, rb=2, wr=2, te=1, flex=1, superflex=0,
        ))
        repl = _replacement_level(config)
        # Standard 1QB: 12 teams × 1 QB = 12 + 1 = 13th QB is replacement.
        assert repl["QB"] == pytest.approx(13.0, abs=0.1)
        # RBs: 12 teams × (2 + 0.5 flex) = 30 + 1 = 31st RB.
        assert repl["RB"] == pytest.approx(31.0, abs=0.1)

    def test_superflex_boosts_qb_replacement(self):
        from fantasyquant.data.adp import _replacement_level

        standard = EngineConfig(roster=RosterSettings(
            teams=12, qb=1, rb=2, wr=2, te=1, flex=1, superflex=0,
        ))
        superflex = EngineConfig(roster=RosterSettings(
            teams=12, qb=1, rb=2, wr=2, te=1, flex=1, superflex=1,
        ))
        repl_std = _replacement_level(standard)
        repl_sf = _replacement_level(superflex)

        # In superflex, QB replacement level should be MUCH higher
        # (more QBs are starters → replacement player is deeper).
        assert repl_sf["QB"] > repl_std["QB"]
        # Roughly: 12 × (1 + 0.7) = 20.4 + 1 = ~21.4 for superflex.
        assert repl_sf["QB"] == pytest.approx(21.4, abs=0.5)

    def test_deep_league_shifts_replacement(self):
        from fantasyquant.data.adp import _replacement_level

        ten_team = EngineConfig(roster=RosterSettings(teams=10, qb=1, rb=2, wr=2, te=1, flex=1))
        fourteen_team = EngineConfig(roster=RosterSettings(teams=14, qb=1, rb=2, wr=2, te=1, flex=1))
        repl_10 = _replacement_level(ten_team)
        repl_14 = _replacement_level(fourteen_team)

        # All replacement levels should increase with more teams.
        for pos in ["QB", "RB", "WR", "TE"]:
            assert repl_14[pos] > repl_10[pos], f"{pos} should increase in deeper leagues"

    def test_two_qb_league(self):
        from fantasyquant.data.adp import _replacement_level

        config = EngineConfig(roster=RosterSettings(
            teams=12, qb=2, rb=2, wr=2, te=1, flex=1, superflex=0,
        ))
        repl = _replacement_level(config)
        # 2QB: 12 × 2 = 24 + 1 = 25th QB is replacement.
        assert repl["QB"] == pytest.approx(25.0, abs=0.1)


class TestVorAdp:
    """Tests that VOR-based ADP respects scoring format."""

    def test_vor_adp_returns_series(self):
        from fantasyquant.data.adp import _derive_vor_adp

        config = EngineConfig()
        result = _derive_vor_adp(config)
        # May return None if nfl_data_py unavailable.
        if result is not None:
            assert isinstance(result, pd.Series)
            assert result.name == "adp"
            assert len(result) > 0
            # ADP ranks should start at 1.
            assert float(result.iloc[0]) == 1.0

    def test_vor_adp_different_for_ppr_vs_standard(self):
        """PPR and Standard should produce different ADP rankings."""
        from fantasyquant.data.adp import _derive_vor_adp

        ppr_config = EngineConfig(scoring=ScoringSettings(receptions=1.0))
        std_config = EngineConfig(scoring=ScoringSettings(receptions=0.0))

        ppr_adp = _derive_vor_adp(ppr_config)
        std_adp = _derive_vor_adp(std_config)

        if ppr_adp is None or std_adp is None:
            pytest.skip("nfl_data_py not available")

        # The rankings should not be identical.
        # In PPR, pass-catching RBs/WRs rise; in standard, rushing-heavy RBs rise.
        common = set(ppr_adp.index) & set(std_adp.index)
        assert len(common) > 50  # Should have many players in common.

        # At least some players should have different ranks.
        differences = sum(
            1 for pid in common
            if abs(float(ppr_adp[pid]) - float(std_adp[pid])) > 2
        )
        assert differences > 10, "PPR vs Standard should produce meaningfully different ADP"


# -----------------------------------------------------------------------
# Player props loader tests
# -----------------------------------------------------------------------


class TestLoadPlayerProps:
    """Tests for data.props.load_player_props()."""

    def test_load_from_csv(self, tmp_path):
        from fantasyquant.data.props import load_player_props

        csv = tmp_path / "props.csv"
        csv.write_text("player_id,season_total\nA,250.5\nB,180.0\n")
        result = load_player_props(str(csv))
        assert isinstance(result, dict)
        assert result["A"] == 250.5
        assert result["B"] == 180.0

    def test_load_from_json(self, tmp_path):
        from fantasyquant.data.props import load_player_props

        j = tmp_path / "props.json"
        j.write_text(json.dumps({"P1": 300.0, "P2": 150.0}))
        result = load_player_props(str(j))
        assert result["P1"] == 300.0

    def test_csv_alternative_columns(self, tmp_path):
        from fantasyquant.data.props import load_player_props

        csv = tmp_path / "props.csv"
        csv.write_text("gsis_id,fpts\nX,200.0\nY,175.0\n")
        result = load_player_props(str(csv))
        assert result["X"] == 200.0

    def test_invalid_csv_raises(self, tmp_path):
        from fantasyquant.data.props import load_player_props

        csv = tmp_path / "bad.csv"
        csv.write_text("name,score\nfoo,10\n")
        with pytest.raises(ValueError, match="Cannot parse props"):
            load_player_props(str(csv))

    def test_returns_dict_without_source(self):
        from fantasyquant.data.props import load_player_props

        result = load_player_props()
        assert isinstance(result, dict)
        # May have data from nfl_data_py or be empty — both are valid.


# -----------------------------------------------------------------------
# Vegas loader tests (existing module, updated delegation)
# -----------------------------------------------------------------------


class TestVegasIntegration:
    """Tests for vegas.py load_player_props delegation to data.props."""

    def test_load_player_props_delegation(self, tmp_path):
        from fantasyquant.data.vegas import load_player_props

        csv = tmp_path / "props.csv"
        csv.write_text("player_id,season_total\nA,250.5\n")
        result = load_player_props(str(csv))
        assert result["A"] == 250.5

    def test_load_win_totals_defaults(self):
        from fantasyquant.data.vegas import load_win_totals

        result = load_win_totals()
        assert isinstance(result, dict)
        assert "KC" in result
        assert len(result) == 32

    def test_load_win_totals_from_json(self, tmp_path):
        from fantasyquant.data.vegas import load_win_totals

        j = tmp_path / "wt.json"
        j.write_text(json.dumps({"KC": 12.0, "BUF": 10.5}))
        result = load_win_totals(str(j))
        assert result["KC"] == 12.0


# -----------------------------------------------------------------------
# Pipeline integration tests
# -----------------------------------------------------------------------


class TestProjectionOutputAdp:
    """Tests that build_projections returns ADP data."""

    def test_projection_output_has_adp_field(self):
        from fantasyquant.prediction.projections import ProjectionOutput

        # Verify the dataclass has the adp field.
        import dataclasses
        fields = {f.name for f in dataclasses.fields(ProjectionOutput)}
        assert "adp" in fields


# -----------------------------------------------------------------------
# Web model tests
# -----------------------------------------------------------------------


class TestCreateSessionRequestDataSources:
    """Tests that CreateSessionRequest accepts data source fields."""

    def test_accepts_data_source_fields(self):
        from fantasyquant.web.models import CreateSessionRequest

        req = CreateSessionRequest(
            preset="espn_ppr",
            my_slot=3,
            odds_api_key="test-key-123",
            adp_source="/tmp/adp.csv",
            win_totals_source="/tmp/wt.json",
            player_props_source="/tmp/props.csv",
        )
        assert req.odds_api_key == "test-key-123"
        assert req.adp_source == "/tmp/adp.csv"
        assert req.win_totals_source == "/tmp/wt.json"
        assert req.player_props_source == "/tmp/props.csv"

    def test_data_source_fields_optional(self):
        from fantasyquant.web.models import CreateSessionRequest

        req = CreateSessionRequest(my_slot=1)
        assert req.odds_api_key is None
        assert req.adp_source is None
        assert req.win_totals_source is None
        assert req.player_props_source is None


# -----------------------------------------------------------------------
# Web upload endpoint tests
# -----------------------------------------------------------------------


class TestUploadEndpoints:
    """Tests for file upload API endpoints."""

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from fantasyquant.web.app import app
        return TestClient(app)

    def test_upload_adp_csv(self, client, tmp_path):
        csv_content = b"player_id,adp\nA,1.5\nB,5.2\n"
        response = client.post(
            "/api/upload/adp",
            files={"file": ("adp.csv", csv_content, "text/csv")},
        )
        assert response.status_code == 200
        data = response.json()
        assert "adp_source" in data
        assert data["filename"] == "adp.csv"
        # Verify file was actually saved.
        assert Path(data["adp_source"]).exists()

    def test_upload_win_totals(self, client):
        content = json.dumps({"KC": 11.5, "SF": 10.5}).encode()
        response = client.post(
            "/api/upload/win-totals",
            files={"file": ("wt.json", content, "application/json")},
        )
        assert response.status_code == 200
        data = response.json()
        assert "win_totals_source" in data

    def test_upload_props(self, client):
        csv_content = b"player_id,season_total\nA,250.5\n"
        response = client.post(
            "/api/upload/props",
            files={"file": ("props.csv", csv_content, "text/csv")},
        )
        assert response.status_code == 200
        data = response.json()
        assert "player_props_source" in data

    def test_data_sources_info(self, client):
        response = client.get("/api/data-sources")
        assert response.status_code == 200
        data = response.json()
        assert "adp" in data
        assert "win_totals" in data
        assert "player_props" in data
        assert "csv_columns" in data["adp"]

    def test_create_session_with_data_sources(self, client):
        response = client.post("/api/sessions", json={
            "my_slot": 1,
            "odds_api_key": "test-key",
        })
        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data


# -----------------------------------------------------------------------
# Session data source storage tests
# -----------------------------------------------------------------------


class TestSessionDataSources:
    """Tests that DraftSession stores data source params."""

    def test_session_stores_data_sources(self):
        from fantasyquant.web.models import CreateSessionRequest
        from fantasyquant.web.session import create_session

        req = CreateSessionRequest(
            my_slot=1,
            odds_api_key="key123",
            adp_source="/tmp/adp.csv",
            win_totals_source="/tmp/wt.json",
            player_props_source="/tmp/props.csv",
        )
        config = EngineConfig()
        session = create_session(req, config)

        assert session.odds_api_key == "key123"
        assert session.adp_source == "/tmp/adp.csv"
        assert session.win_totals_source == "/tmp/wt.json"
        assert session.player_props_source == "/tmp/props.csv"
