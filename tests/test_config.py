"""Tests for configuration defaults."""

from fantasyquant.config import DEFAULT_CONFIG, EngineConfig, RosterSettings


class TestRosterSettings:
    def test_total_slots(self):
        r = RosterSettings()
        expected = r.qb + r.rb + r.wr + r.te + r.flex + r.bench + r.dst + r.k
        assert r.total_slots == expected

    def test_starters(self):
        r = RosterSettings()
        expected = r.qb + r.rb + r.wr + r.te + r.flex + r.dst + r.k
        assert r.starters == expected


class TestEngineConfig:
    def test_defaults(self):
        cfg = DEFAULT_CONFIG
        assert cfg.nfl_weeks == 17
        assert cfg.prediction.training_seasons == 3
        assert cfg.prediction.historical_weight + cfg.prediction.vegas_weight == 1.0
