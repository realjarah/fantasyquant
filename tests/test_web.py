"""Tests for the FantasyQuant web API and session management."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from fantasyquant.config import EngineConfig
from fantasyquant.optimization.solver import PlayerPool
from fantasyquant.web.app import app
from fantasyquant.web.models import CreateSessionRequest
from fantasyquant.web.session import (
    DraftSession,
    SessionStatus,
    create_session,
    get_session,
    _sessions,
    _sessions_lock,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_pool(n_players: int = 40, n_weeks: int = 17) -> PlayerPool:
    """Build a synthetic player pool for testing."""
    positions = ["QB"] * 6 + ["RB"] * 14 + ["WR"] * 14 + ["TE"] * 6
    positions = positions[:n_players]
    pids = [f"P{i:03d}" for i in range(n_players)]
    names = [f"Player {i}" for i in range(n_players)]
    teams = [f"T{i % 8}" for i in range(n_players)]

    info = pd.DataFrame({
        "player_id": pids,
        "player_name": names,
        "position": positions,
        "team": teams,
    })

    # Random projections, higher for earlier players (mimics skill ordering).
    rng = np.random.RandomState(42)
    proj_data = rng.uniform(5, 20, size=(n_players, n_weeks))
    for i in range(n_players):
        proj_data[i] *= (n_players - i) / n_players  # skill decay
    proj = pd.DataFrame(proj_data, index=pids, columns=list(range(1, n_weeks + 1)))

    return PlayerPool(projections=proj, info=info)


@pytest.fixture(autouse=True)
def _clear_sessions():
    """Clear the global session store between tests."""
    with _sessions_lock:
        _sessions.clear()
    yield
    with _sessions_lock:
        _sessions.clear()


@pytest.fixture
def pool():
    return _make_pool()


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def ready_session(pool):
    """Create a session and initialize it with a pool (skip projections loading)."""
    config = EngineConfig(roster=EngineConfig().roster.__class__(teams=4, rounds=5))
    req = CreateSessionRequest(my_slot=1)
    session = create_session(req, config)
    session.initialize(pool)
    return session


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------

class TestSession:
    def test_create_and_retrieve(self):
        config = EngineConfig()
        req = CreateSessionRequest(my_slot=3)
        session = create_session(req, config)
        assert session.session_id
        assert session.my_slot == 3
        assert session.status == SessionStatus.LOADING

        retrieved = get_session(session.session_id)
        assert retrieved is session

    def test_initialize(self, pool):
        config = EngineConfig()
        req = CreateSessionRequest(my_slot=1)
        session = create_session(req, config)
        session.initialize(pool)
        assert session.status == SessionStatus.READY
        assert session.loop is not None

    def test_opponent_pick(self, ready_session):
        s = ready_session
        # First pick is ours (slot 1), so skip to someone else's turn.
        # In a 4-team league, slot 1 picks first. After our pick, it's others.
        assert s.is_my_turn  # pick 1 = slot 1

        # Make our pick first.
        entry, _ = s.record_my_pick("P000")
        assert entry is not None
        assert entry.is_mine

        # Now it's an opponent's turn.
        assert not s.is_my_turn
        opp_entry = s.record_opponent_pick("P001")
        assert opp_entry is not None
        assert not opp_entry.is_mine

    def test_available_players(self, ready_session):
        s = ready_session
        players = s.get_available_players(limit=10)
        assert len(players) == 10

        # After taking a player, they shouldn't appear.
        s.record_my_pick("P000")
        players_after = s.get_available_players(limit=50)
        ids = [p.player_id for p in players_after]
        assert "P000" not in ids

    def test_filter_by_position(self, ready_session):
        s = ready_session
        qbs = s.get_available_players(position="QB")
        for p in qbs:
            assert p.position == "QB"

    def test_search_players(self, ready_session):
        s = ready_session
        results = s.get_available_players(search="Player 5")
        assert any("Player 5" in p.player_name for p in results)


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

class TestAPI:
    def test_landing_page(self, client):
        res = client.get("/")
        assert res.status_code == 200
        assert "FantasyQuant" in res.text

    def test_presets_api(self, client):
        res = client.get("/api/presets")
        assert res.status_code == 200
        data = res.json()
        assert "espn_ppr" in data
        assert "sleeper_superflex" in data

    def test_create_session_api(self, client):
        res = client.post("/api/sessions", json={
            "preset": "espn_ppr",
            "my_slot": 3,
        })
        assert res.status_code == 200
        data = res.json()
        assert "session_id" in data
        assert data["status"] == "loading"

    def test_create_session_custom(self, client):
        res = client.post("/api/sessions", json={
            "my_slot": 1,
            "scoring": {"receptions": 0.5, "passing_tds": 6},
            "roster": {"teams": 8, "rounds": 10},
        })
        assert res.status_code == 200
        data = res.json()
        assert "session_id" in data

    def test_get_state(self, client, ready_session):
        sid = ready_session.session_id
        res = client.get(f"/api/sessions/{sid}/state")
        assert res.status_code == 200
        data = res.json()
        assert data["session_id"] == sid
        assert data["status"] == "ready"
        assert data["current_pick"] == 1
        assert data["is_my_turn"] is True

    def test_make_pick_by_id(self, client, ready_session):
        sid = ready_session.session_id
        res = client.post(f"/api/sessions/{sid}/pick", json={"player_id": "P000"})
        assert res.status_code == 200
        data = res.json()
        assert data["success"] is True
        assert data["pick"]["player_id"] == "P000"
        assert data["pick"]["is_mine"] is True

    def test_make_pick_by_name(self, client, ready_session):
        sid = ready_session.session_id
        res = client.post(f"/api/sessions/{sid}/pick", json={"player_name": "Player 0"})
        assert res.status_code == 200
        data = res.json()
        assert data["success"] is True

    def test_duplicate_pick_rejected(self, client, ready_session):
        sid = ready_session.session_id
        client.post(f"/api/sessions/{sid}/pick", json={"player_id": "P000"})
        # Try to pick same player again.
        res = client.post(f"/api/sessions/{sid}/pick", json={"player_id": "P000"})
        data = res.json()
        assert data["success"] is False
        assert "already" in data["error"].lower()

    def test_player_pool(self, client, ready_session):
        sid = ready_session.session_id
        res = client.get(f"/api/sessions/{sid}/players?limit=5")
        assert res.status_code == 200
        data = res.json()
        assert len(data["players"]) == 5

    def test_player_pool_filter(self, client, ready_session):
        sid = ready_session.session_id
        res = client.get(f"/api/sessions/{sid}/players?position=QB")
        assert res.status_code == 200
        data = res.json()
        for p in data["players"]:
            assert p["position"] == "QB"

    def test_draft_page(self, client, ready_session):
        sid = ready_session.session_id
        res = client.get(f"/draft/{sid}")
        assert res.status_code == 200
        assert "draft-app" in res.text

    def test_404_unknown_session(self, client):
        res = client.get("/api/sessions/nonexistent/state")
        assert res.status_code == 404

    def test_summary_api(self, client, ready_session):
        sid = ready_session.session_id
        # Make a few picks to have a roster.
        ready_session.record_my_pick("P000")
        ready_session.record_opponent_pick("P001")
        ready_session.record_opponent_pick("P002")
        ready_session.record_opponent_pick("P003")

        res = client.get(f"/api/sessions/{sid}/summary")
        assert res.status_code == 200
        data = res.json()
        assert data["session_id"] == sid
        assert len(data["roster"]) >= 1
        assert data["total_projected_points"] > 0

    def test_pay_dev_mode(self, client, ready_session):
        sid = ready_session.session_id
        res = client.post(f"/api/sessions/{sid}/pay")
        assert res.status_code == 200
        data = res.json()
        assert f"/draft/{sid}" in data["checkout_url"]
