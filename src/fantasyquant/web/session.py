"""In-memory draft session manager.

Each paid draft creates a DraftSession that holds all state in memory.
No database, no persistence — when the session ends, the data is gone.
"""

from __future__ import annotations

import secrets
import threading
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd

from fantasyquant.config import EngineConfig
from fantasyquant.league import describe_config, load_league
from fantasyquant.optimization.draft_loop import DraftLoop
from fantasyquant.optimization.solver import DraftState, PlayerPool, Recommendation
from fantasyquant.web.models import (
    CreateSessionRequest,
    PickRecord,
    PlayerInfo,
    RosterPlayer,
)


class SessionStatus(str, Enum):
    LOADING = "loading"
    READY = "ready"
    DRAFTING = "drafting"
    COMPLETE = "complete"


@dataclass
class PickEntry:
    overall_pick: int
    round: int
    pick_in_round: int
    player_id: str
    is_mine: bool


@dataclass
class DraftSession:
    """Stateful draft session wrapping the engine's DraftLoop."""

    session_id: str
    config: EngineConfig
    my_slot: int
    status: SessionStatus = SessionStatus.LOADING
    loop: DraftLoop | None = None
    picks: list[PickEntry] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    # Data source params (stored so background thread can use them).
    odds_api_key: str | None = None
    adp_source: str | None = None
    win_totals_source: str | None = None
    player_props_source: str | None = None

    @property
    def total_teams(self) -> int:
        return self.config.roster.teams

    @property
    def total_rounds(self) -> int:
        return self.config.roster.rounds

    @property
    def current_pick(self) -> int:
        if self.loop is None:
            return 1
        return self.loop.state.current_pick

    @property
    def current_round(self) -> int:
        return (self.current_pick - 1) // self.total_teams + 1

    @property
    def pick_in_round(self) -> int:
        return (self.current_pick - 1) % self.total_teams + 1

    @property
    def is_my_turn(self) -> bool:
        if self.loop is None:
            return False
        return self.loop._is_my_pick

    @property
    def is_complete(self) -> bool:
        if self.loop is None:
            return False
        return self.loop.state.my_roster_size >= self.total_rounds

    @property
    def config_summary(self) -> str:
        return describe_config(self.config)

    def initialize(self, pool: PlayerPool) -> None:
        """Set up the draft loop with a player pool (called after projections load)."""
        with self._lock:
            self.loop = DraftLoop(pool, self.config, my_slot=self.my_slot)
            self.status = SessionStatus.READY

    def get_recommendation(self) -> Recommendation | None:
        """Run the solver for the current pick."""
        if self.loop is None:
            return None
        with self._lock:
            return self.loop.solver.solve(self.loop.state)

    def record_opponent_pick(self, player_id: str) -> PickEntry | None:
        """Record an opponent's pick and advance the draft."""
        if self.loop is None:
            return None
        with self._lock:
            if self.is_my_turn:
                return None  # Can't record opponent pick on our turn.
            if self.is_complete:
                return None

            self.status = SessionStatus.DRAFTING
            pick_num = self.current_pick
            rnd = self.current_round
            pos_in_rnd = self.pick_in_round

            self.loop.state.taken.add(player_id)
            self.loop.state.current_pick += 1

            entry = PickEntry(
                overall_pick=pick_num,
                round=rnd,
                pick_in_round=pos_in_rnd,
                player_id=player_id,
                is_mine=False,
            )
            self.picks.append(entry)

            if self.is_complete:
                self.status = SessionStatus.COMPLETE

            return entry

    def record_my_pick(self, player_id: str | None = None) -> tuple[PickEntry | None, Recommendation | None]:
        """Make our pick — use solver recommendation or override with player_id.

        Returns (pick_entry, recommendation).
        """
        if self.loop is None:
            return None, None
        with self._lock:
            if not self.is_my_turn:
                return None, None
            if self.is_complete:
                return None, None

            self.status = SessionStatus.DRAFTING
            pick_num = self.current_pick
            rnd = self.current_round
            pos_in_rnd = self.pick_in_round

            if player_id is not None:
                # Manual override — user chose a specific player.
                rec = None
                self.loop.state.my_picks.append(player_id)
                self.loop.state.taken.add(player_id)
                self.loop.state.current_pick += 1
            else:
                # Use the solver.
                rec = self.loop.make_my_pick()
                if rec is None:
                    return None, None
                player_id = rec.player_id

            entry = PickEntry(
                overall_pick=pick_num,
                round=rnd,
                pick_in_round=pos_in_rnd,
                player_id=player_id,
                is_mine=True,
            )
            self.picks.append(entry)

            if self.is_complete:
                self.status = SessionStatus.COMPLETE

            return entry, rec

    def get_player_info(self, player_id: str) -> dict | None:
        """Look up player metadata."""
        if self.loop is None:
            return None
        info = self.loop.pool.info
        mask = info["player_id"] == player_id
        if not mask.any():
            return None
        row = info[mask].iloc[0]
        return {
            "player_id": player_id,
            "player_name": str(row["player_name"]),
            "position": str(row["position"]),
            "team": str(row["team"]),
        }

    def get_available_players(
        self,
        position: str | None = None,
        search: str | None = None,
        limit: int = 50,
    ) -> list[PlayerInfo]:
        """Return available (undrafted) players, optionally filtered."""
        if self.loop is None:
            return []

        info = self.loop.pool.info
        proj = self.loop.pool.projections
        taken = self.loop.state.taken

        # Filter to available.
        mask = ~info["player_id"].isin(taken)
        if position:
            mask = mask & (info["position"] == position.upper())
        if search:
            mask = mask & info["player_name"].str.contains(search, case=False, na=False)

        available = info[mask].copy()

        # Add total projected points.
        totals = proj.sum(axis=1)
        available["total_projected"] = available["player_id"].map(
            lambda pid: float(totals.get(pid, 0.0))
        )

        # Add ADP if available.
        adp = self.loop.pool.adp
        if adp is not None:
            available["adp"] = available["player_id"].map(
                lambda pid: float(adp.get(pid)) if pid in adp.index else None
            )
        else:
            available["adp"] = None

        # Sort by total projected descending.
        available = available.sort_values("total_projected", ascending=False)

        players = []
        for _, row in available.head(limit).iterrows():
            players.append(PlayerInfo(
                player_id=str(row["player_id"]),
                player_name=str(row["player_name"]),
                position=str(row["position"]),
                team=str(row["team"]),
                total_projected=float(row["total_projected"]),
                adp=row.get("adp"),
            ))

        return players

    def get_my_roster(self) -> list[RosterPlayer]:
        """Return the user's current roster with pick metadata."""
        if self.loop is None:
            return []

        proj = self.loop.pool.projections
        roster = []
        for entry in self.picks:
            if not entry.is_mine:
                continue
            pinfo = self.get_player_info(entry.player_id)
            if pinfo is None:
                continue
            total = float(proj.sum(axis=1).get(entry.player_id, 0.0))
            roster.append(RosterPlayer(
                player_id=pinfo["player_id"],
                player_name=pinfo["player_name"],
                position=pinfo["position"],
                team=pinfo["team"],
                pick_number=entry.overall_pick,
                total_projected=total,
            ))
        return roster

    def get_roster_needs(self) -> dict[str, int]:
        """How many more of each position the user needs."""
        r = self.config.roster
        current: dict[str, int] = {"QB": 0, "RB": 0, "WR": 0, "TE": 0}
        for entry in self.picks:
            if not entry.is_mine:
                continue
            pinfo = self.get_player_info(entry.player_id)
            if pinfo and pinfo["position"] in current:
                current[pinfo["position"]] += 1

        needs: dict[str, int] = {}
        # Minimum starters per position.
        mins = {"QB": r.qb, "RB": r.rb, "WR": r.wr, "TE": r.te}
        for pos, mn in mins.items():
            remaining = max(0, mn - current[pos])
            if remaining > 0:
                needs[pos] = remaining
        return needs

    def to_pick_record(self, entry: PickEntry) -> PickRecord:
        """Convert internal PickEntry to API PickRecord."""
        pinfo = self.get_player_info(entry.player_id) or {
            "player_id": entry.player_id,
            "player_name": entry.player_id,
            "position": "??",
            "team": "??",
        }
        return PickRecord(
            overall_pick=entry.overall_pick,
            round=entry.round,
            pick_in_round=entry.pick_in_round,
            player_id=pinfo["player_id"],
            player_name=pinfo["player_name"],
            position=pinfo["position"],
            team=pinfo["team"],
            is_mine=entry.is_mine,
        )


# ---------------------------------------------------------------------------
# Global session store (in-memory, no persistence)
# ---------------------------------------------------------------------------

_sessions: dict[str, DraftSession] = {}
_sessions_lock = threading.Lock()


def create_session(request: CreateSessionRequest, config: EngineConfig) -> DraftSession:
    """Create a new draft session and store it."""
    session_id = secrets.token_urlsafe(16)
    session = DraftSession(
        session_id=session_id,
        config=config,
        my_slot=request.my_slot,
        odds_api_key=request.odds_api_key,
        adp_source=request.adp_source,
        win_totals_source=request.win_totals_source,
        player_props_source=request.player_props_source,
    )
    with _sessions_lock:
        _sessions[session_id] = session
    return session


def get_session(session_id: str) -> DraftSession | None:
    """Retrieve a session by ID."""
    with _sessions_lock:
        return _sessions.get(session_id)


def remove_session(session_id: str) -> None:
    """Remove a session (cleanup)."""
    with _sessions_lock:
        _sessions.pop(session_id, None)
