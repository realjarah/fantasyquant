"""Pydantic models for the FantasyQuant web API."""

from __future__ import annotations

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Session creation
# ---------------------------------------------------------------------------

class ScoringInput(BaseModel):
    receptions: float = 1.0
    passing_tds: float = 4.0
    interceptions: float = -2.0
    rushing_yards: float = 0.1
    rushing_tds: float = 6.0
    receiving_yards: float = 0.1
    receiving_tds: float = 6.0
    fumbles_lost: float = -2.0
    te_reception_bonus: float = 0.0


class RosterInput(BaseModel):
    teams: int = 12
    rounds: int = 15
    qb: int = 1
    rb: int = 2
    wr: int = 2
    te: int = 1
    flex: int = 1
    superflex: int = 0
    bench: int = 6
    dst: int = 1
    k: int = 1


class CreateSessionRequest(BaseModel):
    preset: str | None = None
    scoring: ScoringInput | None = None
    roster: RosterInput | None = None
    platform: str = "custom"
    my_slot: int = Field(1, ge=1, description="Draft position, 1-indexed.")
    odds_api_key: str | None = Field(None, description="The Odds API key for live Vegas lines.")
    adp_source: str | None = Field(None, description="Session-uploaded ADP file reference.")
    win_totals_source: str | None = Field(None, description="Session-uploaded win totals file.")
    player_props_source: str | None = Field(None, description="Session-uploaded player props file.")


class CreateSessionResponse(BaseModel):
    session_id: str
    status: str  # "configuring" | "loading" | "ready" | "drafting" | "complete"
    config_summary: str


# ---------------------------------------------------------------------------
# Draft state
# ---------------------------------------------------------------------------

class PlayerInfo(BaseModel):
    player_id: str
    player_name: str
    position: str
    team: str
    total_projected: float = 0.0
    adp: float | None = None


class PickRecord(BaseModel):
    overall_pick: int
    round: int
    pick_in_round: int
    player_id: str
    player_name: str
    position: str
    team: str
    is_mine: bool


class RosterPlayer(BaseModel):
    player_id: str
    player_name: str
    position: str
    team: str
    pick_number: int
    total_projected: float


class RecommendationResponse(BaseModel):
    player_id: str
    player_name: str
    position: str
    team: str
    expected_wins: float
    expected_points: float
    objective_value: float
    alternatives: list[AlternativePlayer] = []


class AlternativePlayer(BaseModel):
    player_id: str
    player_name: str
    position: str
    team: str

# Fix forward reference
RecommendationResponse.model_rebuild()


class DraftStateResponse(BaseModel):
    session_id: str
    status: str
    current_pick: int
    current_round: int
    total_rounds: int
    total_teams: int
    my_slot: int
    is_my_turn: bool
    picks: list[PickRecord]
    my_roster: list[RosterPlayer]
    roster_needs: dict[str, int]
    config_summary: str


# ---------------------------------------------------------------------------
# Pick input
# ---------------------------------------------------------------------------

class MakePickRequest(BaseModel):
    player_id: str | None = None
    player_name: str | None = None


class MakePickResponse(BaseModel):
    success: bool
    pick: PickRecord | None = None
    error: str | None = None


# ---------------------------------------------------------------------------
# Player pool query
# ---------------------------------------------------------------------------

class PlayerPoolQuery(BaseModel):
    position: str | None = None
    search: str | None = None
    limit: int = 50


class PlayerPoolResponse(BaseModel):
    players: list[PlayerInfo]
    total_available: int


# ---------------------------------------------------------------------------
# Draft summary
# ---------------------------------------------------------------------------

class WeekProjection(BaseModel):
    week: int
    projected_points: float


class PositionGrade(BaseModel):
    position: str
    players: list[RosterPlayer]
    grade: str
    note: str


class DraftSummaryResponse(BaseModel):
    session_id: str
    config_summary: str
    roster: list[RosterPlayer]
    total_projected_points: float
    projected_weekly: list[WeekProjection]
    position_grades: list[PositionGrade]
    overall_grade: str
    strengths: list[str]
    weaknesses: list[str]
