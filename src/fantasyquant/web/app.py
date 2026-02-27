"""FastAPI application for the FantasyQuant draft room."""

from __future__ import annotations

import json
import tempfile
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from fantasyquant.config import EngineConfig
from fantasyquant.league import (
    PRESETS,
    describe_config,
    list_presets,
    load_league,
)
from fantasyquant.web.models import (
    CreateSessionRequest,
    CreateSessionResponse,
    DraftStateResponse,
    DraftSummaryResponse,
    MakePickRequest,
    MakePickResponse,
    PlayerPoolResponse,
    RecommendationResponse,
    AlternativePlayer,
)
from fantasyquant.web.payment import create_checkout_session, is_dev_mode
from fantasyquant.web.session import (
    DraftSession,
    SessionStatus,
    create_session,
    get_session,
)
from fantasyquant.web.summary import build_summary

_WEB_DIR = Path(__file__).parent
_TEMPLATE_DIR = _WEB_DIR / "templates"
_STATIC_DIR = _WEB_DIR / "static"

app = FastAPI(title="FantasyQuant", version="0.1.0")
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(_TEMPLATE_DIR))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_config(req: CreateSessionRequest) -> EngineConfig:
    """Resolve request into an EngineConfig."""
    if req.preset:
        config = load_league(preset=req.preset)
    else:
        config = EngineConfig()

    if req.scoring:
        config.scoring = config.scoring.__class__(**req.scoring.model_dump())
    if req.roster:
        config.roster = config.roster.__class__(**req.roster.model_dump())
    if req.platform != "custom":
        config.platform = req.platform
    return config


def _load_projections_async(session: DraftSession) -> None:
    """Load projections in a background thread."""
    try:
        from fantasyquant.prediction.projections import build_projections
        from fantasyquant.optimization.solver import PlayerPool

        result = build_projections(
            session.config,
            win_totals_source=session.win_totals_source,
            player_props_source=session.player_props_source,
            adp_source=session.adp_source,
            odds_api_key=session.odds_api_key,
        )
        pool = PlayerPool(
            projections=result.weekly_projections,
            info=result.player_info,
            adp=result.adp,
        )
        session.initialize(pool)
    except Exception:
        session.status = SessionStatus.LOADING


def _get_session_or_404(session_id: str) -> DraftSession:
    session = get_session(session_id)
    if session is None:
        raise HTTPException(404, "Session not found")
    return session


# ---------------------------------------------------------------------------
# Page routes (HTML)
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def landing(request: Request):
    return templates.TemplateResponse(request, "configure.html", {
        "presets": {name: describe_config(load_league(preset=name)) for name in list_presets()},
        "dev_mode": is_dev_mode(),
    })


@app.get("/configure", response_class=HTMLResponse)
async def configure_page(request: Request):
    return templates.TemplateResponse(request, "configure.html", {
        "presets": {name: describe_config(load_league(preset=name)) for name in list_presets()},
        "dev_mode": is_dev_mode(),
    })


@app.get("/draft/{session_id}", response_class=HTMLResponse)
async def draft_page(request: Request, session_id: str):
    session = _get_session_or_404(session_id)
    return templates.TemplateResponse(request, "draft.html", {
        "session_id": session_id,
        "config_summary": session.config_summary,
        "my_slot": session.my_slot,
        "total_teams": session.total_teams,
        "total_rounds": session.total_rounds,
    })


@app.get("/summary/{session_id}", response_class=HTMLResponse)
async def summary_page(request: Request, session_id: str):
    session = _get_session_or_404(session_id)
    summary = build_summary(session)
    return templates.TemplateResponse(request, "summary.html", {
        "summary": summary,
    })


# ---------------------------------------------------------------------------
# API routes (JSON)
# ---------------------------------------------------------------------------

@app.post("/api/sessions", response_model=CreateSessionResponse)
async def api_create_session(req: CreateSessionRequest):
    config = _build_config(req)
    session = create_session(req, config)

    # Start loading projections in background.
    t = threading.Thread(target=_load_projections_async, args=(session,), daemon=True)
    t.start()

    return CreateSessionResponse(
        session_id=session.session_id,
        status=session.status.value,
        config_summary=session.config_summary,
    )


@app.post("/api/sessions/{session_id}/pay")
async def api_pay(session_id: str):
    session = _get_session_or_404(session_id)
    url = create_checkout_session(session_id)
    return {"checkout_url": url}


@app.get("/api/sessions/{session_id}/state", response_model=DraftStateResponse)
async def api_get_state(session_id: str):
    session = _get_session_or_404(session_id)
    return DraftStateResponse(
        session_id=session.session_id,
        status=session.status.value,
        current_pick=session.current_pick,
        current_round=session.current_round,
        total_rounds=session.total_rounds,
        total_teams=session.total_teams,
        my_slot=session.my_slot,
        is_my_turn=session.is_my_turn,
        picks=[session.to_pick_record(e) for e in session.picks],
        my_roster=session.get_my_roster(),
        roster_needs=session.get_roster_needs(),
        config_summary=session.config_summary,
    )


@app.get("/api/sessions/{session_id}/recommend", response_model=RecommendationResponse)
async def api_recommend(session_id: str):
    session = _get_session_or_404(session_id)
    if session.loop is None:
        raise HTTPException(409, "Session not ready — projections still loading")
    if not session.is_my_turn:
        raise HTTPException(409, "Not your turn")

    rec = session.get_recommendation()
    if rec is None:
        raise HTTPException(500, "Solver could not find a feasible pick")

    # Build alternatives from the recommended set (excluding the top pick).
    alternatives = []
    for pid in rec.recommended_set[:5]:
        if pid == rec.player_id:
            continue
        pinfo = session.get_player_info(pid)
        if pinfo:
            alternatives.append(AlternativePlayer(**pinfo))

    return RecommendationResponse(
        player_id=rec.player_id,
        player_name=rec.player_name,
        position=rec.position,
        team=session.get_player_info(rec.player_id) and session.get_player_info(rec.player_id)["team"] or "",
        expected_wins=round(rec.expected_wins, 1),
        expected_points=round(rec.expected_points, 0),
        objective_value=round(rec.objective_value, 2),
        alternatives=alternatives,
    )


@app.post("/api/sessions/{session_id}/pick", response_model=MakePickResponse)
async def api_make_pick(session_id: str, req: MakePickRequest):
    session = _get_session_or_404(session_id)
    if session.loop is None:
        raise HTTPException(409, "Session not ready — projections still loading")

    # Resolve player by ID or name.
    player_id = req.player_id
    if player_id is None and req.player_name:
        # Fuzzy match.
        from fantasyquant.optimization.draft_loop import _fuzzy_match
        name_lookup = session.loop._name_lookup
        player_id = _fuzzy_match(req.player_name, name_lookup)
        if player_id is None:
            return MakePickResponse(success=False, error=f"Player not found: '{req.player_name}'")

    if player_id is None:
        return MakePickResponse(success=False, error="Provide player_id or player_name")

    # Check the player isn't already taken.
    if player_id in session.loop.state.taken:
        return MakePickResponse(success=False, error="Player already drafted")

    if session.is_my_turn:
        entry, _ = session.record_my_pick(player_id)
    else:
        entry = session.record_opponent_pick(player_id)

    if entry is None:
        return MakePickResponse(success=False, error="Could not record pick")

    return MakePickResponse(
        success=True,
        pick=session.to_pick_record(entry),
    )


@app.post("/api/sessions/{session_id}/auto-pick", response_model=MakePickResponse)
async def api_auto_pick(session_id: str):
    """Use the solver to make our pick automatically."""
    session = _get_session_or_404(session_id)
    if session.loop is None:
        raise HTTPException(409, "Session not ready")
    if not session.is_my_turn:
        raise HTTPException(409, "Not your turn")

    entry, rec = session.record_my_pick(player_id=None)
    if entry is None:
        return MakePickResponse(success=False, error="Solver could not find a pick")

    return MakePickResponse(
        success=True,
        pick=session.to_pick_record(entry),
    )


@app.get("/api/sessions/{session_id}/players", response_model=PlayerPoolResponse)
async def api_player_pool(
    session_id: str,
    position: str | None = None,
    search: str | None = None,
    limit: int = 50,
):
    session = _get_session_or_404(session_id)
    players = session.get_available_players(position=position, search=search, limit=limit)
    total = len(session.get_available_players(position=position, search=search, limit=9999))
    return PlayerPoolResponse(players=players, total_available=total)


@app.get("/api/sessions/{session_id}/summary", response_model=DraftSummaryResponse)
async def api_summary(session_id: str):
    session = _get_session_or_404(session_id)
    return build_summary(session)


@app.get("/api/presets")
async def api_presets():
    result = {}
    for name in list_presets():
        config = load_league(preset=name)
        p = PRESETS[name]
        result[name] = {
            "description": describe_config(config),
            "platform": p.get("platform", "custom"),
            "scoring": p.get("scoring", {}),
            "roster": p.get("roster", {}),
        }
    return result


# ---------------------------------------------------------------------------
# File upload endpoints (data sources)
# ---------------------------------------------------------------------------

def _save_upload(upload: UploadFile, prefix: str) -> str:
    """Save an uploaded file to a temp location and return the path."""
    suffix = Path(upload.filename or "data.csv").suffix or ".csv"
    fd = tempfile.NamedTemporaryFile(
        prefix=f"fq_{prefix}_", suffix=suffix, delete=False,
    )
    content = upload.file.read()
    fd.write(content)
    fd.close()
    return fd.name


@app.post("/api/upload/adp")
async def api_upload_adp(file: UploadFile):
    """Upload an ADP CSV/JSON file. Returns a file reference to pass in session creation."""
    path = _save_upload(file, "adp")
    return {"adp_source": path, "filename": file.filename}


@app.post("/api/upload/win-totals")
async def api_upload_win_totals(file: UploadFile):
    """Upload a team win totals CSV/JSON file."""
    path = _save_upload(file, "wintotals")
    return {"win_totals_source": path, "filename": file.filename}


@app.post("/api/upload/props")
async def api_upload_props(file: UploadFile):
    """Upload a player props CSV/JSON file."""
    path = _save_upload(file, "props")
    return {"player_props_source": path, "filename": file.filename}


@app.get("/api/data-sources")
async def api_data_sources():
    """Describe available data sources and their expected formats."""
    return {
        "adp": {
            "description": "Average Draft Position data",
            "auto_fallback": "Prior season performance via nfl_data_py, then projection-derived ranking",
            "csv_columns": ["player_id", "adp"],
            "json_format": '{"player_id": adp_rank, ...}',
        },
        "win_totals": {
            "description": "Vegas team season win totals",
            "auto_fallback": "Built-in consensus lines (updated each preseason) or The Odds API with API key",
            "csv_columns": ["team", "win_total"],
            "json_format": '{"KC": 11.5, "SF": 10.5, ...}',
        },
        "player_props": {
            "description": "Player season fantasy point total props",
            "auto_fallback": "Prior season actuals via nfl_data_py",
            "csv_columns": ["player_id", "season_total"],
            "json_format": '{"player_id": season_total, ...}',
        },
    }
