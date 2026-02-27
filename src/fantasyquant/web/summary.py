"""Post-draft analysis — grades, projections, strengths/weaknesses."""

from __future__ import annotations

from fantasyquant.optimization.lineup import project_lineup_score
from fantasyquant.web.models import (
    DraftSummaryResponse,
    PositionGrade,
    RosterPlayer,
    WeekProjection,
)
from fantasyquant.web.session import DraftSession


def _grade_from_percentile(pct: float) -> str:
    if pct >= 90:
        return "A+"
    elif pct >= 80:
        return "A"
    elif pct >= 70:
        return "B+"
    elif pct >= 60:
        return "B"
    elif pct >= 50:
        return "C+"
    elif pct >= 40:
        return "C"
    elif pct >= 30:
        return "D+"
    elif pct >= 20:
        return "D"
    return "F"


def _position_depth_grade(
    players: list[RosterPlayer],
    starter_slots: int,
    avg_top_projection: float,
) -> tuple[str, str]:
    """Grade a position group based on projection strength and depth."""
    if not players:
        return "F", "No players drafted"

    total = sum(p.total_projected for p in players)
    avg = total / len(players) if players else 0
    starter_total = sum(
        p.total_projected for p in sorted(players, key=lambda x: -x.total_projected)[:starter_slots]
    )

    if avg_top_projection <= 0:
        return "C", f"{len(players)} players"

    # Grade starters relative to average top projection.
    ratio = starter_total / (starter_slots * avg_top_projection) if starter_slots > 0 else 0
    pct = min(ratio * 70, 100)  # Scale so ~100% of avg = B+

    grade = _grade_from_percentile(pct)
    note = f"{len(players)} players, {starter_total:.0f} starter pts projected"
    return grade, note


def build_summary(session: DraftSession) -> DraftSummaryResponse:
    """Generate the end-of-draft report card."""
    if session.loop is None:
        return DraftSummaryResponse(
            session_id=session.session_id,
            config_summary=session.config_summary,
            roster=[],
            total_projected_points=0,
            projected_weekly=[],
            position_grades=[],
            overall_grade="N/A",
            strengths=[],
            weaknesses=[],
        )

    roster = session.get_my_roster()
    pool = session.loop.pool
    config = session.config
    proj = pool.projections
    info = pool.info

    id_to_pos: dict[str, str] = dict(zip(info["player_id"], info["position"]))
    roster_pids = [p.player_id for p in roster]

    # Weekly projections.
    weekly: list[WeekProjection] = []
    total = 0.0
    for w in range(1, config.nfl_weeks + 1):
        score = project_lineup_score(roster_pids, proj, id_to_pos, w, config)
        weekly.append(WeekProjection(week=w, projected_points=round(score, 1)))
        total += score

    # Position grades.
    pos_groups: dict[str, list[RosterPlayer]] = {"QB": [], "RB": [], "WR": [], "TE": []}
    for p in roster:
        if p.position in pos_groups:
            pos_groups[p.position].append(p)

    # Compute average top projected player per position (baseline).
    pos_avg_top: dict[str, float] = {}
    for pos in pos_groups:
        mask = info["position"] == pos
        pos_pids = info[mask]["player_id"].tolist()
        if pos_pids:
            totals = [float(proj.sum(axis=1).get(pid, 0)) for pid in pos_pids]
            totals.sort(reverse=True)
            # Average of top-N where N = starter slots for that position.
            slots = getattr(config.roster, pos.lower(), 1)
            top_n = totals[:max(slots * config.roster.teams, 1)]
            pos_avg_top[pos] = sum(top_n) / len(top_n) if top_n else 0
        else:
            pos_avg_top[pos] = 0

    position_grades: list[PositionGrade] = []
    for pos, players in pos_groups.items():
        slots = getattr(config.roster, pos.lower(), 1)
        grade, note = _position_depth_grade(players, slots, pos_avg_top.get(pos, 0))
        position_grades.append(PositionGrade(
            position=pos,
            players=players,
            grade=grade,
            note=note,
        ))

    # Overall grade: weighted by position importance.
    grade_values = {"A+": 97, "A": 93, "B+": 87, "B": 83, "C+": 77, "C": 73, "D+": 67, "D": 63, "F": 50}
    if position_grades:
        weights = {"QB": 1.0, "RB": 1.5, "WR": 1.5, "TE": 0.8}
        weighted_sum = sum(
            grade_values.get(pg.grade, 70) * weights.get(pg.position, 1.0)
            for pg in position_grades
        )
        total_weight = sum(weights.get(pg.position, 1.0) for pg in position_grades)
        avg_grade_val = weighted_sum / total_weight if total_weight else 70
        overall = _grade_from_percentile(avg_grade_val - 20)  # Shift to letter scale
    else:
        overall = "N/A"

    # Strengths and weaknesses.
    strengths: list[str] = []
    weaknesses: list[str] = []

    sorted_grades = sorted(position_grades, key=lambda g: grade_values.get(g.grade, 70), reverse=True)
    for pg in sorted_grades:
        gv = grade_values.get(pg.grade, 70)
        if gv >= 85:
            strengths.append(f"{pg.position}: {pg.grade} — {pg.note}")
        elif gv <= 65:
            weaknesses.append(f"{pg.position}: {pg.grade} — {pg.note}")

    # Check for depth concerns.
    if config.roster.superflex and pos_groups["QB"] and len(pos_groups["QB"]) < 2:
        weaknesses.append("Only 1 QB in a superflex league — high risk on bye weeks")

    avg_weekly = total / config.nfl_weeks if config.nfl_weeks else 0
    if avg_weekly > 0:
        strengths.insert(0, f"Projected {total:.0f} total pts ({avg_weekly:.1f}/week)")

    return DraftSummaryResponse(
        session_id=session.session_id,
        config_summary=session.config_summary,
        roster=roster,
        total_projected_points=round(total, 1),
        projected_weekly=weekly,
        position_grades=position_grades,
        overall_grade=overall,
        strengths=strengths,
        weaknesses=weaknesses,
    )
