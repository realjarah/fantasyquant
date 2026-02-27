"""Position-constrained weekly lineup optimizer (Paper Section 4.4).

Given a roster and projected (or actual) weekly scores, select the
starting lineup that maximises total fantasy points subject to
positional slot limits.  Supports standard FLEX (RB/WR/TE) and
SUPERFLEX (QB/RB/WR/TE) slots.

Uses a greedy approach: fill each required slot from best available,
then fill FLEX, then SUPERFLEX from remaining eligible players.
"""

from __future__ import annotations

import pandas as pd

from fantasyquant.config import EngineConfig, DEFAULT_CONFIG


def set_lineup(
    roster_pids: list[str],
    scores: dict[str, float],
    positions: dict[str, str],
    config: EngineConfig = DEFAULT_CONFIG,
) -> tuple[list[str], float]:
    """Select optimal starters from *roster_pids* for one week.

    Parameters
    ----------
    roster_pids:
        player_ids on the roster.
    scores:
        {player_id: projected_or_actual_points} for this week.
    positions:
        {player_id: position} lookup.
    config:
        Engine configuration (roster slot counts).

    Returns
    -------
    (starter_pids, total_score)
    """
    r = config.roster

    # Build scored list per position, sorted descending.
    by_pos: dict[str, list[tuple[str, float]]] = {
        "QB": [], "RB": [], "WR": [], "TE": [],
    }
    for pid in roster_pids:
        pos = positions.get(pid, "")
        pts = scores.get(pid, 0.0)
        if pos in by_pos:
            by_pos[pos].append((pid, pts))

    for pos in by_pos:
        by_pos[pos].sort(key=lambda x: x[1], reverse=True)

    starters: list[str] = []
    used: set[str] = set()

    # Fill required slots.
    slot_reqs = [("QB", r.qb), ("RB", r.rb), ("WR", r.wr), ("TE", r.te)]
    for pos, n_slots in slot_reqs:
        filled = 0
        for pid, _ in by_pos[pos]:
            if pid not in used and filled < n_slots:
                starters.append(pid)
                used.add(pid)
                filled += 1

    # Fill FLEX slots (RB/WR/TE not yet started).
    flex_candidates: list[tuple[str, float]] = []
    for pos in ("RB", "WR", "TE"):
        for pid, pts in by_pos[pos]:
            if pid not in used:
                flex_candidates.append((pid, pts))
    flex_candidates.sort(key=lambda x: x[1], reverse=True)
    flex_filled = 0
    for pid, _ in flex_candidates:
        if flex_filled >= r.flex:
            break
        starters.append(pid)
        used.add(pid)
        flex_filled += 1

    # Fill SUPERFLEX slots (QB/RB/WR/TE not yet started).
    if r.superflex > 0:
        sflex_candidates: list[tuple[str, float]] = []
        for pos in ("QB", "RB", "WR", "TE"):
            for pid, pts in by_pos[pos]:
                if pid not in used:
                    sflex_candidates.append((pid, pts))
        sflex_candidates.sort(key=lambda x: x[1], reverse=True)
        sflex_filled = 0
        for pid, _ in sflex_candidates:
            if sflex_filled >= r.superflex:
                break
            starters.append(pid)
            used.add(pid)
            sflex_filled += 1

    total = sum(scores.get(pid, 0.0) for pid in starters)
    return starters, total


def project_lineup_score(
    roster_pids: list[str],
    projections: pd.DataFrame,
    positions: dict[str, str],
    week: int,
    config: EngineConfig = DEFAULT_CONFIG,
) -> float:
    """Return projected score for the best lineup in a given *week*."""
    scores = {}
    for pid in roster_pids:
        try:
            scores[pid] = float(projections.at[pid, week])
        except (KeyError, ValueError):
            scores[pid] = 0.0

    _, total = set_lineup(roster_pids, scores, positions, config)
    return total
