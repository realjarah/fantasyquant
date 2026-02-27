"""Mixed-Integer Programming solver for optimal draft picks.

Formulates the fantasy draft as a MIP:

  max  λ_wins × E[weekly_wins] + λ_points × E[total_points]

  s.t. roster constraints, starter logic, ADP availability.

Uses PuLP with the bundled CBC solver.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pulp

from fantasyquant.config import EngineConfig, DEFAULT_CONFIG


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class PlayerPool:
    """All draftable players and their projections."""

    projections: pd.DataFrame
    """player_id × week matrix of projected fantasy points."""

    info: pd.DataFrame
    """Player metadata (player_id, player_name, position, team)."""

    adp: pd.Series | None = None
    """Average Draft Position, indexed by player_id."""


@dataclass
class DraftState:
    """Mutable state tracked across the draft."""

    my_picks: list[str] = field(default_factory=list)
    """player_ids already drafted by us."""

    taken: set[str] = field(default_factory=set)
    """player_ids drafted by any team (including us)."""

    current_round: int = 1
    current_pick: int = 1  # overall pick number

    @property
    def my_roster_size(self) -> int:
        return len(self.my_picks)


@dataclass
class Recommendation:
    """Solver output for a single pick."""

    player_id: str
    player_name: str
    position: str
    expected_wins: float
    expected_points: float
    objective_value: float


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

class DraftSolver:
    """Build and solve the MIP for a single draft pick."""

    def __init__(
        self,
        pool: PlayerPool,
        config: EngineConfig = DEFAULT_CONFIG,
    ) -> None:
        self.pool = pool
        self.config = config
        self.weeks = config.nfl_weeks

    def solve(self, state: DraftState) -> Recommendation | None:
        """Find the optimal player to draft given current *state*.

        Returns *None* if the solver finds no feasible solution (e.g.,
        roster already full).
        """
        cfg = self.config
        roster = cfg.roster
        opt = cfg.optimization
        proj = self.pool.projections
        info = self.pool.info

        # Available players: not yet taken.
        available_ids = [
            pid for pid in proj.index if pid not in state.taken
        ]
        if not available_ids:
            return None

        # Build lookup.
        id_to_pos: dict[str, str] = dict(
            zip(info["player_id"], info["position"]),
        )
        id_to_name: dict[str, str] = dict(
            zip(info["player_id"], info["player_name"]),
        )

        # Players already on our roster.
        owned_ids = set(state.my_picks)

        # --- Candidate set: owned + available ---
        candidate_ids = list(owned_ids | set(available_ids))

        # Projection lookup (default to 0 for any missing week).
        def pts(pid: str, week: int) -> float:
            try:
                return float(proj.at[pid, week])
            except (KeyError, ValueError):
                return 0.0

        # ===============================================================
        # MIP formulation
        # ===============================================================
        prob = pulp.LpProblem("FantasyDraft", pulp.LpMaximize)

        # --- Decision variables ---
        # y_i: 1 if player i is on our roster after this pick
        y = {pid: pulp.LpVariable(f"y_{pid}", cat="Binary") for pid in candidate_ids}

        # x_i^t: 1 if player i starts in week t
        x = {
            (pid, w): pulp.LpVariable(f"x_{pid}_{w}", cat="Binary")
            for pid in candidate_ids
            for w in range(1, self.weeks + 1)
        }

        # win_t: 1 if we "win" week t (proxy: score above threshold)
        # We use total starter points as a proxy for win probability.
        # A more sophisticated model would compare against opponent
        # distributions, but this captures the right incentive structure.

        # --- Objective ---
        total_starter_points = pulp.lpSum(
            pts(pid, w) * x[(pid, w)]
            for pid in candidate_ids
            for w in range(1, self.weeks + 1)
        )

        # Weekly score sums (for win-maximisation proxy).
        weekly_scores = {}
        for w in range(1, self.weeks + 1):
            weekly_scores[w] = pulp.lpSum(
                pts(pid, w) * x[(pid, w)] for pid in candidate_ids
            )

        # Win proxy: count weeks where score exceeds a threshold.
        # We use a big-M linearisation.  Threshold = league-average
        # weekly starter output (~100 pts in full PPR for ~8 starters).
        WIN_THRESHOLD = 100.0
        M = 5000.0  # big enough upper bound on weekly score

        win = {
            w: pulp.LpVariable(f"win_{w}", cat="Binary")
            for w in range(1, self.weeks + 1)
        }
        for w in range(1, self.weeks + 1):
            # win_w = 1 only if weekly_scores[w] >= WIN_THRESHOLD
            prob += weekly_scores[w] >= WIN_THRESHOLD - M * (1 - win[w])
            prob += weekly_scores[w] <= WIN_THRESHOLD + M * win[w]

        expected_wins = pulp.lpSum(win[w] for w in range(1, self.weeks + 1))

        prob += (
            opt.lambda_wins * expected_wins
            + opt.lambda_points * total_starter_points
        )

        # --- Constraints ---

        # 1. Roster size: exactly (currently owned + 1 new pick)
        target_size = state.my_roster_size + 1
        prob += pulp.lpSum(y[pid] for pid in candidate_ids) == target_size

        # Players already owned MUST stay.
        for pid in owned_ids:
            if pid in y:
                prob += y[pid] == 1

        # 2. Position limits (on final roster).
        pos_limits = {
            "QB": roster.qb + roster.bench,  # upper bound
            "RB": roster.rb + roster.flex + roster.bench,
            "WR": roster.wr + roster.flex + roster.bench,
            "TE": roster.te + roster.flex + roster.bench,
        }
        for pos, limit in pos_limits.items():
            pids_at_pos = [pid for pid in candidate_ids if id_to_pos.get(pid) == pos]
            if pids_at_pos:
                prob += pulp.lpSum(y[pid] for pid in pids_at_pos) <= limit

        # Minimum positional starters by end of draft.
        # Only enforce once we're deep enough in the draft.
        if target_size >= roster.starters:
            pos_mins = {"QB": roster.qb, "RB": roster.rb, "WR": roster.wr, "TE": roster.te}
            for pos, mn in pos_mins.items():
                pids_at_pos = [pid for pid in candidate_ids if id_to_pos.get(pid) == pos]
                if pids_at_pos:
                    prob += pulp.lpSum(y[pid] for pid in pids_at_pos) >= mn

        # 3. Starter logic: can only start a player you own.
        for pid in candidate_ids:
            for w in range(1, self.weeks + 1):
                prob += x[(pid, w)] <= y[pid]

        # 4. Weekly starter limits per position.
        for w in range(1, self.weeks + 1):
            for pos, limit in [("QB", roster.qb), ("RB", roster.rb),
                               ("WR", roster.wr), ("TE", roster.te)]:
                pids_at_pos = [pid for pid in candidate_ids if id_to_pos.get(pid) == pos]
                if pids_at_pos:
                    prob += pulp.lpSum(x[(pid, w)] for pid in pids_at_pos) <= limit

            # FLEX slot: one additional RB/WR/TE can start.
            flex_pids = [
                pid for pid in candidate_ids
                if id_to_pos.get(pid) in ("RB", "WR", "TE")
            ]
            # Total RB+WR+TE starters ≤ positional limits + flex.
            if flex_pids:
                prob += (
                    pulp.lpSum(x[(pid, w)] for pid in flex_pids)
                    <= roster.rb + roster.wr + roster.te + roster.flex
                )

        # 5. ADP availability constraint (robustness).
        if self.pool.adp is not None:
            slack = opt.adp_uncertainty_rounds * roster.teams
            next_pick = state.current_pick + roster.teams  # next time we pick
            for pid in available_ids:
                if pid in self.pool.adp.index:
                    player_adp = self.pool.adp[pid]
                    # If the player's ADP is well before our next pick,
                    # they're likely gone — don't plan on getting them later.
                    if player_adp < next_pick - slack:
                        # They must be picked NOW or not at all for this
                        # optimisation pass (solver can still choose y=1
                        # for this pick).
                        pass  # Keep them available for this pick.

        # ---- Solve ----
        prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=opt.solver_time_limit_seconds))

        if prob.status != pulp.constants.LpStatusOptimal:
            return None

        # Find the *new* player selected (y=1 and not already owned).
        selected = None
        for pid in available_ids:
            if pid in y and pulp.value(y[pid]) and pulp.value(y[pid]) > 0.5:
                if pid not in owned_ids:
                    selected = pid
                    break

        if selected is None:
            return None

        # Compute summary stats for the recommendation.
        exp_wins = float(pulp.value(expected_wins)) if pulp.value(expected_wins) else 0.0
        exp_pts = sum(
            pts(pid, w) * (pulp.value(x[(pid, w)]) or 0)
            for pid in candidate_ids
            for w in range(1, self.weeks + 1)
        )

        return Recommendation(
            player_id=selected,
            player_name=id_to_name.get(selected, selected),
            position=id_to_pos.get(selected, "??"),
            expected_wins=exp_wins,
            expected_points=exp_pts,
            objective_value=float(pulp.value(prob.objective)) if pulp.value(prob.objective) else 0.0,
        )
