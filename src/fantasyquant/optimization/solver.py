"""Mixed-Integer Programming solver for optimal draft picks.

Implements the full Becker & Sun (2013) formulation:

  (Draft_k)  max  λ₀ Σ f(i,t)·xᵢᵗ  +  λ₁ Σ zᵗ (t≤15)  +  λ₂ Σ zᵗ (t∈{16,17})

  s.t.  roster constraints, starter logic, ADP/rank availability (1b),
        positional minimums (1c), weekly position limits (1d),
        start-only-if-owned (1e), win-threshold β(t) (1f),
        existing picks (1g-1h).

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
    my_slot: int = 1  # 1-indexed draft slot

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
    # The full set Pₖ of players the solver recommends (Algorithm 1).
    recommended_set: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# β(t) estimation  (Paper Section 5.2)
# ---------------------------------------------------------------------------

def estimate_beta(
    state: DraftState,
    pool: PlayerPool,
    config: EngineConfig,
) -> dict[int, float]:
    """Estimate β(t): points needed to beat the DM's opponent in week *t*.

    Per Section 5.2 of Becker & Sun:
    - Simulate the remaining draft for opponents by assigning players
      according to ADP rank order.
    - For each week, compute the opponent's best starting lineup score.
    - For playoff weeks, take the max across possible opponents.
    """
    from fantasyquant.optimization.lineup import set_lineup

    cfg = config
    roster = cfg.roster
    opt = cfg.optimization
    proj = pool.projections
    info = pool.info

    id_to_pos: dict[str, str] = dict(zip(info["player_id"], info["position"]))

    # Rank available players by ADP (or total projected points as proxy).
    if pool.adp is not None:
        ranking = pool.adp.copy()
    else:
        ranking = proj.sum(axis=1).sort_values(ascending=False)
        ranking = pd.Series(range(1, len(ranking) + 1), index=ranking.index)

    # Simulate an "average" opponent roster from remaining players.
    available = [
        pid for pid in ranking.sort_values().index
        if pid not in state.taken
    ]

    # Build a simulated opponent roster using rank-order drafting.
    opp_roster: list[str] = []
    pos_count: dict[str, int] = {"QB": 0, "RB": 0, "WR": 0, "TE": 0}
    pos_limits = {"QB": roster.qb + 1, "RB": roster.rb + roster.flex + 1,
                  "WR": roster.wr + 1, "TE": roster.te + 1}

    for pid in available:
        if len(opp_roster) >= roster.total_slots:
            break
        pos = id_to_pos.get(pid, "")
        if pos not in pos_count:
            continue
        if pos_count[pos] >= pos_limits.get(pos, 99):
            continue
        opp_roster.append(pid)
        pos_count[pos] += 1

    # Compute β(t) for each week.
    beta: dict[int, float] = {}
    for w in range(1, cfg.nfl_weeks + 1):
        scores = {}
        for pid in opp_roster:
            try:
                scores[pid] = float(proj.at[pid, w])
            except (KeyError, ValueError):
                scores[pid] = 0.0
        _, week_score = set_lineup(opp_roster, scores, id_to_pos, config)
        beta[w] = week_score

    # For playoff weeks within our range, be conservative: max across opponents.
    active_playoff_weeks = [pw for pw in opt.playoff_weeks if pw <= cfg.nfl_weeks]
    if active_playoff_weeks:
        playoff_beta = max(beta.get(pw, 0.0) for pw in active_playoff_weeks)
        for pw in active_playoff_weeks:
            beta[pw] = playoff_beta

    return beta


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

class DraftSolver:
    """Build and solve the MIP for a single draft pick (Draftₖ)."""

    def __init__(
        self,
        pool: PlayerPool,
        config: EngineConfig = DEFAULT_CONFIG,
    ) -> None:
        self.pool = pool
        self.config = config
        self.weeks = config.nfl_weeks

    def _build_ranking(self, state: DraftState) -> pd.Series:
        """Build Rₖ: ranking of remaining players by ADP/expert rank."""
        if self.pool.adp is not None:
            rank = self.pool.adp.copy()
        else:
            total_proj = self.pool.projections.sum(axis=1)
            rank = total_proj.rank(ascending=True)  # lower rank = better
        # Remove taken players.
        remaining = rank.drop(labels=list(state.taken), errors="ignore")
        return remaining.sort_values()

    def _snake_pick_numbers(self, state: DraftState) -> list[int]:
        """Compute the DM's remaining overall pick numbers in the snake draft."""
        n_teams = self.config.roster.teams
        n_rounds = self.config.roster.rounds
        total_picks = n_teams * n_rounds
        slot = state.my_slot  # 1-indexed

        our_picks: list[int] = []
        for rnd in range(1, n_rounds + 1):
            if rnd % 2 == 1:  # odd round → normal order
                pick = (rnd - 1) * n_teams + slot
            else:  # even round → reversed
                pick = (rnd - 1) * n_teams + (n_teams - slot + 1)
            if pick >= state.current_pick:
                our_picks.append(pick)

        return our_picks

    def solve(
        self,
        state: DraftState,
        beta: dict[int, float] | None = None,
    ) -> Recommendation | None:
        """Find the optimal player to draft given current *state*.

        Parameters
        ----------
        state:
            Current draft state.
        beta:
            Opponent score thresholds β(t) for each week.  If None,
            estimated automatically.

        Returns *None* if the solver finds no feasible solution.
        """
        cfg = self.config
        roster = cfg.roster
        opt = cfg.optimization
        proj = self.pool.projections
        info = self.pool.info

        # Compute β(t) if not provided.
        if beta is None:
            beta = estimate_beta(state, self.pool, cfg)

        # Available players: not yet taken.
        available_ids = [
            pid for pid in proj.index if pid not in state.taken
        ]
        if not available_ids:
            return None

        # Build lookups.
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

        # Build current ranking Rₖ for constraint (1b).
        ranking = self._build_ranking(state)

        # ===============================================================
        # MIP formulation  (Draftₖ from Becker & Sun)
        # ===============================================================
        prob = pulp.LpProblem("FantasyDraft", pulp.LpMaximize)

        # --- Decision variables ---
        # y_i: 1 if player i is on our roster after this pick
        y = {pid: pulp.LpVariable(f"y_{pid}", cat="Binary")
             for pid in candidate_ids}

        # x_i^t: continuous [0,1] — can relax per paper's observation
        # that the LP relaxation is tight due to problem structure.
        x = {
            (pid, w): pulp.LpVariable(f"x_{pid}_{w}", 0, 1)
            for pid in candidate_ids
            for w in range(1, self.weeks + 1)
        }

        # z_t: 1 if we win week t (score >= β(t))
        z = {
            w: pulp.LpVariable(f"z_{w}", cat="Binary")
            for w in range(1, self.weeks + 1)
        }

        # --- Objective (Eq 1a): λ₀·points + λ₁·regular_wins + λ₂·playoff_wins ---
        total_starter_points = pulp.lpSum(
            pts(pid, w) * x[(pid, w)]
            for pid in candidate_ids
            for w in range(1, self.weeks + 1)
        )

        # Regular season = weeks up to min(regular_season_weeks, total weeks).
        reg_end = min(opt.regular_season_weeks, self.weeks)
        regular_wins = pulp.lpSum(
            z[w] for w in range(1, reg_end + 1)
        )

        playoff_wins = pulp.lpSum(
            z[w] for w in opt.playoff_weeks if w <= self.weeks
        )

        prob += (
            opt.lambda_points * total_starter_points
            + opt.lambda_wins * regular_wins
            + opt.lambda_playoff * playoff_wins
        )

        # --- Constraint (1b): robust opponent draft uncertainty ---
        # At round k̄, the DM can pick at most (k̄ − k) players from
        # the top α·(n_k̄ − n_k) ranked players in Rₖ.
        alpha = opt.alpha
        n_teams = roster.teams
        our_future_picks = self._snake_pick_numbers(state)

        if len(ranking) > 0 and len(our_future_picks) > 1:
            ranked_players = list(ranking.index)
            current_overall = state.current_pick
            for idx, future_pick in enumerate(our_future_picks[1:], 1):
                # Between now and our idx-th future pick, opponents pick
                # (future_pick - current_overall - idx) players.
                picks_between = future_pick - current_overall
                window = int(alpha * picks_between)
                if window > 0 and window <= len(ranked_players):
                    top_window = ranked_players[:window]
                    top_in_candidates = [
                        pid for pid in top_window if pid in y
                    ]
                    if top_in_candidates:
                        # DM can pick at most idx players from this window.
                        prob += (
                            pulp.lpSum(y[pid] for pid in top_in_candidates)
                            - pulp.lpSum(
                                1 for pid in top_in_candidates
                                if pid in owned_ids
                            )
                            <= idx
                        )

        # --- Constraint (1c): positional minimums ---
        # γ_j = PosLimit(j) + 1 per paper's calibration.
        # Build the FULL optimal roster per Draft_k, not just +1 player.
        # Algorithm 1 then selects the most urgent pick from the
        # recommended set (P_k \ owned).
        target_size = roster.rounds
        pos_mins = {
            "QB": roster.qb + 1,
            "RB": roster.rb + 1,
            "WR": roster.wr + 1,
            "TE": roster.te + 1,
        }
        # Only enforce once we have enough picks to fill them.
        total_min = sum(pos_mins.values())
        if target_size >= total_min:
            for pos, mn in pos_mins.items():
                pids_at_pos = [
                    pid for pid in candidate_ids
                    if id_to_pos.get(pid) == pos
                ]
                if pids_at_pos:
                    prob += pulp.lpSum(y[pid] for pid in pids_at_pos) >= mn

        # --- Roster size: full roster (total draft rounds) ---
        prob += pulp.lpSum(y[pid] for pid in candidate_ids) == target_size

        # Players already owned MUST stay (1g).
        for pid in owned_ids:
            if pid in y:
                prob += y[pid] == 1

        # Players taken by opponents cannot be on our roster (1h).
        for pid in candidate_ids:
            if pid in state.taken and pid not in owned_ids:
                prob += y[pid] == 0

        # --- Constraint (1d): weekly starter limits per position ---
        for w in range(1, self.weeks + 1):
            for pos, limit in [("QB", roster.qb), ("RB", roster.rb),
                               ("WR", roster.wr), ("TE", roster.te)]:
                pids_at_pos = [
                    pid for pid in candidate_ids
                    if id_to_pos.get(pid) == pos
                ]
                if pids_at_pos:
                    prob += pulp.lpSum(
                        x[(pid, w)] for pid in pids_at_pos
                    ) <= limit

            # FLEX slot: one additional RB/WR/TE can start.
            flex_pids = [
                pid for pid in candidate_ids
                if id_to_pos.get(pid) in ("RB", "WR", "TE")
            ]
            if flex_pids:
                prob += (
                    pulp.lpSum(x[(pid, w)] for pid in flex_pids)
                    <= roster.rb + roster.wr + roster.te + roster.flex
                )

            # SUPERFLEX slot: one additional QB/RB/WR/TE can start.
            if roster.superflex > 0:
                all_skill_pids = [
                    pid for pid in candidate_ids
                    if id_to_pos.get(pid) in ("QB", "RB", "WR", "TE")
                ]
                if all_skill_pids:
                    prob += (
                        pulp.lpSum(x[(pid, w)] for pid in all_skill_pids)
                        <= roster.qb + roster.rb + roster.wr + roster.te
                        + roster.flex + roster.superflex
                    )

        # --- Constraint (1e): can only start a player you own ---
        for pid in candidate_ids:
            for w in range(1, self.weeks + 1):
                prob += x[(pid, w)] <= y[pid]

        # --- Constraint (1f): win threshold using β(t) ---
        M = 5000.0  # big-M upper bound on weekly score
        for w in range(1, self.weeks + 1):
            weekly_score = pulp.lpSum(
                pts(pid, w) * x[(pid, w)] for pid in candidate_ids
            )
            threshold = beta.get(w, 0.0)
            # z_w = 1 only if weekly_score >= β(t)
            prob += weekly_score >= threshold - M * (1 - z[w])
            prob += weekly_score <= threshold + M * z[w]

        # Position upper bounds on total roster.
        pos_limits = {
            "QB": roster.qb + roster.superflex + roster.bench,
            "RB": roster.rb + roster.flex + roster.superflex + roster.bench,
            "WR": roster.wr + roster.flex + roster.superflex + roster.bench,
            "TE": roster.te + roster.flex + roster.superflex + roster.bench,
        }
        for pos, limit in pos_limits.items():
            pids_at_pos = [
                pid for pid in candidate_ids
                if id_to_pos.get(pid) == pos
            ]
            if pids_at_pos:
                prob += pulp.lpSum(y[pid] for pid in pids_at_pos) <= limit

        # ---- Solve ----
        prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=opt.solver_time_limit_seconds))

        if prob.status != pulp.constants.LpStatusOptimal:
            return None

        # ===============================================================
        # Algorithm 1 (revised): Pick by urgency-weighted value
        # ===============================================================
        # Pₖ = {i : yᵢ = 1, i ∉ DMPlayer(k)} — newly recommended players.
        recommended_set = [
            pid for pid in available_ids
            if pid in y
            and pulp.value(y[pid]) is not None
            and pulp.value(y[pid]) > 0.5
            and pid not in owned_ids
        ]

        if not recommended_set:
            return None

        # --- Rank candidates by urgency × value ---
        # Urgency = probability the player gets taken before our next
        # pick.  Value = season projection minus positional replacement.
        our_picks = self._snake_pick_numbers(state)
        next_pick = our_picks[1] if len(our_picks) > 1 else state.current_pick + n_teams
        picks_until_next = next_pick - state.current_pick

        ranked = self._rank_by_urgency(
            recommended_set, ranking, state, picks_until_next,
            id_to_pos,
        )
        best_pid = ranked[0]
        # Store sorted set so alternatives are meaningful.
        recommended_set = ranked

        # --- Compute summary stats ---
        exp_wins = 0.0
        for w in range(1, self.weeks + 1):
            v = pulp.value(z[w])
            if v is not None:
                exp_wins += v

        exp_pts = sum(
            pts(pid, w) * (pulp.value(x[(pid, w)]) or 0)
            for pid in candidate_ids
            for w in range(1, self.weeks + 1)
        )

        return Recommendation(
            player_id=best_pid,
            player_name=id_to_name.get(best_pid, best_pid),
            position=id_to_pos.get(best_pid, "??"),
            expected_wins=exp_wins,
            expected_points=exp_pts,
            objective_value=float(pulp.value(prob.objective) or 0.0),
            recommended_set=recommended_set,
        )

    def _rank_by_urgency(
        self,
        recommended_set: list[str],
        ranking: pd.Series,
        state: DraftState,
        picks_until_next: int,
        id_to_pos: dict[str, str],
    ) -> list[str]:
        """Sort the MIP's recommended set by urgency-weighted value.

        Score = urgency × marginal_value

        Urgency:  For a player with ADP rank R among remaining players,
        if R <= picks_until_next they're almost certainly gone → urgency ~ 1.
        If R >> picks_until_next they'll be available later → urgency ~ 0.
        Uses a logistic curve centered at picks_until_next.

        Marginal value:  Player's season total projected points relative
        to the best *replacement* at the same position available at our
        next pick (VOR = value over replacement).
        """
        proj = self.pool.projections

        # Position-grouped remaining players by ADP rank.
        pos_ranked: dict[str, list[tuple[str, float]]] = {}
        for pid in ranking.index:
            if pid in state.taken:
                continue
            pos = id_to_pos.get(pid, "")
            if pos not in pos_ranked:
                pos_ranked[pos] = []
            pos_ranked[pos].append((pid, ranking.get(pid, 999.0)))

        # For each position, find the replacement-level player:
        # the best player at that position who'd still be available
        # at our next pick.
        replacement_pts: dict[str, float] = {}
        for pos, players in pos_ranked.items():
            players_sorted = sorted(players, key=lambda x: x[1])
            # ~1/4 of picks go to each position.
            repl_idx = min(len(players_sorted) - 1,
                           max(0, picks_until_next // 4))
            repl_pid = players_sorted[repl_idx][0]
            try:
                replacement_pts[pos] = float(proj.loc[repl_pid].sum())
            except (KeyError, ValueError):
                replacement_pts[pos] = 0.0

        scored: list[tuple[str, float]] = []

        for pid in recommended_set:
            adp_rank = ranking.get(pid, float("inf"))
            pos = id_to_pos.get(pid, "")

            # Urgency: logistic curve centered at picks_until_next.
            k = 0.3  # steepness
            urgency = 1.0 / (1.0 + np.exp(-k * (picks_until_next - adp_rank + 1)))

            # Marginal value: season total minus replacement at same position.
            try:
                player_total = float(proj.loc[pid].sum())
            except (KeyError, ValueError):
                player_total = 0.0
            repl = replacement_pts.get(pos, 0.0)
            marginal = max(player_total - repl, 0.0)

            scored.append((pid, urgency * marginal))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [pid for pid, _ in scored]
