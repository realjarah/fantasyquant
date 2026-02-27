"""Alternating Minimization for skill / defense decomposition.

Implements Algorithm 2 from Becker & Sun: given a matrix of observed
fantasy points where rows are players and columns are (team-week)
matchups, decompose into

    Observed ≈ PlayerSkill × DefenseMultiplier

by alternately fixing one factor and solving for the other via
least-squares / SVD.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fantasyquant.config import PredictionConfig


@dataclass
class DecompositionResult:
    """Output of the alternating minimisation procedure."""

    player_skill: pd.Series
    """Per-player skill rating (u_i), indexed by player_id."""

    defense_multipliers: pd.Series
    """Per-team defensive multiplier (w_d), indexed by team abbreviation."""

    iterations: int
    """Number of iterations until convergence."""

    residual: float
    """Final Frobenius-norm residual."""


class AlternatingMinimization:
    """Isolate player skill from defensive matchup effects.

    The algorithm works on a *stat* matrix **A** (players × game-slots)
    and an accompanying *opponent* matrix **D** that records which
    defense each player faced in each slot.

    We solve::

        min_{u, w}  || A - u ⊗ w[D] ||_F^2

    by alternating between fixing *w* and solving for *u*, then fixing
    *u* and solving for *w*.
    """

    def __init__(self, config: PredictionConfig | None = None) -> None:
        self.config = config or PredictionConfig()

    def fit(
        self,
        stats: pd.DataFrame,
        opponents: pd.DataFrame,
        player_ids: pd.Series,
        teams: pd.Series,
    ) -> DecompositionResult:
        """Run the decomposition.

        Parameters
        ----------
        stats:
            (n_players × n_weeks) matrix of observed fantasy points.
        opponents:
            Same shape as *stats*, each cell is the opponent team
            abbreviation (or NaN for bye/missing).
        player_ids:
            Length-n_players series mapping row → player_id.
        teams:
            Same length; maps row → player's own team.

        Returns
        -------
        DecompositionResult
        """
        A = stats.values.astype(np.float64)
        n_players, n_slots = A.shape

        # Early return for empty input.
        if n_players == 0 or n_slots == 0:
            return DecompositionResult(
                player_skill=pd.Series(dtype=np.float64, name="skill"),
                defense_multipliers=pd.Series(dtype=np.float64, name="defense_multiplier"),
                iterations=0,
                residual=0.0,
            )

        # Build mask for valid (non-bye) entries.
        mask = opponents.notna().values

        # Map opponent names → integer indices.
        all_teams = sorted(opponents.stack().dropna().unique())
        team_to_idx = {t: i for i, t in enumerate(all_teams)}
        n_teams = len(all_teams)

        # D_idx[i, j] = integer index of the defense player i faced in slot j.
        D_idx = np.full((n_players, n_slots), -1, dtype=np.int32)
        for i in range(n_players):
            for j in range(n_slots):
                opp = opponents.iat[i, j]
                if pd.notna(opp) and opp in team_to_idx:
                    D_idx[i, j] = team_to_idx[opp]

        # ----- Initialisation -----
        u = np.ones(n_players, dtype=np.float64)
        w = np.ones(n_teams, dtype=np.float64)

        # Seed u with each player's mean observed points.
        for i in range(n_players):
            valid = mask[i]
            if valid.any():
                u[i] = A[i, valid].mean()

        max_iter = self.config.alt_min_max_iterations
        tol = self.config.alt_min_convergence_tol

        residual = np.inf
        for iteration in range(1, max_iter + 1):
            # --- Step A: fix w, solve for u ---
            for i in range(n_players):
                numer = 0.0
                denom = 0.0
                for j in range(n_slots):
                    if not mask[i, j]:
                        continue
                    d = D_idx[i, j]
                    wj = w[d] if d >= 0 else 1.0
                    numer += A[i, j] * wj
                    denom += wj * wj
                u[i] = numer / denom if denom > 0 else 0.0

            # --- Step B: fix u, solve for w ---
            for d in range(n_teams):
                numer = 0.0
                denom = 0.0
                locs = np.argwhere(D_idx == d)  # (row, col) pairs
                for i, j in locs:
                    if not mask[i, j]:
                        continue
                    numer += A[i, j] * u[i]
                    denom += u[i] * u[i]
                w[d] = numer / denom if denom > 0 else 1.0

            # --- Convergence check ---
            predicted = np.zeros_like(A)
            for i in range(n_players):
                for j in range(n_slots):
                    if mask[i, j]:
                        d = D_idx[i, j]
                        wj = w[d] if d >= 0 else 1.0
                        predicted[i, j] = u[i] * wj

            new_residual = float(np.sqrt(np.sum((A[mask] - predicted[mask]) ** 2)))
            if abs(residual - new_residual) < tol:
                residual = new_residual
                break
            residual = new_residual

        # Package results.
        skill_series = pd.Series(u, index=player_ids.values, name="skill")
        defense_series = pd.Series(w, index=all_teams, name="defense_multiplier")

        return DecompositionResult(
            player_skill=skill_series,
            defense_multipliers=defense_series,
            iterations=iteration,
            residual=residual,
        )
