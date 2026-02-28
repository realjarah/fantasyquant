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
        opp_flat = pd.Series(opponents.values.ravel()).dropna().unique()
        all_teams = sorted(opp_flat)
        team_to_idx = {t: i for i, t in enumerate(all_teams)}
        n_teams = len(all_teams)

        # D_idx[i, j] = integer index of the defense player i faced in slot j.
        opp_values = opponents.values
        D_idx = np.full((n_players, n_slots), -1, dtype=np.int32)
        for team_name, idx in team_to_idx.items():
            D_idx[opp_values == team_name] = idx

        # Valid entries: mask True and opponent mapped.
        valid = mask & (D_idx >= 0)
        # Safe index array for w[...] lookups (replace -1 with 0 to avoid
        # index errors; invalid entries are zeroed out via `valid`).
        safe_idx = np.clip(D_idx, 0, n_teams - 1)

        # ----- Initialisation -----
        w = np.ones(n_teams, dtype=np.float64)

        # Seed u with each player's mean observed points.
        valid_counts = valid.sum(axis=1)
        A_valid = np.where(valid, A, 0.0)
        u = np.where(valid_counts > 0, A_valid.sum(axis=1) / valid_counts, 1.0)

        max_iter = self.config.alt_min_max_iterations
        tol = self.config.alt_min_convergence_tol

        # Pre-flatten for Step B scatter operations.
        d_flat = D_idx.ravel()
        valid_flat = valid.ravel()
        valid_indices = np.where(valid_flat)[0]
        d_valid = d_flat[valid_indices]

        residual = np.inf
        for iteration in range(1, max_iter + 1):
            # --- Step A: fix w, solve for u ---
            # W_mat[i,j] = w[D_idx[i,j]] for valid entries, 0 otherwise.
            W_mat = np.where(valid, w[safe_idx], 0.0)
            numer_u = (A * W_mat).sum(axis=1)
            denom_u = (W_mat ** 2).sum(axis=1)
            u = np.divide(numer_u, denom_u, where=denom_u > 0,
                          out=np.zeros(n_players, dtype=np.float64))

            # --- Step B: fix u, solve for w ---
            # For each valid entry, look up the row's u value.
            row_idx = valid_indices // n_slots
            u_valid = u[row_idx]
            a_valid = A.ravel()[valid_indices]
            au_flat = a_valid * u_valid
            uu_flat = u_valid ** 2

            w_numer = np.zeros(n_teams, dtype=np.float64)
            w_denom = np.zeros(n_teams, dtype=np.float64)
            np.add.at(w_numer, d_valid, au_flat)
            np.add.at(w_denom, d_valid, uu_flat)
            w = np.divide(w_numer, w_denom, where=w_denom > 0,
                          out=np.ones(n_teams, dtype=np.float64))

            # --- Convergence check ---
            W_check = np.where(valid, w[safe_idx], 0.0)
            predicted = u[:, np.newaxis] * W_check
            new_residual = float(np.sqrt(np.sum((A[valid] - predicted[valid]) ** 2)))
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
