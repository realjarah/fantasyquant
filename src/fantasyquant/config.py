"""Central configuration for the FantasyQuant engine.

A league's rules are the foundation — scoring, roster slots, and
platform quirks flow through projections, the solver, and lineup
management.  Users configure via a ``league.json`` file or CLI flags.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field


def _upcoming_season() -> int:
    """Return the upcoming NFL season year.

    The NFL season runs Sep–Feb.  Once the Super Bowl is over (early Feb),
    attention shifts to the next season: free agency, the draft, and
    fantasy prep.  We use the current calendar year from February onward;
    only in January (while the playoffs / Super Bowl are still on) do we
    reference the prior year's season.
    """
    today = datetime.date.today()
    return today.year if today.month >= 2 else today.year - 1


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScoringSettings:
    """Fantasy point weights per stat category.

    Covers the three standard formats (PPR / half-PPR / standard) plus
    platform-specific extras like passing-TD bonuses (4 vs 6 pt).
    """

    passing_yards: float = 0.04
    passing_tds: float = 4.0
    interceptions: float = -2.0
    rushing_yards: float = 0.1
    rushing_tds: float = 6.0
    receptions: float = 1.0        # 1.0 = PPR, 0.5 = half-PPR, 0.0 = standard
    receiving_yards: float = 0.1
    receiving_tds: float = 6.0
    fumbles_lost: float = -2.0
    two_point_conversions: float = 2.0
    passing_2pt: float = 2.0
    te_reception_bonus: float = 0.0  # Extra per TE reception (e.g. 0.5 for TE premium)

    @property
    def reception_format(self) -> str:
        if self.receptions >= 1.0:
            return "PPR"
        elif self.receptions >= 0.5:
            return "Half-PPR"
        return "Standard"

    @property
    def format_tag(self) -> str:
        """Short human-readable tag, e.g. 'PPR TEP' or 'Half-PPR 6ptPass'."""
        parts = [self.reception_format]
        if self.te_reception_bonus > 0:
            parts.append("TEP")
        if self.passing_tds >= 6.0:
            parts.append("6ptPass")
        return " ".join(parts)


# ---------------------------------------------------------------------------
# Roster
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RosterSettings:
    """Roster construction rules for a fantasy league."""

    teams: int = 10
    rounds: int = 15
    qb: int = 1
    rb: int = 2
    wr: int = 2
    te: int = 1
    flex: int = 1         # RB/WR/TE
    superflex: int = 0    # QB/RB/WR/TE — changes QB scarcity dramatically
    bench: int = 6
    dst: int = 1
    k: int = 1

    @property
    def total_slots(self) -> int:
        return (
            self.qb + self.rb + self.wr + self.te
            + self.flex + self.superflex
            + self.bench + self.dst + self.k
        )

    @property
    def starters(self) -> int:
        return (
            self.qb + self.rb + self.wr + self.te
            + self.flex + self.superflex
            + self.dst + self.k
        )


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PredictionConfig:
    """Tunables for the prediction engine."""

    training_seasons: int = 3
    historical_weight: float = 0.5
    vegas_weight: float = 0.5
    alt_min_max_iterations: int = 50
    alt_min_convergence_tol: float = 1e-6
    weeks_in_season: int = 17


# ---------------------------------------------------------------------------
# Optimization
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OptimizationConfig:
    """Tunables for the MIP solver.

    Follows Becker & Sun (2013) notation:
        λ₀ × total_points  +  λ₁ × regular_wins  +  λ₂ × playoff_wins
    """

    lambda_points: float = 1.0       # λ₀: total fantasy points
    lambda_wins: float = 100.0       # λ₁: regular season wins (weeks 1-15)
    lambda_playoff: float = 150.0    # λ₂: playoff wins (weeks 16-17)
    alpha: float = 1.0               # opponent-draft uncertainty (constraint 1b)
    adp_uncertainty_rounds: float = 1.5  # rounds of ADP slack
    solver_time_limit_seconds: int = 30
    regular_season_weeks: int = 15   # weeks 1-15 are regular season
    playoff_weeks: tuple[int, ...] = (16, 17)


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------

@dataclass
class EngineConfig:
    scoring: ScoringSettings = field(default_factory=ScoringSettings)
    roster: RosterSettings = field(default_factory=RosterSettings)
    prediction: PredictionConfig = field(default_factory=PredictionConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    nfl_weeks: int = 17
    current_season: int = field(default_factory=_upcoming_season)
    platform: str = "custom"   # e.g. "espn", "yahoo", "sleeper", "nfl"


DEFAULT_CONFIG = EngineConfig()
