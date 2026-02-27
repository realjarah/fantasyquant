"""Central configuration for the FantasyQuant engine."""

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScoringSettings:
    """PPR scoring weights (easily swappable to half-PPR or standard)."""

    passing_yards: float = 0.04
    passing_tds: float = 4.0
    interceptions: float = -2.0
    rushing_yards: float = 0.1
    rushing_tds: float = 6.0
    receptions: float = 1.0  # PPR
    receiving_yards: float = 0.1
    receiving_tds: float = 6.0
    fumbles_lost: float = -2.0


# ---------------------------------------------------------------------------
# Roster
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RosterSettings:
    """Standard roster construction for a 10-team league."""

    teams: int = 10
    rounds: int = 15
    qb: int = 1
    rb: int = 2
    wr: int = 2
    te: int = 1
    flex: int = 1  # RB/WR/TE
    bench: int = 6
    dst: int = 1
    k: int = 1

    @property
    def total_slots(self) -> int:
        return self.qb + self.rb + self.wr + self.te + self.flex + self.bench + self.dst + self.k

    @property
    def starters(self) -> int:
        return self.qb + self.rb + self.wr + self.te + self.flex + self.dst + self.k


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
    current_season: int = 2025


DEFAULT_CONFIG = EngineConfig()
