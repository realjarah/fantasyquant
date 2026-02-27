"""Command-line interface for FantasyQuant."""

from __future__ import annotations

import json
from pathlib import Path

import click

from fantasyquant.config import EngineConfig, DEFAULT_CONFIG


@click.group()
@click.version_option(package_name="fantasyquant")
def main() -> None:
    """FantasyQuant: Institutional-grade fantasy football analytics."""


# ------------------------------------------------------------------
# project
# ------------------------------------------------------------------

@main.command()
@click.option("--season", default=DEFAULT_CONFIG.current_season, show_default=True)
@click.option("--win-totals", type=click.Path(exists=True), default=None,
              help="Path to win totals JSON/CSV.")
@click.option("--player-props", type=click.Path(exists=True), default=None,
              help="Path to player props JSON/CSV.")
@click.option("--output", "-o", type=click.Path(), default="projections.csv",
              help="Output file for the f(i,t) matrix.")
def project(season: int, win_totals: str | None, player_props: str | None, output: str) -> None:
    """Build the weekly projection matrix f(i,t)."""
    from fantasyquant.prediction.projections import build_projections

    config = EngineConfig(current_season=season)
    click.echo(f"Building projections for {season} season...")

    result = build_projections(
        config,
        win_totals_source=win_totals,
        player_props_source=player_props,
    )

    result.weekly_projections.to_csv(output)
    click.echo(f"Wrote {len(result.weekly_projections)} player projections to {output}")

    # Also dump player info.
    info_path = Path(output).with_suffix(".info.csv")
    result.player_info.to_csv(info_path, index=False)
    click.echo(f"Wrote player metadata to {info_path}")


# ------------------------------------------------------------------
# draft
# ------------------------------------------------------------------

@main.command()
@click.option("--season", default=DEFAULT_CONFIG.current_season, show_default=True)
@click.option("--slot", default=1, show_default=True,
              help="Your draft position (1-indexed).")
@click.option("--teams", default=DEFAULT_CONFIG.roster.teams, show_default=True)
@click.option("--rounds", default=DEFAULT_CONFIG.roster.rounds, show_default=True)
@click.option("--projections", type=click.Path(exists=True), default=None,
              help="Pre-built projections CSV. If omitted, builds fresh.")
@click.option("--adp", type=click.Path(exists=True), default=None,
              help="ADP data CSV (columns: player_id, adp).")
def draft(
    season: int,
    slot: int,
    teams: int,
    rounds: int,
    projections: str | None,
    adp: str | None,
) -> None:
    """Launch the interactive draft assistant."""
    import pandas as pd

    from fantasyquant.optimization.draft_loop import DraftLoop
    from fantasyquant.optimization.solver import PlayerPool

    config = EngineConfig(current_season=season)
    config.roster = config.roster.__class__(teams=teams, rounds=rounds)

    if projections:
        proj_df = pd.read_csv(projections, index_col=0)
        proj_df.columns = [int(c) for c in proj_df.columns]
        info_path = Path(projections).with_suffix(".info.csv")
        info_df = pd.read_csv(info_path) if info_path.exists() else pd.DataFrame()
    else:
        from fantasyquant.prediction.projections import build_projections

        click.echo("Building projections (this may take a minute)...")
        result = build_projections(config)
        proj_df = result.weekly_projections
        info_df = result.player_info

    adp_series = None
    if adp:
        adp_df = pd.read_csv(adp)
        adp_series = pd.Series(adp_df["adp"].values, index=adp_df["player_id"])

    pool = PlayerPool(projections=proj_df, info=info_df, adp=adp_series)
    loop = DraftLoop(pool, config, my_slot=slot)
    loop.run_interactive()


# ------------------------------------------------------------------
# backtest
# ------------------------------------------------------------------

@main.command()
@click.option("--test-season", required=True, type=int,
              help="Season to test against (e.g. 2023).")
@click.option("--slot", default=1, show_default=True,
              help="Simulated draft position.")
def backtest(test_season: int, slot: int) -> None:
    """Run a historical backtest."""
    from fantasyquant.backtest.simulator import backtest as run_backtest

    config = EngineConfig(current_season=test_season)
    train_end = test_season - 1
    train_start = train_end - config.prediction.training_seasons + 1
    training = list(range(train_start, train_end + 1))

    click.echo(f"Backtesting: train on {training}, test on {test_season}")
    click.echo(f"Draft slot: {slot}")

    result = run_backtest(training, test_season, config, my_slot=slot)

    click.echo(f"\nRecord: {result.record}")
    click.echo(f"Total points: {result.total_points:.1f}")
    click.echo(f"Roster: {len(result.roster)} players")

    click.echo("\nWeekly results:")
    for w in sorted(result.weekly_scores):
        my = result.weekly_scores[w]
        opp = result.weekly_opponent_scores.get(w, 0)
        outcome = "W" if my > opp else "L"
        click.echo(f"  Week {w:2d}: {my:6.1f} vs {opp:6.1f}  [{outcome}]")


# ------------------------------------------------------------------
# multipliers
# ------------------------------------------------------------------

@main.command()
@click.option("--win-totals", type=click.Path(exists=True), default=None)
@click.option("--output", "-o", type=click.Path(), default=None)
def multipliers(win_totals: str | None, output: str | None) -> None:
    """Display or export defensive multipliers."""
    from fantasyquant.data.vegas import load_win_totals
    from fantasyquant.prediction.vegas_multipliers import compute_vegas_multipliers

    wt = load_win_totals(source=win_totals)
    mults = compute_vegas_multipliers(wt)

    if output:
        mults.to_csv(output)
        click.echo(f"Wrote multipliers to {output}")
    else:
        click.echo("Team  | W_vegas")
        click.echo("------+--------")
        for team, val in mults.items():
            click.echo(f"{team:5s} | {val:.3f}")
