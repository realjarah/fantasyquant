"""Command-line interface for FantasyQuant."""

from __future__ import annotations

import json
from pathlib import Path

import click

from fantasyquant.config import EngineConfig, DEFAULT_CONFIG
from fantasyquant.league import (
    PRESETS,
    describe_config,
    league_to_dict,
    list_presets,
    load_league,
    save_league_template,
)


# ------------------------------------------------------------------
# Shared: --league / --preset resolution
# ------------------------------------------------------------------

def _load_config(
    league_path: str | None = None,
    preset: str | None = None,
    season: int | None = None,
) -> EngineConfig:
    """Build config from --preset and/or --league file.

    Layering: preset defaults → file overrides → CLI season override.
    Either, both, or neither can be provided.
    """
    if league_path or preset:
        config = load_league(league_path, preset=preset)
    else:
        config = EngineConfig()
    if season is not None:
        config.current_season = season
    return config


# ------------------------------------------------------------------
# CLI group
# ------------------------------------------------------------------

@click.group()
@click.version_option(package_name="fantasyquant")
def main() -> None:
    """FantasyQuant: Institutional-grade fantasy football analytics."""


# ------------------------------------------------------------------
# init
# ------------------------------------------------------------------

@main.command()
@click.option("--preset", type=click.Choice(list_presets(), case_sensitive=False),
              default=None, help="Start from a platform preset.")
@click.option("--output", "-o", type=click.Path(), default="league.json",
              help="Output file path.")
def init(preset: str | None, output: str) -> None:
    """Generate a league.json configuration file.

    Start from a platform preset or answer prompts to configure
    your league's scoring and roster rules.
    """
    if preset:
        path = save_league_template(output, preset=preset)
        click.echo(f"Created {path} from preset '{preset}'")
        config = load_league(preset=preset)
    else:
        click.echo("Let's set up your league.\n")

        # Platform
        platforms = ["espn", "yahoo", "sleeper", "nfl", "underdog", "custom"]
        for i, p in enumerate(platforms, 1):
            click.echo(f"  {i}. {p}")
        choice = click.prompt("Platform", type=int, default=1)
        platform = platforms[min(choice, len(platforms)) - 1]

        # Teams
        teams = click.prompt("Number of teams", type=int, default=12)

        # Scoring format
        formats = {"1": ("PPR", 1.0), "2": ("Half-PPR", 0.5), "3": ("Standard", 0.0)}
        click.echo("\nScoring format:")
        for k, (name, _) in formats.items():
            click.echo(f"  {k}. {name}")
        fmt_choice = click.prompt("Format", default="1")
        _, reception_pts = formats.get(fmt_choice, ("PPR", 1.0))

        # Passing TDs
        pass_td = click.prompt("Points per passing TD", type=float, default=4.0)

        # Superflex
        has_superflex = click.confirm("Superflex league?", default=False)

        # Build config
        config = EngineConfig(
            platform=platform,
            scoring=DEFAULT_CONFIG.scoring.__class__(
                receptions=reception_pts,
                passing_tds=pass_td,
            ),
            roster=DEFAULT_CONFIG.roster.__class__(
                teams=teams,
                superflex=1 if has_superflex else 0,
            ),
        )

        data = league_to_dict(config)
        path = Path(output)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
        click.echo(f"\nCreated {path}")

    click.echo(f"\n  Platform:  {config.platform}")
    click.echo(f"  Teams:     {config.roster.teams}")
    click.echo(f"  Scoring:   {config.scoring.reception_format}")
    click.echo(f"  Pass TD:   {config.scoring.passing_tds} pts")
    click.echo(f"  Superflex: {'Yes' if config.roster.superflex else 'No'}")
    click.echo(f"  Starters:  {config.roster.starters}")
    click.echo(f"\nEdit {output} to fine-tune, then pass --league {output} to any command.")


# ------------------------------------------------------------------
# presets
# ------------------------------------------------------------------

@main.command()
def presets() -> None:
    """List available platform presets."""
    click.echo("Available presets:\n")
    for name in list_presets():
        config = load_league(preset=name)
        click.echo(f"  {name:27s} {describe_config(config)}")
    click.echo(f"\nUsage: fq draft --preset <name>")
    click.echo(f"       fq init --preset <name>  (to generate a league.json for customization)")


# ------------------------------------------------------------------
# project
# ------------------------------------------------------------------

@main.command()
@click.option("--preset", type=click.Choice(list_presets(), case_sensitive=False),
              default=None, help="Platform preset (e.g. espn_ppr, sleeper_superflex).")
@click.option("--league", "league_path", type=click.Path(exists=True), default=None,
              help="Path to league.json (overrides preset values).")
@click.option("--season", default=None, type=int,
              help="Override season year.")
@click.option("--win-totals", type=click.Path(exists=True), default=None,
              help="Path to win totals JSON/CSV.")
@click.option("--player-props", type=click.Path(exists=True), default=None,
              help="Path to player props JSON/CSV.")
@click.option("--output", "-o", type=click.Path(), default="projections.csv",
              help="Output file for the f(i,t) matrix.")
def project(
    preset: str | None,
    league_path: str | None,
    season: int | None,
    win_totals: str | None,
    player_props: str | None,
    output: str,
) -> None:
    """Build the weekly projection matrix f(i,t)."""
    from fantasyquant.prediction.projections import build_projections

    config = _load_config(league_path, preset, season)
    if season is None:
        season = config.current_season

    click.echo(f"Building projections for {season} season...")
    click.echo(f"  {describe_config(config)}")

    result = build_projections(
        config,
        win_totals_source=win_totals,
        player_props_source=player_props,
    )

    result.weekly_projections.to_csv(output)
    click.echo(f"Wrote {len(result.weekly_projections)} player projections to {output}")

    info_path = Path(output).with_suffix(".info.csv")
    result.player_info.to_csv(info_path, index=False)
    click.echo(f"Wrote player metadata to {info_path}")


# ------------------------------------------------------------------
# draft
# ------------------------------------------------------------------

@main.command()
@click.option("--preset", type=click.Choice(list_presets(), case_sensitive=False),
              default=None, help="Platform preset (e.g. espn_ppr, sleeper_superflex).")
@click.option("--league", "league_path", type=click.Path(exists=True), default=None,
              help="Path to league.json (overrides preset values).")
@click.option("--season", default=None, type=int)
@click.option("--slot", default=1, show_default=True,
              help="Your draft position (1-indexed).")
@click.option("--projections", type=click.Path(exists=True), default=None,
              help="Pre-built projections CSV. If omitted, builds fresh.")
@click.option("--adp", type=click.Path(exists=True), default=None,
              help="ADP data CSV (columns: player_id, adp).")
def draft(
    preset: str | None,
    league_path: str | None,
    season: int | None,
    slot: int,
    projections: str | None,
    adp: str | None,
) -> None:
    """Launch the interactive draft assistant."""
    import pandas as pd

    from fantasyquant.optimization.draft_loop import DraftLoop
    from fantasyquant.optimization.solver import PlayerPool

    config = _load_config(league_path, preset, season)

    click.echo(f"  {describe_config(config)}")

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
@click.option("--preset", type=click.Choice(list_presets(), case_sensitive=False),
              default=None, help="Platform preset (e.g. espn_ppr, sleeper_superflex).")
@click.option("--league", "league_path", type=click.Path(exists=True), default=None,
              help="Path to league.json (overrides preset values).")
@click.option("--test-season", required=True, type=int,
              help="Season to test against (e.g. 2023).")
@click.option("--slot", default=1, show_default=True,
              help="Simulated draft position.")
def backtest(preset: str | None, league_path: str | None, test_season: int, slot: int) -> None:
    """Run a historical backtest."""
    from fantasyquant.backtest.simulator import backtest as run_backtest

    config = _load_config(league_path, preset, season=test_season)
    train_end = test_season - 1
    train_start = train_end - config.prediction.training_seasons + 1
    training = list(range(train_start, train_end + 1))

    click.echo(f"Backtesting: train on {training}, test on {test_season}")
    click.echo(f"  {describe_config(config)}  |  Draft slot: {slot}")

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
