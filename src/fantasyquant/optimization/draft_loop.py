"""Interactive draft loop.

Accepts real-time user input ("Player X was taken by Team Y") and
re-runs the MIP solver each time the user is on the clock.
"""

from __future__ import annotations

import click
import pandas as pd

from fantasyquant.config import EngineConfig, DEFAULT_CONFIG
from fantasyquant.optimization.solver import (
    DraftSolver,
    DraftState,
    PlayerPool,
    Recommendation,
)


def _fuzzy_match(query: str, names: pd.Series) -> str | None:
    """Case-insensitive substring match against player names."""
    query_lower = query.strip().lower()
    for pid, name in names.items():
        if query_lower in str(name).lower():
            return str(pid)
    return None


class DraftLoop:
    """Manages the stateful draft session."""

    def __init__(
        self,
        pool: PlayerPool,
        config: EngineConfig = DEFAULT_CONFIG,
        my_slot: int = 1,
    ) -> None:
        self.solver = DraftSolver(pool, config)
        self.pool = pool
        self.config = config
        self.state = DraftState()
        self.my_slot = my_slot  # 1-indexed draft position
        self.total_teams = config.roster.teams
        self.total_rounds = config.roster.rounds

        # Lookup helpers.
        self._name_lookup = pd.Series(
            pool.info["player_name"].values,
            index=pool.info["player_id"].values,
        )

    @property
    def _is_my_pick(self) -> bool:
        """Snake draft: determine if the current overall pick belongs to us."""
        pick = self.state.current_pick
        rnd = (pick - 1) // self.total_teams + 1
        pos_in_round = (pick - 1) % self.total_teams + 1
        if rnd % 2 == 1:  # odd round → normal order
            return pos_in_round == self.my_slot
        else:  # even round → reversed
            return pos_in_round == (self.total_teams - self.my_slot + 1)

    def mark_taken(self, player_query: str) -> str | None:
        """Mark a player as taken by another team.

        Returns the resolved player_id or *None* on no match.
        """
        pid = _fuzzy_match(player_query, self._name_lookup)
        if pid is None:
            return None
        self.state.taken.add(pid)
        self.state.current_pick += 1
        return pid

    def make_my_pick(self) -> Recommendation | None:
        """Run the solver and return the optimal pick."""
        rec = self.solver.solve(self.state)
        if rec is None:
            return None
        # Register the pick.
        self.state.my_picks.append(rec.player_id)
        self.state.taken.add(rec.player_id)
        self.state.current_pick += 1
        return rec

    def my_roster_summary(self) -> pd.DataFrame:
        """Return a summary DataFrame of our current roster."""
        rows = []
        info = self.pool.info
        for pid in self.state.my_picks:
            mask = info["player_id"] == pid
            if mask.any():
                row = info[mask].iloc[0]
                rows.append({
                    "player_id": pid,
                    "player_name": row["player_name"],
                    "position": row["position"],
                    "team": row["team"],
                })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # CLI runner
    # ------------------------------------------------------------------

    def run_interactive(self) -> None:
        """Run the draft as an interactive CLI session."""
        click.echo("=" * 60)
        click.echo("  FantasyQuant Draft Engine")
        click.echo(f"  Your slot: {self.my_slot} / {self.total_teams}")
        click.echo(f"  Rounds: {self.total_rounds}")
        click.echo("=" * 60)
        click.echo()
        click.echo("Commands:")
        click.echo("  <player name>  — mark a player as drafted by opponent")
        click.echo("  pick           — solver recommends + makes your pick")
        click.echo("  roster         — show your current roster")
        click.echo("  quit           — exit")
        click.echo()

        while self.state.my_roster_size < self.total_rounds:
            pick_num = self.state.current_pick
            rnd = (pick_num - 1) // self.total_teams + 1

            if self._is_my_pick:
                click.secho(
                    f"\n>>> Pick #{pick_num} (Round {rnd}) — YOU ARE ON THE CLOCK",
                    fg="green", bold=True,
                )
                click.echo("Type 'pick' to get the solver's recommendation.")
            else:
                click.echo(f"\n--- Pick #{pick_num} (Round {rnd}) ---")

            cmd = click.prompt("Input", default="").strip()

            if cmd.lower() == "quit":
                break
            elif cmd.lower() == "roster":
                click.echo(self.my_roster_summary().to_string(index=False))
                continue
            elif cmd.lower() == "pick":
                if not self._is_my_pick:
                    click.secho("It's not your pick yet!", fg="red")
                    continue
                click.echo("Solving...")
                rec = self.make_my_pick()
                if rec is None:
                    click.secho("No feasible pick found.", fg="red")
                    continue
                click.secho(
                    f"  >>> PICK: {rec.player_name} ({rec.position})",
                    fg="cyan", bold=True,
                )
                click.echo(
                    f"      E[wins]={rec.expected_wins:.1f}  "
                    f"E[pts]={rec.expected_points:.0f}  "
                    f"obj={rec.objective_value:.2f}",
                )
            elif cmd:
                pid = self.mark_taken(cmd)
                if pid is None:
                    click.secho(f"Player not found: '{cmd}'", fg="yellow")
                    continue
                name = self._name_lookup.get(pid, pid)
                click.echo(f"  Marked {name} as taken.")
            # else: empty input, re-prompt

        click.echo("\nDraft complete! Final roster:")
        click.echo(self.my_roster_summary().to_string(index=False))
