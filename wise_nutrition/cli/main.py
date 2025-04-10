"""
Main CLI entry point for Wise Nutrition.
"""
import click

from wise_nutrition.cli.embed import embed


@click.group()
def cli():
    """Wise Nutrition CLI tools."""
    pass


# Add subcommands
cli.add_command(embed)


if __name__ == "__main__":
    cli() 