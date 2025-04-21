"""Main script for Wise Nutrition application."""

import asyncio
from typing import Optional
import sys
from pathlib import Path

import typer

from wise_nutrition.agent import NutritionAgent
from wise_nutrition.cli import console, display_nutrition_response, display_recipe_response

app = typer.Typer(help="Wise Nutrition - AI-powered nutrition and recipe advisor")


async def run_interactive_session(data_dir: Optional[Path] = None):
    """Run an interactive session with the Wise Nutrition agent.
    
    Args:
        data_dir: Optional data directory path
    """
    # Initialize agent
    agent = NutritionAgent(data_dir=data_dir)
    
    console.print(
        "[bold green]Wise Nutrition AI[/bold green] - Your nutrition and recipe assistant\n"
        "Ask questions about nutrition, vitamins, minerals, or request recipes.\n"
        "Type [bold]exit[/bold] to quit.\n"
    )
    
    while True:
        try:
            # Get user query
            query = console.input("[bold cyan]Your question:[/bold cyan] ")
            
            # Check for exit
            if query.lower() in ("exit", "quit", "q"):
                console.print("[bold green]Thank you for using Wise Nutrition AI![/bold green]")
                break
            
            # Skip empty queries
            if not query.strip():
                continue
            
            # Process query
            with console.status("[bold green]Processing your question...[/bold green]"):
                response = await agent.process_query(query)
            
            # Display response
            if response.response_type == "nutrition":
                display_nutrition_response(response)
            else:
                display_recipe_response(response)
            
            console.print()  # Add spacing between interactions
            
        except KeyboardInterrupt:
            console.print("\n[bold yellow]Session terminated.[/bold yellow]")
            break
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {str(e)}")


@app.command()
def interactive(
    data_dir: Optional[Path] = typer.Option(
        None,
        "--data-dir", "-d",
        help="Path to data directory (defaults to project data directory)"
    ),
):
    """Start an interactive session with the Wise Nutrition AI assistant."""
    try:
        asyncio.run(run_interactive_session(data_dir))
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Session terminated.[/bold yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        sys.exit(1)


@app.command()
def query(
    query_text: str = typer.Argument(..., help="Your nutrition question or recipe request"),
    data_dir: Optional[Path] = typer.Option(
        None,
        "--data-dir", "-d",
        help="Path to data directory (defaults to project data directory)"
    ),
):
    """Query the Wise Nutrition agent with a specific question."""
    from wise_nutrition.cli import _run_query
    
    try:
        asyncio.run(_run_query(query_text, data_dir))
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Query cancelled.[/bold yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    app()
