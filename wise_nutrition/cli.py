"""Command-line interface for Wise Nutrition."""

import asyncio
from pathlib import Path
import sys
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from wise_nutrition.agent import NutritionAgent, WiseNutritionResponse

app = typer.Typer(help="Wise Nutrition CLI")
console = Console()


def get_project_root() -> Path:
    """Get the project root directory.
    
    Returns:
        Path to project root
    """
    return Path(__file__).parent.parent


async def _run_query(query: str, data_dir: Optional[Path] = None) -> None:
    """Run a query against the nutrition agent.
    
    Args:
        query: User query
        data_dir: Data directory path
    """
    # Use project data directory by default
    if data_dir is None:
        data_dir = get_project_root() / "data"
    
    # Initialize agent
    agent = NutritionAgent(data_dir=data_dir)
    
    # Process query
    with console.status("[bold green]Processing your nutrition query..."):
        response = await agent.process_query(query)
    
    # Display response based on type
    if response.response_type == "nutrition":
        display_nutrition_response(response)
    else:
        display_recipe_response(response)


def display_nutrition_response(response: WiseNutritionResponse) -> None:
    """Display a nutrition response in a rich format.
    
    Args:
        response: The nutrition response to display
    """
    nutrition_response = response.nutrition_response
    if not nutrition_response:
        console.print("[bold red]Error:[/bold red] No nutrition response available.")
        return
    
    # Display main answer
    console.print(Panel(
        Markdown(nutrition_response.answer),
        title="[bold cyan]Nutrition Information[/bold cyan]",
        expand=False
    ))
    
    # Display sources if available
    if nutrition_response.sources:
        console.print("\n[bold]Sources:[/bold]")
        for source in nutrition_response.sources:
            console.print(f"  • {source}")
    
    # Display nutrient-rich foods if available
    if nutrition_response.nutrient_rich_foods:
        console.print("\n[bold cyan]Foods Rich in Nutrients:[/bold cyan]")
        for food_group in nutrition_response.nutrient_rich_foods:
            console.print(f"[bold]{food_group.nutrient}[/bold]")
            for food in food_group.foods:
                console.print(f"  • {food}")
            if food_group.daily_needs:
                console.print(f"  Daily needs: {food_group.daily_needs}")
    
    # Display recipe recommendations if available
    if nutrition_response.recipe_recommendations:
        console.print("\n[bold cyan]Recommended Recipes:[/bold cyan]")
        for recipe in nutrition_response.recipe_recommendations:
            console.print(f"  • {recipe}")
    
    # Display follow-up questions
    if nutrition_response.follow_up_questions:
        console.print("\n[bold cyan]You might also want to ask:[/bold cyan]")
        for question in nutrition_response.follow_up_questions:
            console.print(f"  • {question}")


def display_recipe_response(response: WiseNutritionResponse) -> None:
    """Display a recipe response in a rich format.
    
    Args:
        response: The recipe response to display
    """
    recipe_response = response.recipe_response
    if not recipe_response:
        console.print("[bold red]Error:[/bold red] No recipe response available.")
        return
    
    # Display introduction
    console.print(Panel(
        Markdown(recipe_response.introduction),
        title="[bold green]Recipe Recommendations[/bold green]",
        expand=False
    ))
    
    # Display recipes
    for recipe in recipe_response.recipes:
        console.print(Panel(
            f"[bold]{recipe.name}[/bold]\n\n"
            f"{recipe.description}\n\n"
            f"[bold]Ingredients:[/bold]\n"
            + "\n".join(f"  • {ingredient}" for ingredient in recipe.ingredients)
            + f"\n\n[bold]Instructions:[/bold]\n{recipe.instructions}",
            title=f"[bold green]{recipe.name}[/bold green]",
            expand=False
        ))
    
    # Display nutrition notes
    console.print(Panel(
        Markdown(recipe_response.nutrition_notes),
        title="[bold green]Nutritional Benefits[/bold green]",
        expand=False
    ))
    
    # Display preparation tips if available
    if recipe_response.preparation_tips:
        console.print(Panel(
            Markdown(recipe_response.preparation_tips),
            title="[bold green]Preparation Tips[/bold green]",
            expand=False
        ))


@app.command()
def query(
    query_text: str = typer.Argument(..., help="Your nutrition question or recipe request"),
    data_dir: Optional[Path] = typer.Option(
        None,
        "--data-dir", "-d",
        help="Path to data directory (defaults to project data directory)"
    ),
):
    """Query the Wise Nutrition agent with your nutrition question or recipe request."""
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
