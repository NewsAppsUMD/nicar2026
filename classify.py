"""Stage 1: Classify Chicago Sun-Times articles as local government stories.

Reads all articles from the 2026 data directory, uses an Ollama model to
determine whether each article covers local government for the city of Chicago,
its neighborhoods, or Cook County. Outputs a filtered JSON file.

Usage:
    uv run python classify.py \\
        --data-dir /path/to/beat_book_work/chicago-public-media/data/2026 \\
        --model llama3.2 \\
        [--output classified_articles.json] \\
        [--state-file .classify_state.json]
"""

import argparse
import json
import sys
from pathlib import Path

import llm
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

from utils import load_json_files_from_dir, strip_html

console = Console()

SYSTEM_PROMPT = """You are a news article classifier for Chicago local government coverage.

Your job is to determine if an article covers LOCAL GOVERNMENT topics for the city of Chicago,
its neighborhoods, or Cook County. Answer only YES or NO.

Topics that qualify as local government:
- Zoning, land use, development permits, building approvals
- Education: Chicago Public Schools (CPS), school board, school closings, curriculum policy
- Public safety: Chicago Police Department (CPD), crime policy, police reform, fire department
- City budget, spending, TIF districts, property taxes, Cook County finances
- Criminal justice: Cook County courts, state's attorney, public defender, Cook County Jail
- Chicago City Council, aldermen, mayoral administration, Mayor Brandon Johnson
- Cook County government, Cook County Board, county executive
- Chicago Transit Authority (CTA), city infrastructure, public works
- Neighborhood development, affordable housing policy, gentrification policy
- Chicago Public Health, city health department policies

Topics that do NOT qualify:
- National or international politics not directly affecting Chicago governance
- General crime stories without policy angle
- Sports, entertainment, arts (unless involving city contracts or policy)
- Immigration enforcement by federal agencies (unless Chicago/Cook County policy response)
- Business news without a city government angle
- Human interest stories without a governance angle

Be conservative: if unclear, answer NO."""

USER_PROMPT_TEMPLATE = """Article title: {title}

Article content:
{summary}

Is this article about local government for the city of Chicago, its neighborhoods,
or Cook County? Answer only YES or NO."""


def load_state(state_file: Path) -> dict:
    if state_file.exists():
        with open(state_file, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_state(state: dict, state_file: Path) -> None:
    with open(state_file, "w", encoding="utf-8") as f:
        json.dump(state, f)


def classify_article(model, article: dict) -> bool:
    """Return True if the article is a local government story."""
    title = article.get("title", "")
    raw_summary = article.get("summary", "")
    summary_text = strip_html(raw_summary)[:2000]

    prompt = USER_PROMPT_TEMPLATE.format(title=title, summary=summary_text)
    try:
        response = model.prompt(prompt, system=SYSTEM_PROMPT)
        answer = response.text().strip().upper()
        return answer.startswith("YES")
    except Exception as exc:
        console.print(f"[yellow]Warning: classification error for '{title[:60]}': {exc}[/yellow]")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Classify Chicago Sun-Times articles as local government stories."
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        type=Path,
        help="Path to the data/2026 directory containing YYYY-MM-DD subdirectories.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Ollama model name to use for classification (e.g. llama3.2, mistral).",
    )
    parser.add_argument(
        "--output",
        default="classified_articles.json",
        type=Path,
        help="Output JSON file path (default: classified_articles.json).",
    )
    parser.add_argument(
        "--state-file",
        default=".classify_state.json",
        type=Path,
        help="State file for resuming interrupted runs (default: .classify_state.json).",
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    if not data_dir.exists():
        console.print(f"[red]Error: data directory not found: {data_dir}[/red]")
        sys.exit(1)

    console.print(f"[bold]Stage 1: Classification[/bold]")
    console.print(f"  Data dir : {data_dir}")
    console.print(f"  Model    : {args.model}")
    console.print(f"  Output   : {args.output}")
    console.print()

    # Load articles
    console.print("Loading articles from data directory...")
    articles = load_json_files_from_dir(data_dir)
    console.print(f"  Found {len(articles)} total articles across all date folders.")
    console.print()

    # Load state (already-classified article IDs)
    state = load_state(args.state_file)
    already_done = sum(1 for v in state.values() if v is not None)
    if already_done:
        console.print(f"  Resuming: {already_done} articles already classified, skipping.")

    # Load model
    try:
        model = llm.get_model(args.model)
    except Exception as exc:
        console.print(f"[red]Error loading model '{args.model}': {exc}[/red]")
        console.print("  Make sure the model is pulled via Ollama and llm-ollama is installed.")
        sys.exit(1)

    # Classify articles
    local_gov_articles = []
    skipped = 0

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Classifying...", total=len(articles))

        for article in articles:
            article_id = article.get("id") or article.get("link", "")

            if article_id in state:
                # Already processed — use cached result
                if state[article_id]:
                    local_gov_articles.append(article)
                skipped += 1
                progress.advance(task)
                continue

            is_local_gov = classify_article(model, article)
            state[article_id] = is_local_gov
            save_state(state, args.state_file)

            if is_local_gov:
                local_gov_articles.append(article)

            progress.advance(task)

    console.print()
    console.print(f"[bold green]Classification complete.[/bold green]")
    console.print(f"  Total input   : {len(articles)}")
    console.print(f"  Local gov     : {len(local_gov_articles)}")
    console.print(
        f"  Filtered out  : {len(articles) - len(local_gov_articles)}"
    )
    if skipped:
        console.print(f"  Skipped (cached): {skipped}")

    # Write output
    output = {
        "total_input": len(articles),
        "total_classified": len(local_gov_articles),
        "model": args.model,
        "articles": local_gov_articles,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    console.print(f"\nWrote {len(local_gov_articles)} articles to [bold]{args.output}[/bold]")


if __name__ == "__main__":
    main()
