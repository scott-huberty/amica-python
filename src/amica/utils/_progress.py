"""Rich progress helpers for AMICA optimization."""

from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


def make_progress_bar(*, total: int, lrate: float) -> tuple[Progress, int]:
    """Create and start the compact AMICA optimization progress bar."""
    progress = Progress(
        TextColumn("[bold]AMICA[/bold]"),
        BarColumn(complete_style="#9A5CD0"),
        TextColumn("{task.completed}/{task.total}"),
        TextColumn(
            "[dim]LL={task.fields[ll]} nd={task.fields[nd]} "
            "lrate={task.fields[lrate]}[/dim]"
        ),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        transient=True,
    )
    progress.start()
    task_id = progress.add_task(
        "amica-optimize",
        total=total,
        ll="--",
        nd="--",
        lrate=f"{lrate:.5f}",
    )
    return progress, task_id
