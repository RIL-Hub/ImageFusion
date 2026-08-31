"""Command line interface for Aligner.

    aligner doctor                              what is installed, and is napari
                                                where we expect it
    aligner inspect <series-dir> [...]          report geometry, open nothing
    aligner view <series-dir> [...]             open them together in napari
    aligner solve <session.yaml>                re-run the fits, report residuals
    aligner apply <session.yaml> <out-dir>      write registered series + transforms

`solve` and `apply` import no Qt and no napari, so a session file replays the whole
registration with the viewer uninstalled.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import typer
from rich.console import Console
from rich.table import Table

from . import loading
from .geometry import axis_tilt_degrees

app = typer.Typer(add_completion=False, help="View and register DICOM series.")
console = Console()


@app.command()
def doctor():
    """Report what is installed and whether napari is where we expect it.

    Run this first when something misbehaves. Aligner reaches into a few napari
    internals that have moved between versions, and each move looks like an
    unrelated feature being broken.

    Raises:
        typer.Exit: Code 1 if anything essential is missing.
    """
    from .doctor import report

    lines, healthy = report()
    # markup=False because the status markers are bracketed, and rich reads square
    # brackets as style tags - it silently ate them, so a FAIL looked like a pass.
    console.print("\n".join(lines), markup=False, highlight=False)
    raise typer.Exit(code=0 if healthy else 1)


@app.command()
def view(
    series: List[Path] = typer.Argument(..., help="Image2Dicom output directories."),
    budget_mb: int = typer.Option(
        1024,
        "--budget-mb",
        help="RAM per volume before it is decimated to fit.",
    ),
):
    """Open the given series together in one napari viewer.

    Load order sets the registration chain: moving first, fixed last.

    Args:
        series: Image2Dicom output directories, in chain order.
        budget_mb: Memory per volume before it is decimated to fit.

    Raises:
        typer.Exit: Code 1 if a directory is missing, a series will not load, or Qt
            is not importable.
    """
    from .ui.app import launch

    missing = [path for path in series if not path.is_dir()]
    if missing:
        for path in missing:
            console.print(f"[bold red]No such series directory: {path}[/bold red]")
        console.print("[dim]These must be Image2Dicom output directories.[/dim]")
        raise typer.Exit(code=1)

    try:
        launch(series, budget_bytes=budget_mb * 1024 * 1024)
    except loading.LoadError as exc:
        console.print(f"[bold red]{exc}[/bold red]")
        raise typer.Exit(code=1)
    except ImportError as exc:
        # Almost always Qt: the one failure a new user is most likely to hit.
        console.print(f"[bold red]{exc}[/bold red]")
        console.print("[dim]Run 'aligner doctor' to see what is missing.[/dim]")
        raise typer.Exit(code=1)


@app.command()
def solve(
    session: Path = typer.Argument(..., help="Session file written by the viewer."),
):
    """Re-run every registration in a session and report the residuals.

    Imports no Qt and no napari.

    Args:
        session: A session file written by the viewer.

    Raises:
        typer.Exit: Code 1 if the session cannot be read or solved.
    """
    from .landmarks import LandmarkError, load_session
    from .solve import SolveError, describe, solve_session

    try:
        solution = solve_session(load_session(session))
    except (LandmarkError, SolveError) as exc:
        console.print(f"[bold red]{exc}[/bold red]")
        raise typer.Exit(code=1)
    console.print("\n".join(describe(solution)))


@app.command()
def apply(
    session: Path = typer.Argument(..., help="Session file written by the viewer."),
    output: Path = typer.Argument(..., help="Directory to write registered series into."),
    overwrite: bool = typer.Option(
        False, "--overwrite", help="Replace series already written there."
    ),
):
    """Write every moving volume into the fixed volume's frame, plus the transforms.

    Geometry is rewritten and voxels are copied through untouched; nothing is
    resampled and no array changes size. Imports no Qt and no napari.

    Args:
        session: A session file written by the viewer.
        output: Directory to write the registered series and transforms.yaml into.
        overwrite: Replace series already written there.

    Raises:
        typer.Exit: Code 1 if the session cannot be read, solved, or written out.
    """
    from .apply import ApplyError, apply_session, save_transforms
    from .landmarks import LandmarkError, load_session
    from .solve import SolveError, describe, solve_session

    try:
        loaded = load_session(session)
        solution = solve_session(loaded)
    except (LandmarkError, SolveError) as exc:
        console.print(f"[bold red]{exc}[/bold red]")
        raise typer.Exit(code=1)

    console.print("\n".join(describe(solution)))

    try:
        written = apply_session(loaded, solution, output, overwrite=overwrite)
        transforms = save_transforms(output / "transforms.yaml", loaded, solution)
    except (ApplyError, OSError) as exc:
        console.print(f"[bold red]{exc}[/bold red]")
        raise typer.Exit(code=1)

    console.print()
    for record in written:
        console.print(
            f"wrote {record.slices} slices to {record.destination}  "
            f"(rotated {record.rotation_degrees:.2f}deg, shifted {record.shift_mm:.2f} mm)"
        )
    console.print(f"wrote {transforms}")
    console.print(
        f"[dim]{solution.fixed} was not copied; it is the fixed frame and never moves.[/dim]"
    )


@app.command()
def inspect(
    series: List[Path] = typer.Argument(..., help="Image2Dicom output directories."),
):
    """Report what each series holds, without opening a viewer.

    Args:
        series: Image2Dicom output directories.

    Raises:
        typer.Exit: Code 1 if a series will not load.
    """
    table = Table(header_style="bold")
    for column in (
        "name",
        "shape (k,j,i)",
        "spacing (k,j,i) mm",
        "modality",
        "units",
        "axis tilt",
        "full res",
    ):
        table.add_column(column)

    for path in series:
        try:
            volume = loading.load_series(path, materialise=False)
        except loading.LoadError as exc:
            console.print(f"[bold red]{exc}[/bold red]")
            raise typer.Exit(code=1)
        g = volume.geometry
        tilt = axis_tilt_degrees(volume.affine)
        table.add_row(
            volume.name,
            " x ".join(str(v) for v in volume.shape),
            f"{g.slice_spacing:.5g}, {g.pixel_spacing[0]:.5g}, {g.pixel_spacing[1]:.5g}",
            volume.modality,
            volume.units,
            f"{tilt:.4f} deg",
            " x ".join(str(v) for v in (volume.full_shape or volume.shape)),
        )

    console.print(table)


def main():
    """Entry point for ``python -m aligner``."""
    app()


if __name__ == "__main__":
    main()
