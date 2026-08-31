"""Command line interface for Image2Dicom.

    probe <input> <job.yaml>      inspect an image set, write an editable job file
    convert <job.yaml>            run the conversion that job file describes

The input must resolve to exactly one volume. If it doesn't, narrow the path.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console

from . import config as config_mod
from . import discovery, report, sources, verify as verify_mod, writer
from .model import to_jsonable

app = typer.Typer(
    add_completion=False,
    help="Convert one image set into a canonical DICOM series directory.",
)
console = Console()


@app.command()
def probe(
    target: Path = typer.Argument(..., help="Path to one image set (file or directory)."),
    job_file: Path = typer.Argument(..., help="Job file to write."),
):
    """Inspect an image set and write a job file for it. Writes no image data.

    Args:
        target: One image set - a file or a directory.
        job_file: Job file to write; its directory is created.

    Raises:
        typer.Exit: Code 1 if the target does not exist or does not resolve
            to exactly one volume.
    """
    try:
        with console.status(f"scanning {target} …", spinner="dots") as status:
            scanned = {"n": 0}

            def tick():
                scanned["n"] += 1
                if scanned["n"] % 25 == 0:
                    status.update(f"reading headers … {scanned['n']} files")

            result = discovery.probe(target, on_file=tick)
    except FileNotFoundError as exc:
        console.print(f"[bold red]{exc}[/bold red]")
        raise typer.Exit(code=1)

    report.render(result, console=console)
    console.print(report.verdict(result))

    if result.volume is None:
        raise typer.Exit(code=1)

    written = config_mod.write_template(result, result.volume, job_file)
    console.print(f"[green]wrote job file {written}[/green]")
    console.print(f"[dim]Edit it, then: image2dicom convert {written}[/dim]")


@app.command()
def convert(
    job_file: Path = typer.Argument(..., help="Path to a YAML job file."),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Resolve and print the plan; write nothing."
    ),
):
    """Run the conversion described by a job file.

    Args:
        job_file: A YAML job file, as `probe` writes.
        dry_run: Resolve and print the plan, then stop without writing.

    Raises:
        typer.Exit: Code 2 for a bad job file or a non-empty output
            directory; code 1 for an input that will not convert, an
            unacknowledged warning or override, or a failed verification.
    """
    try:
        job = config_mod.load(job_file)
    except config_mod.ConfigError as exc:
        console.print(f"[bold red]{exc}[/bold red]")
        raise typer.Exit(code=2)

    try:
        with console.status(f"inspecting {job.input_path} …", spinner="dots"):
            result = discovery.probe(job.input_path)
    except FileNotFoundError as exc:
        console.print(f"[bold red]{exc}[/bold red]")
        raise typer.Exit(code=2)

    candidate = result.volume
    if candidate is None:
        report.render(result, console=console)
        console.print(report.verdict(result))
        raise typer.Exit(code=1)

    if candidate.warnings and not job.accept_warnings:
        for issue in candidate.warnings:
            console.print(f"[yellow]  [!] {issue.code}: {issue.message}[/yellow]")
        console.print(
            "[dim]Set options.accept_warnings: true in the job file to proceed.[/dim]"
        )
        raise typer.Exit(code=1)

    try:
        source = sources.open_source(candidate, raw=job.raw)
    except ValueError as exc:
        console.print(f"[bold red]{exc}[/bold red]")
        raise typer.Exit(code=1)

    try:
        try:
            with console.status("scanning value range …", spinner="dots") as status:

                def scanned(index):
                    if index % 10 == 0:
                        status.update(f"scanning value range … slice {index + 1}")

                plan = writer.build_plan(
                    candidate,
                    source,
                    modality=job.modality,
                    units=job.units,
                    spacing=job.spacing_mm,
                    rescale_slope=job.rescale_slope,
                    rescale_intercept=job.rescale_intercept,
                    patient_name=job.patient_name,
                    patient_id=job.patient_id,
                    center=job.center,
                    on_scan=scanned,
                )
        except ValueError as exc:
            console.print(f"[bold red]{exc}[/bold red]")
            raise typer.Exit(code=1)

        console.print(report.plan_panel(plan))

        if plan.contradictions and not job.accept_overrides:
            fields = ", ".join(o["field"] for o in plan.contradictions)
            console.print(
                f"[bold yellow]The job file replaces {fields}, which the source "
                "actually declares. Set options.accept_overrides: true to confirm "
                "that is intended.[/bold yellow]"
            )
            raise typer.Exit(code=1)

        if dry_run:
            console.print("[dim]--dry-run: nothing written.[/dim]")
            raise typer.Exit(code=0)

        if job.out_dir.exists() and any(job.out_dir.iterdir()):
            console.print(f"[bold red]{job.out_dir} exists and is not empty.[/bold red]")
            raise typer.Exit(code=2)

        with console.status("writing …", spinner="dots") as status:

            def tick(index):
                if index % 10 == 0:
                    status.update(f"writing slice {index + 1}/{plan.n_slices}")

            writer.write_series(plan, source, job.out_dir, on_slice=tick)

        with console.status("verifying …", spinner="dots"):
            outcome = verify_mod.verify_series(plan, source, job.out_dir)

        record = job.out_dir / "conversion.json"
        record.write_text(
            json.dumps(
                {
                    "job_file": str(job.source_file),
                    "source": str(job.input_path),
                    "plan": to_jsonable(plan),
                    "verification": to_jsonable(outcome),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        console.print(report.verify_panel(outcome))
        console.print(f"[dim]wrote {plan.n_slices} files and {record}[/dim]")
        raise typer.Exit(code=0 if outcome.passed else 1)
    finally:
        source.close()


def main():
    """Entry point for ``python -m image2dicom``."""
    app()


if __name__ == "__main__":
    main()
