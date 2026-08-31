"""Rendering of probe results."""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .model import ProbeResult, Severity

_SEVERITY_STYLE = {
    Severity.INFO: "dim",
    Severity.WARN: "yellow",
    Severity.ERROR: "bold red",
}

_SEVERITY_MARK = {
    Severity.INFO: "i",
    Severity.WARN: "!",
    Severity.ERROR: "x",
}


def _fmt_shape(shape) -> str:
    """Format a shape as ``k x j x i``, or a dash if unknown.

    Args:
        shape: The shape, or None.

    Returns:
        The formatted string.
    """
    if not shape:
        return "-"
    return " x ".join(str(int(v)) for v in shape)


def _fmt_spacing(characterization) -> str:
    """Format voxel spacing in mm, or say it is undeclared.

    Args:
        characterization: The volume's characterization.

    Returns:
        The formatted string.
    """
    ps = characterization.pixel_spacing
    ss = characterization.slice_spacing
    if ps is None and ss is None:
        return "required"
    z = f"{ss:.5g}" if ss is not None else "?"
    y = f"{ps[0]:.5g}" if ps else "?"
    x = f"{ps[1]:.5g}" if ps else "?"
    return f"{z}, {y}, {x} mm"


def _fmt_extent(characterization) -> str:
    """Format the volume's physical size in mm, where spacing is known.

    Args:
        characterization: The volume's characterization.

    Returns:
        The formatted string.
    """
    extent = characterization.extent_mm
    if extent is None:
        return "-"
    return " x ".join(f"{v:.4g}" for v in extent)


def _issue_lines(issues) -> Text:
    """Render issues as coloured lines, worst first.

    Args:
        issues: The issues to show.

    Returns:
        Rich text, one line per issue.
    """
    text = Text()
    for issue in issues:
        style = _SEVERITY_STYLE[issue.severity]
        text.append(f"  [{_SEVERITY_MARK[issue.severity]}] ", style=style)
        text.append(f"{issue.code}: ", style=style)
        text.append(f"{issue.message}\n")
    return text


def render(result: ProbeResult, console: Console | None = None) -> None:
    """Print everything probe found about an image set.

    Args:
        result: The probe result.
        console: Console to print to; a new one is made if omitted.
    """
    console = console or Console()

    header = Table.grid(padding=(0, 2))
    header.add_column(style="bold")
    header.add_column()
    header.add_row("target", str(result.target))
    header.add_row("class", result.input_class.value)
    header.add_row("files scanned", str(result.n_files_scanned))
    header.add_row("volumes found", str(result.n_candidates))
    if result.unreadable:
        header.add_row("unreadable", str(len(result.unreadable)))
    console.print(Panel(header, title="probe", title_align="left"))

    if result.issues:
        console.print(_issue_lines(result.issues), end="")
        console.print()

    if not result.candidates:
        return

    table = Table(show_lines=False, header_style="bold")
    table.add_column("#", justify="right", width=3)
    table.add_column("label", overflow="fold")
    table.add_column("shape (z,y,x)")
    table.add_column("dtype")
    table.add_column("mod")
    table.add_column("units")
    table.add_column("spacing (z,y,x)")
    table.add_column("extent mm")
    table.add_column("order")
    table.add_column("", width=3)

    for index, candidate in enumerate(result.candidates, start=1):
        c = candidate.characterization
        if candidate.errors:
            status = Text("x", style="bold red")
        elif candidate.warnings:
            status = Text("!", style="yellow")
        else:
            status = Text("ok", style="green")
        modality = c.modality or "-"
        if c.declared_modality and c.declared_modality != c.modality:
            modality = f"{modality} (was {c.declared_modality})"
        units = c.units or "required"
        table.add_row(
            str(index),
            candidate.label,
            _fmt_shape(c.shape),
            c.dtype or "-",
            modality,
            units,
            _fmt_spacing(c),
            _fmt_extent(c),
            c.order_source,
            status,
        )

    console.print(table)
    console.print()

    for index, candidate in enumerate(result.candidates, start=1):
        c = candidate.characterization
        console.print(Text(f"[{index}] {candidate.label}", style="bold"))
        console.print(
            Text(
                f"    modality={c.modality or '-'} "
                f"(declared {c.declared_modality or '-'})  "
                f"units={c.units or '-'}  "
                f"slope={c.rescale_slope}  intercept={c.rescale_intercept}  "
                f"spacing source={c.spacing_source}",
                style="dim",
            )
        )
        for key, value in candidate.detail.items():
            if value:
                console.print(Text(f"    {key}={value}", style="dim"))
        console.print(Text(f"    first file: {candidate.files[0]}", style="dim"))
        if len(candidate.files) > 1:
            console.print(Text(f"    last  file: {candidate.files[-1]}", style="dim"))
        if candidate.issues:
            console.print(_issue_lines(candidate.issues), end="")
        console.print()

    if result.unreadable:
        console.print(Text("unreadable files (excluded):", style="yellow"))
        for path in result.unreadable[:10]:
            console.print(Text(f"  {path}", style="dim"))
        if len(result.unreadable) > 10:
            console.print(Text(f"  … and {len(result.unreadable) - 10} more", style="dim"))
        console.print()


def plan_panel(plan) -> Panel:
    """Show every decision before anything is written.

    Args:
        plan: The resolved conversion plan.

    Returns:
        A rich Panel.
    """
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="bold")
    grid.add_column()

    grid.add_row("source", plan.source_label)
    grid.add_row("slices", f"{plan.n_slices} x {plan.rows} x {plan.columns}")
    grid.add_row("modality", plan.modality)
    grid.add_row("units", plan.units)
    grid.add_row(
        "spacing (z,y,x)",
        f"{plan.slice_spacing:.6g}, {plan.pixel_spacing[0]:.6g}, "
        f"{plan.pixel_spacing[1]:.6g} mm",
    )
    grid.add_row("geometry", plan.geometry_source)
    grid.add_row(
        "window",
        f"centre {plan.window_center_ds}  width {plan.window_width_ds} "
        f"{plan.units}",
    )
    grid.add_row("frame of reference", plan.frame_of_reference_uid or "-")
    grid.add_row(
        "pixels",
        f"{plan.output_dtype}, "
        + ("requantized" if plan.requantized else "passthrough (lossless)"),
    )
    grid.add_row(
        "rescale",
        f"slope {plan.rescale_slope_ds}  intercept {plan.rescale_intercept_ds}",
    )
    if plan.source_value_range:
        low, high = plan.source_value_range
        grid.add_row("source range", f"[{low:.6g}, {high:.6g}]")
    if plan.quantization_step:
        grid.add_row("quantization step", f"{plan.quantization_step:.6g} {plan.units}")
    if plan.n_nonfinite:
        grid.add_row("non-finite voxels", str(plan.n_nonfinite))

    body = Table.grid()
    body.add_column()
    body.add_row(grid)
    for note in plan.notes:
        body.add_row(Text(f"- {note}", style="dim"))

    if plan.overrides:
        body.add_row(Text(""))
        body.add_row(Text("overrides from the job file", style="bold"))
        for override in plan.overrides:
            if override["kind"] == "replace":
                line = Text()
                line.append("  [replace] ", style="bold yellow")
                line.append(
                    f"{override['field']}: source said {override['source']!r}, "
                    f"using {override['supplied']!r}",
                    style="yellow",
                )
            else:
                line = Text(
                    f"  [fill]    {override['field']}: {override['supplied']!r}",
                    style="dim",
                )
            body.add_row(line)

    border = "yellow" if plan.contradictions else "none"
    return Panel(body, title="conversion plan", title_align="left", border_style=border)


def verify_panel(outcome) -> Panel:
    """Show what reading the written series back proved.

    Args:
        outcome: The `VerifyResult`.

    Returns:
        A rich Panel.
    """
    table = Table.grid(padding=(0, 2))
    table.add_column(width=3)
    table.add_column(style="bold")
    table.add_column(style="dim")
    for name, ok, detail in outcome.checks:
        mark = Text("ok", style="green") if ok else Text("x", style="bold red")
        table.add_row(mark, name, detail)

    title = "verification passed" if outcome.passed else "VERIFICATION FAILED"
    style = "green" if outcome.passed else "bold red"
    return Panel(table, title=title, title_align="left", border_style=style)


def verdict(result: ProbeResult) -> Text:
    """Summarise whether convert could proceed.

    Args:
        result: The probe result.

    Returns:
        One line of rich text.
    """
    candidate = result.volume
    if candidate is None:
        if result.n_candidates == 0:
            return Text("NO VOLUME FOUND", style="bold red")
        return Text(
            f"{result.n_candidates} VOLUMES FOUND - narrow the input path to one",
            style="bold red",
        )
    if candidate.errors:
        return Text(
            f"NEEDS EDITING - {len(candidate.errors)} value(s) must be supplied "
            "in the job file",
            style="bold yellow",
        )
    if candidate.warnings:
        return Text(
            f"READY, with {len(candidate.warnings)} warning(s) to acknowledge",
            style="yellow",
        )
    return Text("READY", style="bold green")
