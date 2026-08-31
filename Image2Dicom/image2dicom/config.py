"""Conversion jobs described by a YAML file.

A job file is the editable record of what you asked for. Paths inside it resolve
relative to the file itself, so a job travels alongside the data it describes.

Generated job files carry their findings as comments: the source's own value sits
next to each commented-out override, and anything the source cannot supply is left
uncommented and marked REQUIRED.

Anything set under ``overrides`` replaces or fills what the source declares. Overrides
are never silent: each is recorded in conversion.json with both the source value and
the supplied value, and one that *contradicts* the source is refused unless
``options.accept_overrides`` is true.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

from .model import InputClass


@dataclass
class RawDeclaration:
    """How to read a headerless file. Nothing here is inferred."""

    shape: tuple  # (z, y, x)
    dtype: str
    byte_order: str = "little"
    header_bytes: int = 0


@dataclass
class JobConfig:
    input_path: Path
    out_dir: Path
    raw: Optional[RawDeclaration] = None

    patient_name: Optional[str] = None
    patient_id: Optional[str] = None
    center: bool = True  # place the volume centre at (0, 0, 0)

    spacing_mm: Optional[tuple] = None  # (z, y, x)
    modality: Optional[str] = None
    units: Optional[str] = None
    rescale_slope: Optional[float] = None
    rescale_intercept: Optional[float] = None

    accept_warnings: bool = False  # acknowledges what probe said about the source
    accept_overrides: bool = False  # authorises replacing what the source declares
    source_file: Optional[Path] = None


class ConfigError(ValueError):
    pass


def _check_keys(section_name: str, section: dict, known: set) -> None:
    """Reject unknown keys rather than silently ignoring them.

    Args:
        section_name: Section being checked, for the message.
        section: The mapping as read.
        known: Keys this section accepts.

    Raises:
        ConfigError: If the section holds a key that is not in `known`.
    """
    unknown = set(section) - known
    if unknown:
        raise ConfigError(
            f"Unknown key(s) in '{section_name}': {', '.join(sorted(unknown))}. "
            f"Known keys: {', '.join(sorted(known))}"
        )


def _resolve(base: Path, value: str) -> Path:
    """Resolve a path from the job file, relative to the job file itself.

    Args:
        base: The job file's directory.
        value: The path as written in the file.

    Returns:
        An absolute path. Relative to *the file*, not the working
        directory - the distinction that produced a real bug here once.
    """
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    return (base / candidate).resolve()


def load(path: Path) -> JobConfig:
    """Read and validate a job file.

    Args:
        path: The YAML job file.

    Returns:
        The `JobConfig` it describes, with paths already resolved.

    Raises:
        ConfigError: If the file is missing, is not valid YAML, is not a
            mapping, holds an unknown key, or omits something required.
    """
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Job file not found: {path}")

    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path} is not valid YAML: {exc}") from exc

    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a YAML mapping.")

    base = path.parent
    _check_keys(
        "(top level)", data, {"input", "output", "overrides", "options", "raw"}
    )

    section = data.get("input")
    if not isinstance(section, dict) or not section.get("path"):
        raise ConfigError("An 'input:' mapping with a 'path' key is required.")
    _check_keys("input", section, {"path"})
    input_path = _resolve(base, str(section["path"]))

    out = data.get("output")
    if not isinstance(out, dict) or not out.get("directory"):
        raise ConfigError("An 'output:' mapping with a 'directory' key is required.")
    _check_keys("output", out, {"directory", "patient_name", "patient_id", "center"})
    out_dir = _resolve(base, str(out["directory"]))

    # A section whose every key is commented out parses as None, not {}.
    overrides = data.get("overrides") or {}
    if not isinstance(overrides, dict):
        raise ConfigError("'overrides:' must be a mapping.")
    _check_keys(
        "overrides",
        overrides,
        {"spacing_mm", "modality", "units", "rescale_slope", "rescale_intercept"},
    )

    spacing = overrides.get("spacing_mm")
    if spacing is not None:
        if not isinstance(spacing, (list, tuple)) or len(spacing) != 3:
            raise ConfigError("overrides.spacing_mm must be three numbers: [z, y, x]")
        try:
            spacing = tuple(float(v) for v in spacing)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"overrides.spacing_mm must be numbers: {exc}") from exc
        if any(v <= 0 for v in spacing):
            raise ConfigError("overrides.spacing_mm values must all be positive.")

    for name in ("modality", "units"):
        value = overrides.get(name)
        if value is not None and not isinstance(value, str):
            # YAML coerces bare words: `units: NO` becomes False.
            raise ConfigError(
                f"overrides.{name} must be a quoted string, got {value!r}."
            )

    slope = overrides.get("rescale_slope")
    intercept = overrides.get("rescale_intercept")
    if (slope is None) != (intercept is None):
        raise ConfigError(
            "rescale_slope and rescale_intercept must be given together - supplying "
            "one without the other would mix a new mapping with the source's."
        )
    if slope is not None and float(slope) == 0.0:
        raise ConfigError("rescale_slope cannot be zero.")

    options = data.get("options") or {}
    if not isinstance(options, dict):
        raise ConfigError("'options:' must be a mapping.")
    _check_keys("options", options, {"accept_warnings", "accept_overrides"})

    raw = None
    raw_section = data.get("raw")
    if raw_section:
        if not isinstance(raw_section, dict):
            raise ConfigError("'raw:' must be a mapping.")
        _check_keys(
            "raw", raw_section, {"shape", "dtype", "byte_order", "header_bytes"}
        )
        shape = raw_section.get("shape")
        if shape is None:
            raise ConfigError("raw.shape is required for headerless input: [z, y, x].")
        if not isinstance(shape, (list, tuple)) or len(shape) != 3:
            raise ConfigError("raw.shape must be three integers: [z, y, x]")
        if any(int(v) <= 0 for v in shape):
            raise ConfigError("raw.shape values must all be positive.")
        dtype = raw_section.get("dtype")
        if not isinstance(dtype, str) or not dtype:
            raise ConfigError('raw.dtype is required, e.g. "float32".')
        byte_order = str(raw_section.get("byte_order") or "little").lower()
        if byte_order not in ("little", "big"):
            raise ConfigError('raw.byte_order must be "little" or "big".')
        header_bytes = int(raw_section.get("header_bytes") or 0)
        if header_bytes < 0:
            raise ConfigError("raw.header_bytes cannot be negative.")
        raw = RawDeclaration(
            shape=tuple(int(v) for v in shape),
            dtype=dtype,
            byte_order=byte_order,
            header_bytes=header_bytes,
        )

    return JobConfig(
        input_path=input_path,
        out_dir=out_dir,
        raw=raw,
        patient_name=out.get("patient_name") or None,
        patient_id=out.get("patient_id") or None,
        center=bool(out.get("center", True)),
        spacing_mm=spacing,
        modality=overrides.get("modality") or None,
        units=overrides.get("units") or None,
        rescale_slope=float(slope) if slope is not None else None,
        rescale_intercept=float(intercept) if intercept is not None else None,
        accept_warnings=bool(options.get("accept_warnings", False)),
        accept_overrides=bool(options.get("accept_overrides", False)),
        source_file=path,
    )


def _rel(target: Path, base: Path) -> str:
    """Express a path relative to the job file, with `..` where needed.

    Must round-trip through `_resolve`, which joins onto the job file's
    directory - so this cannot fall back to the path as typed on the command
    line, which is relative to the working directory instead.

    Args:
        target: The path to express.
        base: The job file's directory.

    Returns:
        A forward-slashed relative path, or an absolute one when no relative
        path exists - on Windows, across drives.
    """
    try:
        return Path(os.path.relpath(Path(target).resolve(), base.resolve())).as_posix()
    except ValueError:
        # Different drives on Windows; only an absolute path can express it.
        return Path(target).resolve().as_posix()


def write_template(
    result, candidate, job_path: Path, out_dir: Optional[Path] = None
) -> Path:
    """Emit a job file pre-filled with what probe discovered.

    Hand-built rather than dumped, because the comments are most of the value
    and a YAML serializer discards them.

    Args:
        result: The probe result, for the warnings to acknowledge.
        candidate: The volume the job will convert.
        job_path: File to write; its directory is created.
        out_dir: Suggested output directory; derived from the input if
            omitted.

    Returns:
        The path written.
    """
    job_path = Path(job_path)
    base = job_path.parent if str(job_path.parent) else Path(".")
    base.mkdir(parents=True, exist_ok=True)

    c = candidate.characterization
    # Anchor the suggestion to the working directory, then express it relative to
    # the job file - otherwise it lands inside whatever folder the job lives in.
    suggested_out = Path(out_dir) if out_dir else (Path.cwd() / "out" / job_path.stem)

    lines = [
        "# Image2Dicom conversion job",
        f"# Generated from a probe of: {result.target}",
        f"# Candidate: {candidate.label}",
        "#",
        "# A commented-out override shows the value the source already declares.",
        "# An uncommented one marked REQUIRED is a placeholder you must replace.",
        "",
        "input:",
        f'  path: "{_rel(result.target, base)}"',
    ]
    lines += [
        "",
        "output:",
        f'  directory: "{_rel(suggested_out, base)}"',
        '  patient_name: "Specimen"',
        '  patient_id: "UNKNOWN"',
        "  # Place the volume centre at (0, 0, 0). Set false to keep the scanner's",
        "  # own position, which is what preserves a shared FrameOfReferenceUID.",
        "  center: true",
    ]

    if candidate.input_class is InputClass.UNIDENTIFIED:
        size = candidate.detail.get("size_bytes", 0)
        lines += [
            "",
            "# This file has no header, so everything below must be declared.",
            f"# It is {size:,} bytes; the declared geometry is checked against that",
            "# and refused if it disagrees.",
        ]
        suggestions = candidate.detail.get("shape_suggestions") or []
        if suggestions:
            lines.append("# Shapes that would fit exactly - arithmetic, NOT evidence:")
            lines += [f"#   {s}" for s in suggestions]
        lines += [
            "raw:",
            "  shape: [0, 0, 0]        # z, y, x - REQUIRED",
            '  dtype: "float32"        # uint8 int8 uint16 int16 uint32 int32 float32 float64',
            '  byte_order: "little"',
            "  header_bytes: 0",
        ]

    lines += [
        "",
        "overrides:",
        "  # Uncomment to fill a gap or replace what the source declares.",
        "  # Every override is recorded; one that contradicts the source is refused",
        "  # unless options.accept_overrides is true.",
    ]

    if c.has_spacing:
        lines.append(
            f"  # spacing_mm: [{c.slice_spacing!r}, {c.pixel_spacing[0]!r}, "
            f"{c.pixel_spacing[1]!r}]   # z, y, x - source value, from "
            f"{c.spacing_source}"
        )
    else:
        lines.append(
            "  spacing_mm: [1.0, 1.0, 1.0]   # z, y, x - REQUIRED, source has none"
        )

    if c.modality:
        lines.append(f'  # modality: "{c.modality}"   # source value')
    else:
        lines.append(
            '  modality: "PT"   # REQUIRED, source has none. CT, PT, MR, NM'
        )

    if c.units and c.units != "arbitrary":
        lines.append(f'  # units: "{c.units}"   # source value')
    elif c.units == "arbitrary":
        lines.append(
            '  # units: "HU"   # source says arbitrary reconstruction units; set this'
        )
        lines.append(
            "  #                only alongside a calibrated slope/intercept below"
        )
    else:
        lines.append('  units: "arbitrary"   # REQUIRED, source declares none')

    lines += [
        "",
        "  # Calibration. On an integer source the stored voxels are still copied",
        "  # verbatim - supplying a mapping changes only how they are interpreted,",
        "  # so this stays lossless. Both values must be given together.",
        f"  # rescale_slope: {c.rescale_slope if c.rescale_slope is not None else 1.0}",
        f"  # rescale_intercept: "
        f"{c.rescale_intercept if c.rescale_intercept is not None else 0.0}",
        "",
        "options:",
        "  # Acknowledges what probe reported about the source itself.",
        f"  accept_warnings: {'true' if candidate.warnings else 'false'}",
    ]
    if candidate.warnings:
        lines.append("  # Set because probe reported:")
        for issue in candidate.warnings:
            lines.append(f"  #   {issue.code}: {issue.message}")

    lines += [
        "",
        "  # Authorises any override that REPLACES a value the source declares.",
        "  # Filling a gap the source left blank does not need this.",
        "  accept_overrides: false",
    ]

    job_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return job_path
