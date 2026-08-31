"""Geometry, ordering and integrity helpers.

Pure functions over already-read DICOM headers or TIFF metadata. No file walking,
no candidate grouping, no pixel reads.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Optional

import numpy as np
import tifffile

from .model import error, warn

_NUM_RE = re.compile(r"\d+")

# Fractional deviation from the median slice gap. DICOM stores positions as decimal
# strings, so a little jitter is normal; a missing or duplicated slice shows up as a
# deviation of 100% or more, so the error threshold can be generous.
SPACING_JITTER = 1e-3  # above this: worth mentioning
SPACING_TOLERANCE = 1e-2  # above this: not one uniformly sampled volume


def numeric_key(path: Path):
    """Build a sort key that compares embedded integers numerically.

    Args:
        path: The file to key.

    Returns:
        A key giving ``I1 < I5 < I10``, where a plain string sort gives
        ``I1, I10, I5``.
    """
    nums = tuple(int(m) for m in _NUM_RE.findall(path.stem))
    return (nums, path.stem.lower())


def label_fields(path: Path) -> str:
    """Parse the fields that distinguish one file from its siblings.

    Used only for labelling choices, never for deciding what a volume is.

    Args:
        path: The file to parse.

    Returns:
        Its ``letter+digits`` tokens (I25, F30, T2) joined for display - which
        is how the PET reconstructions distinguish themselves.
    """
    tokens = re.findall(r"\b([A-Za-z])(\d+)\b", path.stem)
    if tokens:
        return " ".join(f"{a.upper()}{b}" for a, b in tokens)
    return path.stem


# --- DICOM geometry -------------------------------------------------------


def as_floats(value) -> Optional[tuple]:
    """Read a DICOM multi-valued element as numbers.

    Args:
        value: The element, or None.

    Returns:
        Its values as a tuple of floats, or None if unreadable.
    """
    if value is None:
        return None
    try:
        return tuple(float(x) for x in value)
    except (TypeError, ValueError):
        return None



def slice_normal(items) -> Optional[np.ndarray]:
    """Find the direction slices are stacked along.

    Args:
        items: (FileRecord, Dataset) pairs.

    Returns:
        Unit normal of the slice plane, from the first usable
        ImageOrientationPatient, or None if none declares one.
    """
    for _, ds in items:
        iop = as_floats(ds.get("ImageOrientationPatient"))
        if iop is not None and len(iop) == 6:
            row = np.array(iop[:3], dtype=float)
            col = np.array(iop[3:], dtype=float)
            normal = np.cross(row, col)
            norm = np.linalg.norm(normal)
            if norm > 0:
                return normal / norm
    return None


def order_dicom(items):
    """Put a series' slices into their physical order.

    Args:
        items: (FileRecord, Dataset) pairs, in any order.

    Returns:
        (ordered_items, order_source, positions, issues). `order_source` names
        what the order was derived from. `positions` are in mm along the slice
        normal where available, or None - they are what makes spacing
        verifiable rather than merely asserted.
    """
    issues: list = []

    normal = slice_normal(items)
    positions = None
    if normal is not None:
        projected = []
        for _, ds in items:
            ipp = as_floats(ds.get("ImagePositionPatient"))
            if ipp is None or len(ipp) != 3:
                projected = None
                break
            projected.append(float(np.dot(np.array(ipp, dtype=float), normal)))
        if projected is not None:
            order = sorted(range(len(items)), key=lambda i: projected[i])
            ordered = [items[i] for i in order]
            positions = [projected[i] for i in order]
            return ordered, "ImagePositionPatient", positions, issues

    if all(ds.get("SliceLocation") is not None for _, ds in items):
        issues.append(
            warn(
                "order-fallback",
                "Ordered by SliceLocation; ImagePositionPatient/Orientation "
                "unavailable, so slice spacing cannot be independently verified.",
            )
        )
        ordered = sorted(items, key=lambda it: float(it[1].SliceLocation))
        positions = [float(it[1].SliceLocation) for it in ordered]
        return ordered, "SliceLocation", positions, issues

    if all(ds.get("InstanceNumber") is not None for _, ds in items):
        issues.append(
            warn(
                "order-fallback",
                "Ordered by InstanceNumber; no positional geometry present, so slice "
                "order is plausible but unverified.",
            )
        )
        ordered = sorted(items, key=lambda it: int(it[1].InstanceNumber))
        return ordered, "InstanceNumber", None, issues

    issues.append(
        warn(
            "order-fallback",
            "Ordered by numeric filename key; no DICOM ordering tags present at all.",
        )
    )
    ordered = sorted(items, key=lambda it: numeric_key(it[0].path))
    return ordered, "filename", None, issues


def check_positions(positions) -> tuple:
    """Check slice positions are monotonic and uniformly spaced.

    Args:
        positions: Slice positions in mm along the normal, or None.

    Returns:
        (slice_spacing, issues): the measured spacing in mm, or None if it
        cannot be trusted, and what was wrong.
    """
    issues: list = []
    if positions is None or len(positions) < 2:
        return None, issues

    diffs = np.diff(np.asarray(positions, dtype=float))

    n_dup = int(np.sum(np.isclose(diffs, 0.0)))
    if n_dup:
        issues.append(
            error(
                "duplicate-positions",
                f"{n_dup} slice(s) share a position with their neighbour — the set "
                "likely contains more than one volume under a single series UID.",
            )
        )
        nonzero = diffs[~np.isclose(diffs, 0.0)]
        if nonzero.size == 0:
            return None, issues
        diffs = nonzero

    median = float(np.median(np.abs(diffs)))
    if median <= 0:
        return None, issues

    deviation = float(np.max(np.abs(np.abs(diffs) - median)) / median)
    detail = (
        f"(median {median:.6g} mm, min {np.min(np.abs(diffs)):.6g}, "
        f"max {np.max(np.abs(diffs)):.6g})"
    )
    if deviation > SPACING_TOLERANCE:
        issues.append(
            error(
                "irregular-spacing",
                f"Slice spacing varies by {deviation * 100:.2f}% {detail} — "
                "gaps, truncation or interleaving.",
            )
        )
    elif deviation > SPACING_JITTER:
        issues.append(
            warn(
                "spacing-jitter",
                f"Slice spacing varies by {deviation * 100:.3f}% {detail} — within "
                "tolerance, but the median will be used as the spacing.",
            )
        )

    return median, issues


def derive_spacing(items, positions):
    """Determine voxel size, preferring geometry measured over geometry declared.

    Args:
        items: (FileRecord, Dataset) pairs, already ordered.
        positions: Slice positions in mm, or None.

    Returns:
        (pixel_spacing, slice_spacing, source, issues): in-plane spacing as
        (row_mm, col_mm), the spacing between slices, what each came from, and
        anything the user should know.
    """
    issues: list = []
    _, first = items[0]

    pixel_spacing = None
    ps = as_floats(first.get("PixelSpacing"))
    if ps is not None and len(ps) == 2:
        pixel_spacing = (ps[0], ps[1])

    measured, pos_issues = check_positions(positions)
    issues.extend(pos_issues)

    if measured is not None:
        return pixel_spacing, measured, "ImagePositionPatient", issues

    for tag, name in (("SpacingBetweenSlices", "SpacingBetweenSlices"),
                      ("SliceThickness", "SliceThickness")):
        value = first.get(tag)
        if value is not None:
            try:
                spacing = float(value)
            except (TypeError, ValueError):
                continue
            if spacing > 0:
                issues.append(
                    warn(
                        "spacing-asserted",
                        f"Slice spacing taken from {name}; not verified against slice "
                        "positions.",
                    )
                )
                return pixel_spacing, spacing, name, issues

    return pixel_spacing, None, "none", issues


def check_consistency(items) -> list:
    """Check every slice agrees on the things that make it one volume.

    Args:
        items: (FileRecord, Dataset) pairs.

    Returns:
        Issues for any tag that varies between slices - shape, pixel depth or
        rescale mapping.
    """
    issues: list = []
    checks = (
        ("Rows", "in-plane height"),
        ("Columns", "in-plane width"),
        ("BitsAllocated", "pixel depth"),
        ("PixelRepresentation", "pixel signedness"),
        ("RescaleSlope", "rescale slope"),
        ("RescaleIntercept", "rescale intercept"),
    )
    for tag, description in checks:
        values = {str(ds.get(tag)) for _, ds in items}
        if len(values) > 1:
            shown = ", ".join(sorted(values)[:4])
            issues.append(
                error(
                    "inconsistent-slices",
                    f"Slices disagree on {description} ({tag}): {shown}"
                    + (" …" if len(values) > 4 else ""),
                )
            )
    return issues


def dicom_dtype(ds) -> Optional[str]:
    """Work out how a series stores its pixels.

    Args:
        ds: A slice's dataset.

    Returns:
        The dtype name, such as "uint16", or None if the tags do not say.
    """
    bits = ds.get("BitsAllocated")
    signed = ds.get("PixelRepresentation")
    if bits is None:
        return None
    try:
        bits = int(bits)
    except (TypeError, ValueError):
        return None
    kind = "int" if str(signed) == "1" else "uint"
    return f"{kind}{bits}"


def dicom_shape(items) -> Optional[tuple]:
    """Work out a series' volume shape.

    Args:
        items: (FileRecord, Dataset) pairs.

    Returns:
        (k, j, i), counting frames for a multi-frame file, or None if the tags
        do not say.
    """
    _, first = items[0]
    rows = first.get("Rows")
    cols = first.get("Columns")
    if rows is None or cols is None:
        return None
    frames = first.get("NumberOfFrames")
    if len(items) == 1 and frames is not None:
        try:
            nz = int(frames)
        except (TypeError, ValueError):
            nz = 1
    else:
        nz = len(items)
    return (nz, int(rows), int(cols))


# What a SOP class says the image actually is. More trustworthy than the Modality
# tag, which some scanners set from the machine rather than the reconstruction --
# the Siemens Zeego reports Modality=XA for CT Image Storage data in HU.
SOP_CLASS_MODALITY = {
    "1.2.840.10008.5.1.4.1.1.2": "CT",
    "1.2.840.10008.5.1.4.1.1.2.1": "CT",  # Enhanced CT
    "1.2.840.10008.5.1.4.1.1.128": "PT",  # PET
    "1.2.840.10008.5.1.4.1.1.130": "PT",  # Enhanced PET
    "1.2.840.10008.5.1.4.1.1.20": "NM",
    "1.2.840.10008.5.1.4.1.1.4": "MR",
    "1.2.840.10008.5.1.4.1.1.4.1": "MR",
    "1.2.840.10008.5.1.4.1.1.12.1": "XA",
    "1.2.840.10008.5.1.4.1.1.12.1.1": "XA",
    "1.2.840.10008.5.1.4.1.1.7": "SC",  # Secondary Capture - says nothing
}

# RescaleType values that mean "these numbers are in a defined physical unit".
KNOWN_UNITS = {"HU", "MGML", "OD", "BQML", "CNTS"}


def resolve_modality(ds) -> tuple:
    """Decide what kind of image this is.

    The SOP class wins over the Modality tag: the Zeego writes CT images while
    declaring Modality XA, and the SOP class is the one that describes the
    pixels.

    Args:
        ds: A slice's dataset.

    Returns:
        (modality, issues): the resolved modality, and a note where the two
        sources disagree.
    """
    issues: list = []
    declared = ds.get("Modality")
    declared = str(declared).strip() if declared else None

    sop = ds.get("SOPClassUID")
    derived = SOP_CLASS_MODALITY.get(str(sop).strip()) if sop else None

    if derived is None or derived == "SC":
        if declared is None:
            issues.append(
                warn("missing-modality", "No Modality and no usable SOP class; must be supplied.")
            )
        return declared, issues

    if declared and declared != derived:
        issues.append(
            warn(
                "modality-mismatch",
                f"Declared Modality is {declared!r} but the SOP class is "
                f"{derived} Image Storage. Using {derived}; some scanners report the "
                "modality of the machine rather than the reconstruction.",
            )
        )
    return derived, issues


def resolve_units(ds) -> tuple:
    """Determine what the rescaled values physically mean.

    Args:
        ds: A slice's dataset.

    Returns:
        (units, issues): the units, such as "HU", and a note where the source
        declares none or declares them arbitrary - which blocks any later
        physical conversion.
    """
    issues: list = []
    rescale_type = ds.get("RescaleType")
    rescale_type = str(rescale_type).strip().upper() if rescale_type else None
    pet_units = ds.get("Units")
    pet_units = str(pet_units).strip().upper() if pet_units else None

    declared = rescale_type or pet_units

    if declared in KNOWN_UNITS:
        return declared, issues

    if declared == "US":
        issues.append(
            warn(
                "arbitrary-units",
                "RescaleType is 'US' (unspecified): values are in arbitrary "
                "reconstruction units, not a calibrated physical scale. Quantitative "
                "use (e.g. deriving attenuation from HU) requires calibration first.",
            )
        )
        return "arbitrary", issues

    if declared is None:
        issues.append(
            warn(
                "unknown-units",
                "No RescaleType or Units tag; the physical meaning of the values is "
                "undeclared and must be supplied.",
            )
        )
        return None, issues

    issues.append(
        warn("unknown-units", f"Unrecognised unit declaration {declared!r}.")
    )
    return declared, issues


def rescale_pair(ds) -> tuple:
    """Read the mapping from stored values to physical ones.

    Args:
        ds: A slice's dataset.

    Returns:
        (slope, intercept, issues). Either may be None if the source declares
        it alone; both are needed for the mapping to mean anything.
    """
    def _get(tag):
        value = ds.get(tag)
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    return _get("RescaleSlope"), _get("RescaleIntercept")


# --- TIFF -----------------------------------------------------------------


# --- headerless raw -------------------------------------------------------

# The pixel types a headerless file may be declared as, and their byte widths.
RAW_DTYPES = {
    "uint8": 1,
    "int8": 1,
    "uint16": 2,
    "int16": 2,
    "uint32": 4,
    "int32": 4,
    "float32": 4,
    "float64": 8,
}


def suggest_shapes(n_bytes: int, max_results: int = 8) -> list:
    """Suggest shapes whose voxel count exactly fills a file of a given size.

    Suggestions only - nothing here is evidence about what the file actually
    is. Assumes square in-plane dimensions, which covers reconstructed volumes.

    Args:
        n_bytes: The file's size.
        max_results: How many suggestions to return.

    Returns:
        (shape, dtype) pairs that would fit exactly, roughly-cubic first.
    """
    results = []
    for name, itemsize in RAW_DTYPES.items():
        if n_bytes % itemsize:
            continue
        n_voxels = n_bytes // itemsize
        side = 32
        while side * side <= n_voxels and side <= 4096:
            if n_voxels % (side * side) == 0:
                depth = n_voxels // (side * side)
                if 1 <= depth <= 20000:
                    results.append((name, (int(depth), side, side)))
            side += 1

    results.sort(key=lambda r: abs(math.log(r[1][1] / max(r[1][0], 1))))
    return results[:max_results]


def tiff_info(path: Path) -> tuple:
    """Read a TIFF's dimensions from its headers. No pixel data is read.

    Args:
        path: The TIFF file.

    Returns:
        (shape, dtype, n_pages). The page count is what separates a multi-page
        volume from a stack of single-page slices.
    """
    with tifffile.TiffFile(str(path)) as tf:
        n_pages = len(tf.pages)
        series = tf.series[0] if tf.series else None
        if series is not None:
            shape = tuple(int(x) for x in series.shape)
            dtype = str(series.dtype)
        else:
            page = tf.pages[0]
            shape = tuple(int(x) for x in page.shape)
            dtype = str(page.dtype)
    return shape, dtype, n_pages
