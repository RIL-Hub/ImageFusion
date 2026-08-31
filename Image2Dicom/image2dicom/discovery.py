"""Inventory, content-based classification, and candidate enumeration.

Implements VALIDATOR.md. The one question answered here is "does this input hold
exactly one 3D volume, and if not, what are the choices".
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Callable, Optional

import pydicom

from . import characterize as ch
from .model import (
    Characterization,
    FileRecord,
    InputClass,
    ProbeResult,
    VolumeCandidate,
    error,
    info,
    warn,
)

# Never image data.
_IGNORED_NAMES = {"dicomdir", "thumbs.db", ".ds_store", "desktop.ini"}
_IGNORED_SUFFIXES = {
    ".txt", ".xml", ".json", ".csv", ".log", ".md", ".ini",
    ".yml", ".yaml", ".pdf", ".png", ".jpg", ".jpeg", ".zip", ".py",
}

# Only the tags the validator needs. Limiting the parse keeps a 2000-file scan cheap.
DICOM_TAGS = [
    "SOPClassUID",
    "SeriesInstanceUID",
    "SeriesNumber",
    "SeriesDescription",
    "StudyDescription",
    "Modality",
    "Rows",
    "Columns",
    "NumberOfFrames",
    "ImagePositionPatient",
    "ImageOrientationPatient",
    "FrameOfReferenceUID",
    "SliceLocation",
    "InstanceNumber",
    "PixelSpacing",
    "SliceThickness",
    "SpacingBetweenSlices",
    "RescaleSlope",
    "RescaleIntercept",
    "RescaleType",
    "Units",
    "BitsAllocated",
    "BitsStored",
    "PixelRepresentation",
    "PhotometricInterpretation",
    "Manufacturer",
    "ManufacturerModelName",
]

TIFF_MAGIC = (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+")


# --- inventory ------------------------------------------------------------


def inventory(target: Path) -> tuple:
    """Walk a path, listing every file worth looking at.

    Args:
        target: A file or a directory; directories are walked recursively.

    Returns:
        (records, skipped): a `FileRecord` per candidate file, and the paths
        passed over as sidecars or metadata.
    """
    target = Path(target)
    if target.is_file():
        return [FileRecord(target, target.stat().st_size)], []

    records: list = []
    skipped: list = []
    def walk_key(path: Path):
        # Group by directory first, then numerically within it, so files from
        # different subdirectories do not interleave.
        return (str(path.parent).lower(), ch.numeric_key(path))

    for path in sorted(target.rglob("*"), key=walk_key):
        if not path.is_file():
            continue
        if path.name.startswith("."):
            skipped.append(path)
            continue
        if path.name.lower() in _IGNORED_NAMES or path.suffix.lower() in _IGNORED_SUFFIXES:
            skipped.append(path)
            continue
        try:
            size = path.stat().st_size
        except OSError:
            skipped.append(path)
            continue
        records.append(FileRecord(path, size))
    return records, skipped


# --- content probes -------------------------------------------------------


def read_dicom_header(path: Path, tags=DICOM_TAGS):
    """Read a file's DICOM metadata, if it has any.

    Tries a normal read first. Files without the 128-byte preamble need
    force=True, which will happily "succeed" on arbitrary bytes, so a forced
    read is accepted only with corroborating evidence.

    Args:
        path: The file to read.
        tags: Which tags to keep.

    Returns:
        The dataset, or None if this is not DICOM.
    """
    try:
        return pydicom.dcmread(str(path), stop_before_pixels=True, specific_tags=tags)
    except Exception:
        pass

    try:
        ds = pydicom.dcmread(
            str(path), stop_before_pixels=True, force=True, specific_tags=tags
        )
    except Exception:
        return None

    if ds.get("SOPClassUID") is not None:
        return ds
    if ds.get("Rows") is not None and ds.get("Columns") is not None:
        return ds
    return None


def is_tiff(path: Path) -> bool:
    """Test whether a file is a TIFF, by its magic bytes.

    Args:
        path: The file to test.

    Returns:
        True if it starts with a TIFF signature.
    """
    try:
        with open(path, "rb") as handle:
            return handle.read(4) in TIFF_MAGIC
    except OSError:
        return False


# Files sampled when deciding DICOM vs TIFF vs raw. Spread across the listing, so a
# stray file at either end cannot decide the class.
CLASSIFY_SAMPLE_SIZE = 8


def _spread_sample(records, n: int) -> list:
    """Sample evenly across a listing, so a stray leading file cannot decide.

    Args:
        records: The files found.
        n: How many to sample.

    Returns:
        Up to `n` records spread across the listing, or all of them if there
        are fewer.
    """
    if n <= 0 or len(records) <= n:
        return list(records)
    step = len(records) / n
    return [records[min(int(i * step), len(records) - 1)] for i in range(n)]


def classify(records) -> tuple:
    """Decide what kind of data these files hold, by content rather than name.

    Args:
        records: The files found.

    Returns:
        (input_class, issues): what the input is, and anything the user should
        know about a mixed or unreadable set.
    """
    issues: list = []
    if not records:
        return InputClass.EMPTY, issues

    samples = _spread_sample(records, CLASSIFY_SAMPLE_SIZE)
    dicom_hits = sum(1 for r in samples if read_dicom_header(r.path) is not None)
    tiff_hits = sum(1 for r in samples if is_tiff(r.path))
    n = len(samples)

    if dicom_hits == n:
        return InputClass.DICOM, issues
    if tiff_hits == n:
        return InputClass.TIFF, issues
    if dicom_hits and tiff_hits:
        issues.append(
            error(
                "mixed-content",
                f"Sample of {n} files contains both DICOM ({dicom_hits}) and TIFF "
                f"({tiff_hits}). Narrow the target to one format.",
            )
        )
        return InputClass.MIXED, issues
    if dicom_hits:
        issues.append(
            warn(
                "partial-dicom",
                f"{dicom_hits} of {n} sampled files parsed as DICOM; the rest did not. "
                "Foreign files will be reported as unreadable.",
            )
        )
        return InputClass.DICOM, issues
    if tiff_hits:
        issues.append(
            warn(
                "partial-tiff",
                f"{tiff_hits} of {n} sampled files are TIFF; the rest are not.",
            )
        )
        return InputClass.TIFF, issues

    return InputClass.UNIDENTIFIED, issues


# --- candidate enumeration ------------------------------------------------


def _dicom_label(uid: str, first_ds, n_files: int) -> str:
    """Build a readable name for one series, distinguishing it from siblings.

    Args:
        uid: Its SeriesInstanceUID.
        first_ds: The first slice's dataset.
        n_files: How many slices it has.

    Returns:
        A one-line label.
    """
    parts = []
    modality = first_ds.get("Modality")
    if modality:
        parts.append(str(modality))
    number = first_ds.get("SeriesNumber")
    if number is not None:
        parts.append(f"series {number}")
    description = first_ds.get("SeriesDescription")
    if description:
        parts.append(str(description).strip())
    if not parts:
        parts.append(uid[-12:] if len(uid) > 12 else uid)
    parts.append(f"{n_files} file(s)")
    return " | ".join(parts)


def _build_dicom_candidate(uid: str, items) -> VolumeCandidate:
    """Characterize one DICOM series into a candidate volume.

    Args:
        uid: Its SeriesInstanceUID.
        items: (record, dataset) pairs belonging to it.

    Returns:
        The `VolumeCandidate`, with its slices ordered and any issues found.
    """
    issues: list = []

    ordered, order_source, positions, order_issues = ch.order_dicom(items)
    issues.extend(order_issues)
    issues.extend(ch.check_consistency(ordered))

    pixel_spacing, slice_spacing, spacing_source, spacing_issues = ch.derive_spacing(
        ordered, positions
    )
    issues.extend(spacing_issues)

    _, first = ordered[0]
    slope, intercept = ch.rescale_pair(first)

    modality, modality_issues = ch.resolve_modality(first)
    issues.extend(modality_issues)

    units, units_issues = ch.resolve_units(first)
    issues.extend(units_issues)

    if pixel_spacing is None:
        issues.append(
            error("missing-pixel-spacing", "PixelSpacing absent; must be supplied.")
        )
    if slice_spacing is None:
        issues.append(
            error("missing-slice-spacing", "Slice spacing undeterminable; must be supplied.")
        )
    if slope is None or intercept is None:
        issues.append(
            warn(
                "missing-rescale",
                "RescaleSlope/RescaleIntercept absent; stored values will be treated "
                "as already being in physical units.",
            )
        )

    characterization = Characterization(
        shape=ch.dicom_shape(ordered),
        dtype=ch.dicom_dtype(first),
        modality=modality,
        declared_modality=str(first.get("Modality")) if first.get("Modality") else None,
        units=units,
        pixel_spacing=pixel_spacing,
        slice_spacing=slice_spacing,
        spacing_source=spacing_source,
        rescale_slope=slope,
        rescale_intercept=intercept,
        order_source=order_source,
    )

    origin = ch.as_floats(first.get("ImagePositionPatient"))
    orientation = ch.as_floats(first.get("ImageOrientationPatient"))

    detail = {
        "series_instance_uid": uid,
        "origin": origin,
        "orientation": orientation,
        "frame_of_reference": str(first.get("FrameOfReferenceUID") or ""),
        "manufacturer": str(first.get("Manufacturer") or ""),
        "model": str(first.get("ManufacturerModelName") or ""),
        "sop_class_uid": str(first.get("SOPClassUID") or ""),
        "rescale_type": str(first.get("RescaleType") or ""),
        "units": str(first.get("Units") or ""),
        "number_of_frames": str(first.get("NumberOfFrames") or ""),
        "directory": str(ordered[0][0].path.parent),
    }

    return VolumeCandidate(
        key=uid,
        label=_dicom_label(uid, first, len(ordered)),
        input_class=InputClass.DICOM,
        files=[rec.path for rec, _ in ordered],
        characterization=characterization,
        issues=issues,
        detail=detail,
    )


def dicom_candidates(records, on_file: Optional[Callable] = None) -> tuple:
    """Group DICOM files into series, one candidate volume each.

    Args:
        records: The files found.
        on_file: Called as each file is read, for progress.

    Returns:
        (candidates, unreadable): one `VolumeCandidate` per
        SeriesInstanceUID, and the paths that would not parse.
    """
    groups: dict = defaultdict(list)
    unreadable: list = []

    for record in records:
        ds = read_dicom_header(record.path)
        if on_file is not None:
            on_file()
        if ds is None:
            unreadable.append(record.path)
            continue
        uid = ds.get("SeriesInstanceUID")
        uid = str(uid) if uid else "<missing-series-uid>"
        groups[uid].append((record, ds))

    candidates = [_build_dicom_candidate(uid, items) for uid, items in groups.items()]
    candidates.sort(key=lambda c: c.label)

    if "<missing-series-uid>" in groups:
        for candidate in candidates:
            if candidate.key == "<missing-series-uid>":
                candidate.issues.append(
                    warn(
                        "no-series-uid",
                        "These files carry no SeriesInstanceUID, so they were grouped "
                        "together by default. Verify they are one volume.",
                    )
                )
    return candidates, unreadable


def tiff_candidates(records, on_file: Optional[Callable] = None) -> tuple:
    """Group TIFF files into candidate volumes.

    A multi-page TIFF is a volume of its own; single-page TIFFs are slices of
    one volume, ordered numerically by name.

    Args:
        records: The files found.
        on_file: Called as each file is read, for progress.

    Returns:
        (candidates, unreadable): the volumes found, and the paths that would
        not parse.
    """
    multipage: list = []
    singlepage: list = []
    unreadable: list = []

    for record in records:
        try:
            shape, dtype, n_pages = ch.tiff_info(record.path)
        except Exception:
            unreadable.append(record.path)
            if on_file is not None:
                on_file()
            continue
        if on_file is not None:
            on_file()
        # A 3D series is a volume, however it is paged. Usually that is N pages of
        # one slice each; it can also be a single page whose trailing axis TIFF
        # calls "samples". Page count alone would miss the second case.
        if len(shape) == 3:
            multipage.append((record, shape, dtype, n_pages))
        else:
            singlepage.append((record, shape, dtype))

    candidates: list = []

    for record, shape, dtype, n_pages in multipage:
        volume_shape = tuple(shape)
        issues = [
            error(
                "missing-spacing",
                "TIFF carries no voxel spacing; it must be supplied.",
            ),
            warn("missing-modality", "TIFF carries no modality; it must be supplied."),
            warn(
                "unknown-units",
                "TIFF declares no units; the physical meaning of these float values "
                "must be supplied.",
            ),
        ]
        candidates.append(
            VolumeCandidate(
                key=record.path.name,
                label=f"{ch.label_fields(record.path)} | {n_pages} pages | {record.path.name}",
                input_class=InputClass.TIFF,
                files=[record.path],
                characterization=Characterization(
                    shape=volume_shape,
                    dtype=dtype,
                    order_source="page order",
                    spacing_source="none",
                ),
                issues=issues,
                detail={
                    "pages": n_pages,
                    "single_file_volume": True,
                    "directory": str(record.path.parent),
                },
            )
        )

    if singlepage:
        ordered = sorted(singlepage, key=lambda t: ch.numeric_key(t[0].path))
        shapes = {t[1] for t in ordered}
        dtypes = {t[2] for t in ordered}
        issues = [
            error("missing-spacing", "TIFF carries no voxel spacing; it must be supplied."),
            warn("missing-modality", "TIFF carries no modality; it must be supplied."),
            warn(
                "order-fallback",
                "Ordered by numeric filename key; TIFF has no positional geometry.",
            ),
        ]
        if len(shapes) > 1:
            issues.append(
                error(
                    "inconsistent-slices",
                    f"Slice dimensions differ across files: {sorted(shapes)[:4]}",
                )
            )
        if len(dtypes) > 1:
            issues.append(
                error("inconsistent-slices", f"Pixel types differ across files: {sorted(dtypes)}")
            )
        first_shape = ordered[0][1]
        candidates.append(
            VolumeCandidate(
                key="tiff-stack",
                label=f"single-page TIFF stack | {len(ordered)} slices",
                input_class=InputClass.TIFF,
                files=[t[0].path for t in ordered],
                characterization=Characterization(
                    shape=(len(ordered),) + tuple(first_shape[-2:]),
                    dtype=ordered[0][2],
                    order_source="filename",
                    spacing_source="none",
                ),
                issues=issues,
                detail={"directory": str(ordered[0][0].path.parent)},
            )
        )

    if multipage and singlepage:
        for candidate in candidates:
            candidate.issues.append(
                error(
                    "mixed-tiff",
                    "The input mixes multi-page TIFF volumes with single-page TIFF "
                    "slices. Narrow the target.",
                )
            )

    candidates.sort(key=lambda c: ch.numeric_key(Path(c.key)))
    return candidates, unreadable


def raw_candidates(records) -> tuple:
    """Treat each headerless file as one volume.

    Nothing about such a file can be inferred, so each candidate carries an
    error until the job file declares its shape and dtype.

    Args:
        records: The files found.

    Returns:
        (candidates, unreadable): one candidate per file, and the paths that
        could not even be sized.
    """
    candidates: list = []

    for record in records:
        suggestions = ch.suggest_shapes(record.size)
        issues = [
            error(
                "raw-declaration-required",
                f"{record.name} has no readable header ({record.size:,} bytes). "
                "Declare shape, dtype, byte_order and header_bytes in a [raw] "
                "section; the declared size is checked against the file.",
            ),
            error("missing-spacing", "Voxel spacing must be supplied."),
            warn("missing-modality", "Modality must be supplied."),
            warn("unknown-units", "Units must be supplied."),
        ]
        candidates.append(
            VolumeCandidate(
                key=record.name,
                label=f"headerless | {record.size:,} bytes | {record.name}",
                input_class=InputClass.UNIDENTIFIED,
                files=[record.path],
                characterization=Characterization(
                    order_source="declared", spacing_source="none"
                ),
                issues=issues,
                detail={
                    "size_bytes": record.size,
                    "directory": str(record.path.parent),
                    "shape_suggestions": [
                        f"{name} {d}x{h}x{w}" for name, (d, h, w) in suggestions
                    ],
                },
            )
        )

    candidates.sort(key=lambda c: ch.numeric_key(Path(c.key)))
    return candidates, []


# --- top level ------------------------------------------------------------


def probe(target: Path, on_file: Optional[Callable] = None) -> ProbeResult:
    """Inspect an image set: what is here, and does it resolve to one volume.

    Reads headers only; pixel data is never touched.

    Args:
        target: A file or directory holding one image set.
        on_file: Called as each file is read, for progress.

    Returns:
        The `ProbeResult`, whose `volume` is the single candidate found, or
        None if the input holds none or several.

    Raises:
        FileNotFoundError: If the target does not exist.
    """
    target = Path(target)
    if not target.exists():
        raise FileNotFoundError(f"Target not found: {target}")

    records, skipped = inventory(target)
    result = ProbeResult(
        target=target,
        input_class=InputClass.EMPTY,
        n_files_scanned=len(records),
        skipped=skipped,
    )

    dicomdir = [p for p in skipped if p.name.lower() == "dicomdir"]
    if dicomdir:
        result.issues.append(
            info(
                "dicomdir-present",
                f"A DICOMDIR is present ({dicomdir[0]}). Files were scanned directly, "
                "which is more reliable than trusting the index.",
            )
        )

    if not records:
        result.issues.append(error("empty-input", "No candidate image files found."))
        return result

    input_class, class_issues = classify(records)
    result.input_class = input_class
    result.issues.extend(class_issues)

    if input_class is InputClass.DICOM:
        candidates, unreadable = dicom_candidates(records, on_file=on_file)
    elif input_class is InputClass.TIFF:
        candidates, unreadable = tiff_candidates(records, on_file=on_file)
    elif input_class is InputClass.UNIDENTIFIED:
        result.issues.append(
            info(
                "unidentified",
                "Nothing here parses as DICOM or TIFF. Treating each file as a "
                "headerless volume; geometry and pixel type must be declared in a "
                "[raw] section - none of it is inferred.",
            )
        )
        candidates, unreadable = raw_candidates(records)
    else:  # MIXED
        return result

    result.candidates = candidates
    result.unreadable = unreadable

    if unreadable:
        result.issues.append(
            warn(
                "unreadable-files",
                f"{len(unreadable)} file(s) in the target could not be parsed and were "
                "excluded. The volume may be incomplete.",
            )
        )

    if not candidates:
        result.issues.append(error("no-volumes", "No volumes could be formed."))
    elif len(candidates) > 1:
        result.issues.append(
            error(
                "ambiguous-input",
                f"{len(candidates)} distinct volumes found. Narrow the target directory "
                "or re-run with --select.",
            )
        )

    return result
