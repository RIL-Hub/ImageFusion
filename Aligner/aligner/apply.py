"""Applying a registration: rewriting geometry, copying voxels untouched.

A rigid transform maps one voxel grid onto a rotated, translated grid, and DICOM
already expresses that exactly - `ImagePositionPatient` says where the first voxel
sits and `ImageOrientationPatient` says which way the axes point. So applying a
registration is a change of about forty numbers per slice. The pixel data is copied
through byte for byte.

That is the whole reason nothing is resampled. Interpolating the volume onto PET's
grid would blur every voxel, change the array size, and destroy the values a later
attenuation conversion needs - to express something the file format can already say.

The output is oblique with respect to PET's axes whenever the rotation is not a
multiple of a quarter turn, which is honest and which Amide and Slicer read
correctly. Our own viewer cannot *render* an oblique affine - that is what the
oblique sampler and "show fit" exist for - so a registered series loaded back into
`aligner view` will report a tilt larger than a voxel. Believe the file, not that
view.

Pure: no napari, no Qt. `aligner apply` re-runs a whole registration from a session
file with the viewer uninstalled.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pydicom
import yaml

from .landmarks import Session
from .solve import ChainSolution, Rigid


class ApplyError(RuntimeError):
    """A registration cannot be written out."""


# DICOM's DS value representation is a decimal string of at most 16 characters, so
# a full-precision repr does not fit. Same rule as Image2Dicom's writer; duplicated
# rather than shared because the two programs are deliberately separate.
DS_LIMIT = 16


def ds_decimal(value: float) -> str:
    """Format a number as a DICOM DS value.

    Args:
        value: The number to write.

    Returns:
        The most precise decimal string of at most `DS_LIMIT` characters, so
        positions survive the round trip.
    """
    value = float(value)
    for digits in range(DS_LIMIT, 0, -1):
        text = f"{value:.{digits}g}"
        if len(text) <= DS_LIMIT:
            return text
    return "0"


def rewrite_geometry(dataset, transform: Rigid) -> None:
    """Move one slice by rewriting its geometry tags, in place.

    Pixel data is not read, let alone written.

    Args:
        dataset: A pydicom dataset with ImagePositionPatient and
            ImageOrientationPatient.
        transform: The rigid transform to apply, in patient mm.
    """
    position = np.array([float(v) for v in dataset.ImagePositionPatient], dtype=float)
    orientation = np.array(
        [float(v) for v in dataset.ImageOrientationPatient], dtype=float
    ).reshape(2, 3)

    dataset.ImagePositionPatient = [
        ds_decimal(v) for v in transform.rotation @ position + transform.translation
    ]
    # The direction cosines rotate but do not translate: they are directions, and a
    # translation applied to them would shear the volume rather than move it.
    dataset.ImageOrientationPatient = [
        ds_decimal(v) for v in (orientation @ transform.rotation.T).reshape(6)
    ]


def frame_of_reference(directory) -> Optional[str]:
    """Read the frame of reference a moved volume should adopt.

    Args:
        directory: A series directory.

    Returns:
        Its FrameOfReferenceUID, or None if the first slice carries none.
    """
    for path in sorted(Path(directory).glob("*.dcm")):
        dataset = pydicom.dcmread(str(path), stop_before_pixels=True)
        uid = getattr(dataset, "FrameOfReferenceUID", None)
        if uid:
            return str(uid)
        break
    return None


@dataclass
class Written:
    """What one transformed series cost and where it went.

    Attributes:
        name: Source series directory name.
        destination: Directory written to.
        slices: Number of files written.
        rotation_degrees: How far the applied transform turns.
        shift_mm: How far it translates.
    """

    name: str
    destination: Path
    slices: int
    rotation_degrees: float
    shift_mm: float


def transform_series(
    source,
    destination,
    transform: Rigid,
    frame_uid: Optional[str] = None,
    description: Optional[str] = None,
    overwrite: bool = False,
) -> Written:
    """Write a copy of one series, moved by a rigid transform.

    Args:
        source: Series directory to copy.
        destination: Directory to write into; created if absent.
        transform: The rigid transform, in patient mm.
        frame_uid: Frame of reference the volume is being registered *into*.
            Adopting it is the point of the exercise: after registration the two
            series genuinely share a coordinate system, and that tag is how a viewer
            knows. None leaves the source's own.
        description: SeriesDescription to write, truncated to 64 characters.
        overwrite: Replace .dcm files already in `destination`.

    Returns:
        A `Written` record of what was produced.

    Raises:
        ApplyError: If the source holds no .dcm files, or the destination is not
            empty and `overwrite` is false.
    """
    source, destination = Path(source), Path(destination)
    paths = sorted(source.glob("*.dcm"))
    if not paths:
        raise ApplyError(f"No .dcm files in {source}")

    existing = sorted(destination.glob("*.dcm")) if destination.is_dir() else []
    if existing and not overwrite:
        raise ApplyError(
            f"{destination} already holds {len(existing)} .dcm files. "
            "Pass overwrite to replace them."
        )
    destination.mkdir(parents=True, exist_ok=True)

    series_uid = pydicom.uid.generate_uid()
    for path in paths:
        dataset = pydicom.dcmread(str(path))
        rewrite_geometry(dataset, transform)

        # A different geometry is a different image, so it needs its own identity.
        # Reusing the source's UIDs would make two contradictory objects claim to be
        # the same one, and archives resolve that by keeping whichever they saw first.
        dataset.SeriesInstanceUID = series_uid
        dataset.SOPInstanceUID = pydicom.uid.generate_uid()
        if getattr(dataset, "file_meta", None) is not None:
            dataset.file_meta.MediaStorageSOPInstanceUID = dataset.SOPInstanceUID
        if frame_uid:
            dataset.FrameOfReferenceUID = frame_uid
        if description:
            dataset.SeriesDescription = description[:64]

        dataset.save_as(str(destination / path.name))

    return Written(
        name=source.name,
        destination=destination,
        slices=len(paths),
        rotation_degrees=transform.angle_degrees,
        shift_mm=float(np.linalg.norm(transform.translation)),
    )


def apply_session(
    session: Session,
    solution: ChainSolution,
    output,
    overwrite: bool = False,
) -> List[Written]:
    """Write every moving volume into the fixed volume's frame.

    The fixed volume is not copied: it never moves, so a copy would be the same
    series under a new identity - exactly the confusion new UIDs exist to prevent.
    Volumes the solution passed over are not copied either.

    Args:
        session: The session naming each volume's series directory.
        solution: The solved chain.
        output: Directory to write each `<name>_to_<fixed>` series into.
        overwrite: Replace series already written there.

    Returns:
        A `Written` record per series produced.

    Raises:
        ApplyError: If a volume's series directory is missing, or a destination is
            not empty and `overwrite` is false.
    """
    output = Path(output)
    fixed_directory = session.volumes.get(solution.fixed)
    frame_uid = frame_of_reference(fixed_directory) if fixed_directory else None

    written = []
    for name, transform in solution.to_fixed.items():
        if name == solution.fixed:
            continue
        source = session.volumes.get(name)
        if source is None or not Path(source).is_dir():
            raise ApplyError(f"{name}: no such series directory: {source}")
        written.append(
            transform_series(
                source,
                output / f"{name}_to_{solution.fixed}",
                transform,
                frame_uid=frame_uid,
                description=f"{name} registered to {solution.fixed}",
                overwrite=overwrite,
            )
        )
    return written


def save_transforms(path, session: Session, solution: ChainSolution) -> Path:
    """Write the registration itself: the matrices, and how well they fit.

    The numbers, not the pixels. Records the residuals alongside each transform, so
    a matrix is never separated from the evidence for it.

    Args:
        path: File to write.
        session: The session that was solved.
        solution: The solved chain.

    Returns:
        The path written.
    """
    path = Path(path)

    def matrix(transform: Rigid):
        return [[round(float(v), 9) for v in row] for row in transform.matrix]

    document = {
        "fixed": solution.fixed,
        "chain": list(session.chain),
        "links": [
            {
                "source": source,
                "target": target,
                "landmarks": list(solution.fits[(source, target)].ids),
                "rmsd_mm": round(solution.fits[(source, target)].rmsd, 6),
                "worst_mm": round(solution.fits[(source, target)].worst, 6),
                "measured_scale": round(solution.fits[(source, target)].scale, 6),
                "collinear": bool(solution.fits[(source, target)].collinear),
                "matrix": matrix(solution.fits[(source, target)].transform),
            }
            for source, target in solution.links
        ],
        # Patient mm of that volume -> patient mm of the fixed volume.
        "to_fixed": {
            name: matrix(transform) for name, transform in solution.to_fixed.items()
        },
        "end_to_end": {
            "landmarks": list(solution.end_to_end_ids),
            "rmsd_mm": (
                None
                if solution.end_to_end_rmsd is None
                else round(solution.end_to_end_rmsd, 6)
            ),
        },
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return path
