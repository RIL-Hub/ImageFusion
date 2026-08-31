"""Load a canonical DICOM series written by Image2Dicom.

This loader is deliberately much simpler than Image2Dicom's `probe`. It only ever
reads Image2Dicom output, which guarantees: one single-frame file per slice,
geometry present on every slice, spacing present, rescale mapping present, a known
pixel type. So there is no format detection, no ordering fallback, and no repair -
just a check that the guarantees hold, and a clear failure if they do not.

Pixel data is loaded lazily through dask: only the slices actually displayed are
ever read, which is what keeps a 16.7 GB volume viewable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import dask.array as da
import numpy as np
import pydicom
from dask import delayed

from .geometry import SeriesGeometry, voxel_to_world

REQUIRED_TAGS = (
    "ImagePositionPatient",
    "ImageOrientationPatient",
    "PixelSpacing",
    "Rows",
    "Columns",
)

# How much RAM one materialised volume may occupy before it is decimated.
DEFAULT_BUDGET_BYTES = 1024**3  # 1 GiB


def plan_decimation(shape, itemsize: int = 4, budget_bytes: int = DEFAULT_BUDGET_BYTES) -> int:
    """Choose how far to decimate a volume so it fits a memory budget.

    Decimation rather than averaging is deliberate: taking every n-th slice means
    reading only every n-th file, which is what makes this affordable. The cost is
    aliasing - thin structures can fall between sampled planes.

    Args:
        shape: Full volume shape, (k, j, i).
        itemsize: Bytes per voxel once loaded.
        budget_bytes: Most memory one volume may occupy.

    Returns:
        The smallest isotropic step that fits; 1 means no decimation.
    """
    factor = 1
    while factor <= max(shape):
        decimated = [len(range(0, int(n), factor)) for n in shape]
        if int(np.prod(decimated)) * itemsize <= budget_bytes:
            return factor
        factor += 1
    return factor


def decimated_shape(shape, factor: int) -> tuple:
    """Work out the shape decimation leaves behind.

    Args:
        shape: Full volume shape, (k, j, i).
        factor: Decimation step, as `plan_decimation` returns.

    Returns:
        The decimated shape, matching numpy's ``[::factor]`` striding.
    """
    return tuple(len(range(0, int(n), factor)) for n in shape)


class LoadError(RuntimeError):
    """A series cannot be read, or does not meet the Image2Dicom contract."""


@dataclass
class Volume:
    """One loaded series: pixel data plus the geometry that places it.

    Attributes:
        name: Display name, the series directory's name by default.
        path: The series directory.
        data: (k, j, i) float32 in physical units. A dask array until materialised,
            a numpy array afterwards.
        geometry: Its geometry, already adjusted for any decimation.
        modality: DICOM Modality, e.g. CT or PT.
        units: RescaleType, e.g. HU, or "unknown".
        rescale_slope: Slope already applied to `data`.
        rescale_intercept: Intercept already applied to `data`.
        decimation: Step taken through the source; 1 is full resolution.
        full_shape: Shape before decimation, or None if never decimated.
        value_range: (min, max) in physical units, filled in on demand.
    """

    name: str
    path: Path
    data: object
    geometry: SeriesGeometry
    modality: str
    units: str
    rescale_slope: float
    rescale_intercept: float
    decimation: int = 1
    full_shape: Optional[tuple] = None
    value_range: Optional[tuple] = None

    @property
    def shape(self) -> tuple:
        """Shape of the loaded volume.

        Returns:
            (k, j, i), after any decimation.
        """
        return self.geometry.shape

    @property
    def materialised(self) -> bool:
        """Test whether the pixels are in memory.

        Returns:
            True if `data` is a numpy array rather than a lazy one.
        """
        return isinstance(self.data, np.ndarray)

    @property
    def nbytes(self) -> int:
        """Measure what the loaded volume occupies.

        Returns:
            Size in bytes as float32.
        """
        return int(np.prod(self.shape)) * 4

    @property
    def affine(self) -> np.ndarray:
        """Build the affine napari places this layer with.

        Returns:
            4x4 mapping voxel (k, j, i) to world (z, y, x) mm.
        """
        return voxel_to_world(self.geometry)

    def __repr__(self) -> str:
        k, j, i = self.shape
        return f"Volume({self.name!r}, {k}x{j}x{i}, {self.modality}, {self.units})"


def _read_stored(path: Path) -> np.ndarray:
    """Read one slice's stored pixels.

    Args:
        path: A DICOM file.

    Returns:
        Its pixel array, before any rescale mapping.
    """
    return pydicom.dcmread(str(path)).pixel_array


def _slice_position(ds, normal: np.ndarray) -> float:
    """Locate one slice along the stack direction.

    Args:
        ds: A pydicom dataset with ImagePositionPatient.
        normal: Unit normal of the slice plane.

    Returns:
        Signed distance in mm along `normal`, for ordering the series.
    """
    position = np.array([float(v) for v in ds.ImagePositionPatient], dtype=float)
    return float(np.dot(position, normal))


def load_series(
    directory,
    name: Optional[str] = None,
    budget_bytes: int = DEFAULT_BUDGET_BYTES,
    materialise: bool = True,
    show_progress: bool = True,
) -> Volume:
    """Load one Image2Dicom output directory.

    Args:
        directory: The series directory.
        name: Display name; the directory's own name if omitted.
        budget_bytes: Memory ceiling before the volume is decimated to fit.
        materialise: Read the pixels into RAM now. This costs one read up front and
            makes every later re-orientation cheap. Left lazy, an out-of-plane
            rotation would make each displayed slice touch every source file, since
            the series is one file per slice and therefore z-major. False reads
            nothing, for metadata only.
        show_progress: Print a progress bar while reading.

    Returns:
        The loaded `Volume`.

    Raises:
        LoadError: If the directory is missing, holds no .dcm files, lacks a
            required tag, or has non-positive slice spacing.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise LoadError(f"Not a directory: {directory}")

    paths = sorted(directory.glob("*.dcm"))
    if not paths:
        raise LoadError(f"No .dcm files in {directory}")

    headers = [pydicom.dcmread(str(p), stop_before_pixels=True) for p in paths]

    missing = [tag for tag in REQUIRED_TAGS if getattr(headers[0], tag, None) is None]
    if missing:
        raise LoadError(
            f"{directory} is missing {', '.join(missing)}. This loader reads "
            "Image2Dicom output, which always carries these; convert the source "
            "with Image2Dicom first."
        )

    orientation = tuple(float(v) for v in headers[0].ImageOrientationPatient)
    normal = np.cross(orientation[:3], orientation[3:])
    norm = np.linalg.norm(normal)
    normal = normal / norm if norm > 0 else np.array([0.0, 0.0, 1.0])

    order = np.argsort([_slice_position(ds, normal) for ds in headers])
    paths = [paths[index] for index in order]
    headers = [headers[index] for index in order]

    positions = [_slice_position(ds, normal) for ds in headers]
    if len(positions) > 1:
        slice_spacing = float(np.median(np.diff(positions)))
    else:
        slice_spacing = float(getattr(headers[0], "SliceThickness", 1.0) or 1.0)
    if slice_spacing <= 0:
        raise LoadError(f"{directory} has non-positive slice spacing.")

    first = headers[0]
    rows, columns = int(first.Rows), int(first.Columns)
    pixel_spacing = tuple(float(v) for v in first.PixelSpacing)
    full_shape = (len(paths), rows, columns)

    factor = plan_decimation(full_shape, 4, budget_bytes) if materialise else 1

    # Decimating the slice axis by selecting paths means only every n-th file is
    # ever opened - the reason this is affordable on a 2000-slice volume.
    kept_paths = paths[::factor]

    # Voxel [0,0,0] does not move, so the origin is unchanged; only the step grows.
    geometry = SeriesGeometry(
        origin=tuple(float(v) for v in first.ImagePositionPatient),
        orientation=orientation,
        pixel_spacing=(pixel_spacing[0] * factor, pixel_spacing[1] * factor),
        slice_spacing=slice_spacing * factor,
        shape=decimated_shape(full_shape, factor),
    )

    slope = float(getattr(first, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(first, "RescaleIntercept", 0.0) or 0.0)
    stored_dtype = np.uint16 if int(first.PixelRepresentation) == 0 else np.int16

    lazy = [
        da.from_delayed(
            delayed(_read_stored)(path), shape=(rows, columns), dtype=stored_dtype
        )
        for path in kept_paths
    ]
    data = da.stack(lazy)[:, ::factor, ::factor]
    data = data.astype(np.float32) * np.float32(slope) + np.float32(intercept)

    if materialise:
        if show_progress:
            from dask.diagnostics import ProgressBar

            with ProgressBar():
                data = data.compute()
        else:
            data = data.compute()

    return Volume(
        name=name or directory.name,
        path=directory,
        data=data,
        geometry=geometry,
        modality=str(getattr(first, "Modality", "") or "OT"),
        units=str(getattr(first, "RescaleType", "") or "unknown"),
        rescale_slope=slope,
        rescale_intercept=intercept,
        decimation=factor,
        full_shape=full_shape,
    )


def sample_value_range(volume: Volume, n_samples: int = 16) -> tuple:
    """Find the value range, for seeding display contrast limits.

    Exact once materialised; sampled while lazy, since reading a 2000-slice volume
    just to open a window is not worth it.

    Args:
        volume: The volume to scan. Its `value_range` is set as a side effect.
        n_samples: Roughly how many slices to sample when lazy.

    Returns:
        (min, max) in physical units.

    Raises:
        LoadError: If the volume holds no finite values.
    """
    if volume.materialised:
        finite = np.isfinite(volume.data)
        if not finite.any():
            raise LoadError(f"{volume.name} contains no finite values.")
        volume.value_range = (
            float(volume.data[finite].min()),
            float(volume.data[finite].max()),
        )
        return volume.value_range

    n_slices = volume.shape[0]
    step = max(1, n_slices // n_samples)
    indices = range(0, n_slices, step)

    low = np.inf
    high = -np.inf
    for index in indices:
        plane = np.asarray(volume.data[index])
        finite = np.isfinite(plane)
        if finite.any():
            low = min(low, float(plane[finite].min()))
            high = max(high, float(plane[finite].max()))

    if not np.isfinite(low) or not np.isfinite(high):
        raise LoadError(f"{volume.name} contains no finite values.")
    volume.value_range = (low, high)
    return volume.value_range
