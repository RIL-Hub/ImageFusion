"""DICOM geometry, and its mapping into napari's coordinate model.

Three coordinate systems are in play, and conflating them is the main hazard here -
a mirrored volume renders perfectly happily:

* **voxel** - array indices, ``(k, j, i)`` = (slice, row, column).
* **patient** - DICOM's physical space, ``(x, y, z)`` mm, from ImagePositionPatient
  and ImageOrientationPatient.
* **world** - what napari renders in, ``(z, y, x)`` mm, so that its "display the last
  two axes" shows a plane perpendicular to the slice axis.

Pure: no napari, no Qt.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Reverses a homogeneous coordinate's first three components: (a,b,c,1) -> (c,b,a,1).
# Applied on both sides of the DICOM matrix, since our array axes are reversed
# relative to DICOM's (i, j, k) and we emit (z, y, x).
REVERSE_XYZ = np.array(
    [
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)

# Its 3x3 half, for conjugating a patient-space rotation into world space.
REVERSE3 = REVERSE_XYZ[:3, :3]


@dataclass(frozen=True)
class SeriesGeometry:
    """The geometry of one DICOM series, exactly as the files declare it.

    Attributes:
        origin: ImagePositionPatient of the first slice, (x, y, z) in mm.
        orientation: ImageOrientationPatient, six direction cosines.
        pixel_spacing: (between rows, between columns) in mm, DICOM's order.
        slice_spacing: Distance between slice centres, in mm.
        shape: (n_slices, rows, columns).
    """

    origin: tuple
    orientation: tuple
    pixel_spacing: tuple
    slice_spacing: float
    shape: tuple

    @property
    def spacing(self) -> tuple:
        """Voxel size along each array axis.

        Returns:
            (k, j, i) voxel size in mm.
        """
        # The one place the PixelSpacing pairing is written down: [0] is between
        # rows and belongs to row index j, [1] is between columns and belongs to i.
        row_spacing, column_spacing = self.pixel_spacing
        return (self.slice_spacing, row_spacing, column_spacing)

    @property
    def row_direction(self) -> np.ndarray:
        """Direction of increasing *column* index.

        Returns:
            Unit vector, IOP's first triplet.
        """
        return np.array(self.orientation[:3], dtype=float)

    @property
    def column_direction(self) -> np.ndarray:
        """Direction of increasing *row* index.

        Returns:
            Unit vector, IOP's second triplet.
        """
        return np.array(self.orientation[3:], dtype=float)

    @property
    def slice_direction(self) -> np.ndarray:
        """Direction of increasing slice index.

        Returns:
            Unit normal of the slice plane. Falls back to +z for a degenerate
            orientation.
        """
        normal = np.cross(self.row_direction, self.column_direction)
        norm = np.linalg.norm(normal)
        return normal / norm if norm > 0 else np.array([0.0, 0.0, 1.0])


def voxel_to_patient(geometry: SeriesGeometry) -> np.ndarray:
    """Build the DICOM standard's own voxel-to-patient mapping.

    Args:
        geometry: The series geometry.

    Returns:
        4x4 affine mapping voxel (i, j, k) to patient (x, y, z) in mm.
    """
    # DICOM PS3.3 C.7.6.2.1.1. The spacing pairing is the usual place to go wrong:
    # i is the column index and steps by PixelSpacing[1]; j is the row index and
    # steps by PixelSpacing[0].
    row_spacing, column_spacing = geometry.pixel_spacing

    matrix = np.eye(4)
    matrix[:3, 0] = geometry.row_direction * column_spacing
    matrix[:3, 1] = geometry.column_direction * row_spacing
    matrix[:3, 2] = geometry.slice_direction * geometry.slice_spacing
    matrix[:3, 3] = np.array(geometry.origin, dtype=float)
    return matrix


def voxel_to_world(geometry: SeriesGeometry) -> np.ndarray:
    """Build the affine napari places a layer with.

    The DICOM matrix with both its input and output axis orders reversed. Reversing
    only one gives a mirrored volume that looks entirely plausible.

    Args:
        geometry: The series geometry.

    Returns:
        4x4 affine mapping array (k, j, i) to world (z, y, x) in mm.
    """
    return REVERSE_XYZ @ voxel_to_patient(geometry) @ REVERSE_XYZ


def apply(matrix: np.ndarray, points) -> np.ndarray:
    """Transform points by a homogeneous matrix.

    Args:
        matrix: 4x4 affine.
        points: (N, 3) array of points, or a single (3,) point.

    Returns:
        The transformed points, in the same shape as `points`.
    """
    points = np.asarray(points, dtype=float)
    single = points.ndim == 1
    if single:
        points = points.reshape(1, 3)

    homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])
    transformed = (matrix @ homogeneous.T).T[:, :3]
    return transformed.reshape(3) if single else transformed


def centre_world(affine: np.ndarray, shape) -> np.ndarray:
    """Locate the centre voxel of a volume in world space.

    Args:
        affine: 4x4 voxel-to-world affine.
        shape: Volume shape, (k, j, i).

    Returns:
        World position (z, y, x) in mm of the centre voxel.
    """
    centre_index = np.array([(n - 1) / 2.0 for n in shape], dtype=float)
    return affine[:3, :3] @ centre_index + affine[:3, 3]


def axis_tilt_degrees(matrix: np.ndarray) -> float:
    """Measure how far an affine is from being axis-aligned.

    napari can only slice orthogonally, so it drops any out-of-plane component and
    warns. This measures what that costs.

    Args:
        matrix: 4x4 affine.

    Returns:
        Largest angle in degrees between any axis of `matrix` and the world axis it
        most nearly follows. 0 means axis-aligned.
    """
    worst = 0.0
    for column in range(3):
        direction = matrix[:3, column]
        norm = np.linalg.norm(direction)
        if norm == 0:
            continue
        alignment = float(np.max(np.abs(direction / norm)))
        worst = max(worst, float(np.degrees(np.arccos(min(1.0, alignment)))))
    return worst


def tilt_displacement_mm(geometry: SeriesGeometry) -> float:
    """Measure what ignoring a series' tilt costs in millimetres.

    Args:
        geometry: The series geometry.

    Returns:
        Worst-case error in mm, at the far corner: the sine of the tilt times the
        volume's longest extent.
    """
    tilt = np.radians(axis_tilt_degrees(voxel_to_world(geometry)))
    return float(np.sin(tilt) * max(extent_mm(geometry)))


def tilt_is_subvoxel(geometry: SeriesGeometry) -> bool:
    """Decide whether a series' tilt is small enough to ignore.

    Args:
        geometry: The series geometry.

    Returns:
        True when ignoring the tilt moves nothing by as much as one voxel.
    """
    return tilt_displacement_mm(geometry) < min(geometry.spacing)


def extent_mm(geometry: SeriesGeometry) -> tuple:
    """Measure the physical size of a volume.

    Args:
        geometry: The series geometry.

    Returns:
        (k, j, i) extent in mm.
    """
    return tuple(n * s for n, s in zip(geometry.shape, geometry.spacing))


def display_state(geometry: SeriesGeometry, rotation, translation) -> tuple:
    """Convert a solved transform into display state, changing no data.

    Two conversions are involved: patient becomes world by conjugation, and the
    series' direction cosines are folded in - the display grid ignores them, so even
    an identity transform needs them to show the volume truthfully.

    Args:
        geometry: The moving series' geometry.
        rotation: 3x3 rotation in patient (x, y, z).
        translation: (3,) translation in patient (x, y, z) mm.

    Returns:
        (matrix, nudge): the 3x3 orientation for `ObliqueView`, and the (3,) offset
        in world mm for its display affine.
    """
    affine = voxel_to_world(geometry)
    centre = centre_world(affine, geometry.shape)

    # Affine columns are direction vectors scaled by voxel size; divide it out.
    cosines = _orthonormal(affine[:3, :3] / np.asarray(geometry.spacing, dtype=float))

    world_rotation = REVERSE3 @ np.asarray(rotation, dtype=float) @ REVERSE3
    matrix = world_rotation @ cosines
    nudge = (
        REVERSE3 @ np.asarray(translation, dtype=float)
        + world_rotation @ centre
        - centre
    )
    return matrix, nudge


def _orthonormal(matrix: np.ndarray) -> np.ndarray:
    """Square up a matrix that should be orthonormal but may not quite be.

    Scanners round their direction cosines. The oblique sampler inverts its matrix
    by transposing, which is only correct for an orthonormal one, so a rounding
    error would otherwise become a shear.

    Args:
        matrix: 3x3, nearly orthonormal.

    Returns:
        The nearest orthonormal 3x3, by polar decomposition.
    """
    left, _, right = np.linalg.svd(np.asarray(matrix, dtype=float))
    return left @ right
