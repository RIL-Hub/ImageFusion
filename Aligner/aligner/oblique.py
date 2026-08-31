"""Viewing a volume along an arbitrarily oriented grid, one plane at a time.

This is how Amide stays instant while a whole-volume resample takes seconds. The data
is never transformed. An orientation matrix is kept instead, and each plane the viewer
asks for is sampled from the source on demand: one 512x512 plane of a 397x512x512
volume costs 262,000 samples rather than 104 million.

There is one orientation - a 3x3 matrix mapping *source physical* directions to
*display physical* directions. Coarse turns and arbitrary rotations are the same
operation on it; a quarter turn merely happens to land on exact integer coordinates.
The three ways of building one live here too, since the sampler is their only consumer.

The display grid follows from the matrix: spacings take the matrix's nearest axis
permutation, and the grid is sized to the rotated bounding box so corners survive.

Pure: no napari, no Qt.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np

CORNERS = np.array(
    [(k, j, i) for k in (0, 1) for j in (0, 1) for i in (0, 1)], dtype=float
)

# Which plane each world axis rotates in. World order is (z, y, x).
ROTATION_PLANE = ((1, 2), (2, 0), (0, 1))


def nearest_signed_permutation(matrix: np.ndarray) -> np.ndarray:
    """Find the signed permutation closest to an orthonormal matrix.

    Used only to decide which source spacing belongs to which display axis, so a
    quarter turn of an anisotropic volume keeps its proportions.

    Args:
        matrix: 3x3, orthonormal.

    Returns:
        3x3 signed permutation, entries in {-1, 0, 1}.
    """
    matrix = np.asarray(matrix, dtype=float)
    result = np.zeros((3, 3))

    candidates = sorted(
        ((abs(matrix[r, c]), r, c) for r in range(3) for c in range(3)),
        reverse=True,
    )
    rows, columns = set(range(3)), set(range(3))
    for _, row, column in candidates:
        if row in rows and column in columns:
            result[row, column] = np.sign(matrix[row, column]) or 1.0
            rows.discard(row)
            columns.discard(column)
    return result


def axis_rotation(axis: int, degrees: float) -> np.ndarray:
    """Build a rotation about one display axis, by any angle.

    Args:
        axis: World axis to rotate about, 0=z, 1=y, 2=x.
        degrees: Angle; positive is clockwise in this convention.

    Returns:
        3x3 rotation in (z, y, x).
    """
    angle = np.radians(degrees)
    cos, sin = np.cos(angle), np.sin(angle)
    a, b = ROTATION_PLANE[axis]

    matrix = np.eye(3)
    matrix[a, a] = cos
    matrix[a, b] = -sin
    matrix[b, a] = sin
    matrix[b, b] = cos
    return matrix


def quarter_turn(axis: int, clockwise: bool = True) -> np.ndarray:
    """Build an exact 90 degree rotation about one display axis.

    Args:
        axis: World axis to rotate about, 0=z, 1=y, 2=x.
        clockwise: Direction of the turn.

    Returns:
        3x3 rotation in (z, y, x), exactly a signed permutation.
    """
    # Built from integers rather than cos/sin, so the matrix stays exactly a signed
    # permutation - otherwise entries drift to 6e-17 and sampled coordinates stop
    # landing on grid points.
    plane = ROTATION_PLANE[axis]
    a, b = plane if clockwise else plane[::-1]
    matrix = np.eye(3)
    matrix[a, a] = matrix[b, b] = 0.0
    matrix[a, b] = -1.0
    matrix[b, a] = 1.0
    return matrix


def flip(axis: int) -> np.ndarray:
    """Build a mirror of one display axis.

    Args:
        axis: World axis to mirror, 0=z, 1=y, 2=x.

    Returns:
        3x3 improper transform, so it reverses handedness.
    """
    matrix = np.eye(3)
    matrix[axis, axis] = -1.0
    return matrix


@dataclass
class ObliqueView:
    """A lazily-sampled view of a volume under an orientation matrix.

    Attributes:
        source: The volume, indexed (k, j, i).
        source_spacing: Its voxel size in mm per array axis.
        matrix: 3x3 mapping source physical directions to display physical ones.
        order: scipy interpolation order; 0 is nearest neighbour, 1 linear.
        fill: Value returned outside the volume.
        spacing: Derived voxel size of the display grid, (k, j, i) in mm.
        shape: Derived display grid shape, sized to the rotated bounding box.
    """

    source: np.ndarray
    source_spacing: Tuple[float, float, float]
    matrix: np.ndarray = field(default_factory=lambda: np.eye(3))
    order: int = 1
    fill: float = 0.0

    def __post_init__(self):
        self.matrix = np.asarray(self.matrix, dtype=float)
        source_spacing = np.asarray(self.source_spacing, dtype=float)
        source_shape = np.array(self.source.shape, dtype=float)

        permutation = nearest_signed_permutation(self.matrix)
        # Display axis a indexes whichever source axis the permutation sends there.
        self.spacing = tuple(
            float(source_spacing[int(np.argmax(np.abs(permutation[a])))])
            for a in range(3)
        )

        # Size the grid to the rotated bounding box, so corners survive.
        source_centre = (source_shape - 1.0) / 2.0
        offsets = (CORNERS * (source_shape - 1.0) - source_centre) * source_spacing
        rotated = offsets @ self.matrix.T
        extent = rotated.max(axis=0) - rotated.min(axis=0)
        self.shape = tuple(
            int(np.ceil(extent[a] / self.spacing[a])) + 1 for a in range(3)
        )

        scale_source = np.diag(source_spacing)
        scale_display = np.diag(np.asarray(self.spacing, dtype=float))

        # Output index -> source index. The matrix maps source to display, so
        # sampling needs its inverse; being orthonormal, that is its transpose.
        self._index_matrix = (
            np.linalg.inv(scale_source) @ self.matrix.T @ scale_display
        )
        display_centre = (np.array(self.shape, dtype=float) - 1.0) / 2.0
        self._offset = source_centre - self._index_matrix @ display_centre

    # --- sampling ---------------------------------------------------------

    def plane(self, index: int, axis: int = 0) -> np.ndarray:
        """Sample one plane of the display grid from the source.

        Args:
            index: Which plane, along `axis`.
            axis: Display axis the plane is perpendicular to.

        Returns:
            The 2D plane as float32, with `axis` squeezed out.
        """
        from scipy.ndimage import affine_transform

        shape = list(self.shape)
        shape[axis] = 1

        step = np.zeros(3)
        step[axis] = float(index)
        offset = self._offset + self._index_matrix @ step

        sampled = affine_transform(
            self.source,
            self._index_matrix,
            offset=offset,
            output_shape=tuple(shape),
            order=self.order,
            mode="constant",
            cval=self.fill,
            output=np.float32,
        )
        return np.squeeze(sampled, axis=axis)

    def as_dask(self, axis: int = 0):
        """Build a lazy array of the whole display grid.

        Args:
            axis: The axis on napari's slider. Chunked along any other, napari
                computes every chunk to take one row from each - the difference
                between instant and unusable.

        Returns:
            A dask array of `shape`, chunked one plane thick along `axis`.
        """
        import dask.array as da
        from dask import delayed

        plane_shape = tuple(n for a, n in enumerate(self.shape) if a != axis)
        planes = [
            da.from_delayed(
                delayed(self.plane)(index, axis), shape=plane_shape, dtype=np.float32
            )
            for index in range(self.shape[axis])
        ]
        return da.stack(planes, axis=axis)

    # --- coordinates ------------------------------------------------------

    def to_source_index(self, display_index) -> np.ndarray:
        """Trace a display voxel back to the source voxel it was drawn from.

        Exact: interpolation degrades values, never positions. Landmarks must be
        recorded through this, never from the displayed coordinate, since the
        display deliberately misplaces the volume.

        Args:
            display_index: (k, j, i) index into the display grid.

        Returns:
            (k, j, i) index into the source, as floats.
        """
        display_index = np.asarray(display_index, dtype=float)
        return self._index_matrix @ display_index + self._offset

    def to_display_index(self, source_index) -> np.ndarray:
        """Find where a source voxel is shown. Exact inverse of `to_source_index`.

        Needed to redraw a stored landmark after the orientation changes.

        Args:
            source_index: (k, j, i) index into the source.

        Returns:
            (k, j, i) index into the display grid, as floats.
        """
        source_index = np.asarray(source_index, dtype=float)
        return np.linalg.solve(self._index_matrix, source_index - self._offset)

    def display_affine(self, world_centre, nudge=(0.0, 0.0, 0.0)) -> np.ndarray:
        """Build the affine that places this grid in world space.

        Pure scale and translation, which is all napari can render - every part of
        the orientation lives in the sampling instead.

        Args:
            world_centre: (z, y, x) mm the volume's centre should sit at.
            nudge: (z, y, x) mm offset added to that.

        Returns:
            4x4 affine mapping display index to world mm.
        """
        linear = np.diag(np.asarray(self.spacing, dtype=float))
        centre = (np.array(self.shape, dtype=float) - 1.0) / 2.0

        affine = np.eye(4)
        affine[:3, :3] = linear
        affine[:3, 3] = (
            np.asarray(world_centre, dtype=float)
            - linear @ centre
            + np.asarray(nudge, dtype=float)
        )
        return affine
