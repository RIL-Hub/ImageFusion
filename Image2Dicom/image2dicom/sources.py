"""Slice-at-a-time readers for each input format.

Every source exposes the same tiny interface so the writer never branches on format:
``len(source)`` slices, ``source.raw(i)`` returns one 2D array of *stored* values.
Nothing loads a whole volume - peak memory is one slice.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pydicom
import tifffile

from .characterize import RAW_DTYPES
from .model import InputClass


class DicomSliceSource:
    """One file per slice, or one multi-frame file."""

    def __init__(self, files):
        self.files = [Path(f) for f in files]
        self._n_frames = None  # set only for a multi-frame single file
        self._frames = None  # its pixel array, loaded on first use

        if len(self.files) == 1:
            ds = pydicom.dcmread(str(self.files[0]), stop_before_pixels=True)
            n_frames = int(ds.get("NumberOfFrames") or 1)
            if n_frames > 1:
                self._n_frames = n_frames

    def __len__(self) -> int:
        """Count the slices.

        Returns:
            How many this source can read.
        """
        return self._n_frames or len(self.files)

    def raw(self, index: int) -> np.ndarray:
        """Read one slice of stored values.

        Args:
            index: Slice number.

        Returns:
            A 2D array, exactly as stored - no rescaling.
        """
        if self._n_frames is None:
            return pydicom.dcmread(str(self.files[index])).pixel_array
        if self._frames is None:
            self._frames = pydicom.dcmread(str(self.files[0])).pixel_array
        return self._frames[index]

    def close(self):
        """Release whatever the source is holding open."""
        self._frames = None


class TiffVolumeSource:
    """A single multi-page TIFF holding one whole volume."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._tf = tifffile.TiffFile(str(self.path))
        self._shape = tuple(int(v) for v in self._tf.series[0].shape)
        # Normally one page per slice, which keeps reads streaming. A volume that
        # TIFF stored as a single multi-sample page has to be read whole.
        self._paged = len(self._tf.pages) == self._shape[0]
        self._whole = None

    def __len__(self) -> int:
        """Count the slices.

        Returns:
            How many this source can read.
        """
        return self._shape[0]

    def raw(self, index: int) -> np.ndarray:
        """Read one slice of stored values.

        Args:
            index: Slice number.

        Returns:
            A 2D array, exactly as stored - no rescaling.
        """
        if self._paged:
            return self._tf.pages[index].asarray()
        if self._whole is None:
            self._whole = self._tf.series[0].asarray()
        return self._whole[index]

    def close(self):
        """Release whatever the source is holding open."""
        self._whole = None
        self._tf.close()


class TiffStackSource:
    """One single-page TIFF per slice."""

    def __init__(self, files):
        self.files = [Path(f) for f in files]

    def __len__(self) -> int:
        """Count the slices.

        Returns:
            How many this source can read.
        """
        return len(self.files)

    def raw(self, index: int) -> np.ndarray:
        """Read one slice of stored values.

        Args:
            index: Slice number.

        Returns:
            A 2D array, exactly as stored - no rescaling.
        """
        return tifffile.imread(str(self.files[index]))

    def close(self):
        """Release whatever the source is holding open."""
        pass


class RawVolumeSource:
    """A headerless binary volume, described entirely by the job file.

    Memory-mapped, so only the slices actually touched are ever resident - a 16 GB
    raw file costs the same as a small one.
    """

    def __init__(self, path, shape, dtype, byte_order="little", header_bytes=0):
        path = Path(path)
        shape = tuple(int(v) for v in shape)
        if len(shape) != 3:
            raise ValueError("raw.shape must be three values: [z, y, x]")
        if dtype not in RAW_DTYPES:
            raise ValueError(
                f"Unsupported raw.dtype {dtype!r}. "
                f"Known: {', '.join(sorted(RAW_DTYPES))}"
            )

        itemsize = RAW_DTYPES[dtype]
        expected = shape[0] * shape[1] * shape[2] * itemsize + header_bytes
        actual = path.stat().st_size
        if expected != actual:
            raise ValueError(
                f"Declared geometry does not match the file. "
                f"{shape[0]}x{shape[1]}x{shape[2]} {dtype} + {header_bytes} header "
                f"bytes needs {expected:,} bytes, but {path.name} is {actual:,} "
                f"({actual - expected:+,})."
            )

        numpy_dtype = np.dtype(dtype).newbyteorder(
            "<" if byte_order == "little" else ">"
        )
        self.path = path
        self.shape = shape
        self._map = np.memmap(
            str(path), dtype=numpy_dtype, mode="r", offset=header_bytes, shape=shape
        )

    def __len__(self) -> int:
        """Count the slices.

        Returns:
            How many this source can read.
        """
        return int(self.shape[0])

    def raw(self, index: int) -> np.ndarray:
        """Read one slice of stored values.

        Args:
            index: Slice number.

        Returns:
            A 2D array, exactly as stored - no rescaling.
        """
        return np.asarray(self._map[index])

    def close(self):
        """Release whatever the source is holding open."""
        self._map = None


def open_source(candidate, raw=None):
    """Build the right slice reader for a volume candidate.

    Args:
        candidate: The volume to read.
        raw: The job file's `raw:` declaration, required for input with no
            readable header.

    Returns:
        A source exposing ``len()`` and ``raw(i)``.

    Raises:
        ValueError: If the input has no header and no `raw:` declaration, or
            its class has no reader.
    """
    if candidate.input_class is InputClass.DICOM:
        return DicomSliceSource(candidate.files)
    if candidate.input_class is InputClass.TIFF:
        if candidate.detail.get("single_file_volume"):
            return TiffVolumeSource(candidate.files[0])
        return TiffStackSource(candidate.files)
    if candidate.input_class is InputClass.UNIDENTIFIED:
        if raw is None:
            raise ValueError(
                "This input has no readable header. Describe it with a 'raw:' block "
                "in the job file: shape, dtype, byte_order, header_bytes."
            )
        return RawVolumeSource(
            candidate.files[0],
            shape=raw.shape,
            dtype=raw.dtype,
            byte_order=raw.byte_order,
            header_bytes=raw.header_bytes,
        )
    raise ValueError(f"No reader for input class {candidate.input_class}")
