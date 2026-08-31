"""Shared vocabulary for Image2Dicom.

These types are the contract between the pipeline stages. Nothing here reads files
or knows about a specific image format.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional


class InputClass(str, Enum):
    """What kind of data the target path holds, decided by content."""

    DICOM = "dicom"
    TIFF = "tiff"
    MIXED = "mixed"
    UNIDENTIFIED = "unidentified"
    EMPTY = "empty"


class Severity(str, Enum):
    INFO = "info"
    WARN = "warn"
    ERROR = "error"


@dataclass(frozen=True)
class Issue:
    """Something the user needs to know. Errors block conversion."""

    severity: Severity
    code: str
    message: str


def info(code: str, message: str) -> Issue:
    """Build an informational issue.

    Args:
        code: Short stable identifier.
        message: What to tell the user.

    Returns:
        The `Issue`.
    """
    return Issue(Severity.INFO, code, message)


def warn(code: str, message: str) -> Issue:
    """Build a warning, which convert will proceed past once acknowledged.

    Args:
        code: Short stable identifier.
        message: What to tell the user.

    Returns:
        The `Issue`.
    """
    return Issue(Severity.WARN, code, message)


def error(code: str, message: str) -> Issue:
    """Build an error, which blocks conversion outright.

    Args:
        code: Short stable identifier.
        message: What to tell the user.

    Returns:
        The `Issue`.
    """
    return Issue(Severity.ERROR, code, message)


@dataclass(frozen=True)
class FileRecord:
    """One file found while scanning.

    Attributes:
        path: Where it is.
        size: Its size in bytes.
    """

    path: Path
    size: int

    @property
    def name(self) -> str:
        """Read the file name without its directory.

        Returns:
            The bare file name.
        """
        return self.path.name


@dataclass
class Characterization:
    """Everything known about a candidate volume, without reading pixel data."""

    shape: Optional[tuple] = None  # (nz, ny, nx)
    dtype: Optional[str] = None
    modality: Optional[str] = None  # resolved from SOP class where possible
    declared_modality: Optional[str] = None  # the Modality tag as found
    units: Optional[str] = None  # what the rescaled values mean: HU, arbitrary, ...

    pixel_spacing: Optional[tuple] = None  # (row_mm, col_mm)
    slice_spacing: Optional[float] = None
    spacing_source: str = "none"

    rescale_slope: Optional[float] = None
    rescale_intercept: Optional[float] = None

    order_source: str = "none"

    @property
    def has_spacing(self) -> bool:
        """Test whether voxel size is fully known.

        Returns:
            True if both pixel and slice spacing were found.
        """
        return self.pixel_spacing is not None and self.slice_spacing is not None

    @property
    def extent_mm(self) -> Optional[tuple]:
        """Measure the volume's physical size.

        Returns:
            (k, j, i) extent in mm, or None if shape or spacing is unknown.
        """
        if self.shape is None or not self.has_spacing:
            return None
        nz, ny, nx = self.shape
        return (
            nz * self.slice_spacing,
            ny * self.pixel_spacing[0],
            nx * self.pixel_spacing[1],
        )


@dataclass
class VolumeCandidate:
    """One 3D volume that could be converted."""

    key: str  # stable identifier for this volume
    label: str  # human-readable, distinguishes it from siblings
    input_class: InputClass
    files: list = field(default_factory=list)  # ordered list[Path]
    characterization: Characterization = field(default_factory=Characterization)
    issues: list = field(default_factory=list)  # list[Issue]
    detail: dict = field(default_factory=dict)  # free-form, shown under each candidate

    @property
    def n_files(self) -> int:
        """Count the files making up this volume.

        Returns:
            How many there are.
        """
        return len(self.files)

    @property
    def errors(self) -> list:
        """Collect the issues that block conversion.

        Returns:
            The error-severity issues.
        """
        return [i for i in self.issues if i.severity is Severity.ERROR]

    @property
    def warnings(self) -> list:
        """Collect the issues that need acknowledging.

        Returns:
            The warning-severity issues.
        """
        return [i for i in self.issues if i.severity is Severity.WARN]

    @property
    def convertible(self) -> bool:
        """Test whether this volume can be converted at all.

        Returns:
            True if nothing blocks it. Warnings do not.
        """
        return not self.errors


@dataclass
class ProbeResult:
    target: Path
    input_class: InputClass
    n_files_scanned: int = 0
    candidates: list = field(default_factory=list)  # list[VolumeCandidate]
    issues: list = field(default_factory=list)  # input-level issues
    unreadable: list = field(default_factory=list)  # list[Path]
    skipped: list = field(default_factory=list)  # list[Path]

    @property
    def n_candidates(self) -> int:
        """Count the volumes this input resolves to.

        Returns:
            How many were found. Convert needs exactly one.
        """
        return len(self.candidates)

    @property
    def volume(self) -> Optional[VolumeCandidate]:
        """Pick out the single volume this input holds.

        Returns:
            The `VolumeCandidate`, or None if the input resolves to none or to
            several - in which case the path must be narrowed.
        """
        return self.candidates[0] if len(self.candidates) == 1 else None


def to_jsonable(obj: Any) -> Any:
    """Convert a dataclass tree to something json.dump can handle.

    Args:
        obj: Any dataclass, Enum, Path, mapping, sequence or scalar.

    Returns:
        The same structure in plain JSON types.
    """
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if hasattr(obj, "__dataclass_fields__"):
        return {
            name: to_jsonable(getattr(obj, name))
            for name in obj.__dataclass_fields__  # type: ignore[attr-defined]
        }
    return obj
