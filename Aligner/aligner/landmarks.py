"""Recovering a true coordinate from a picked one, and storing it.

The display deliberately misplaces a volume. Orientation is baked into the sampling
and translation into the layer affine, so the world position of a click is *not* the
patient coordinate of the voxel under it. A picker that stored the displayed position
would produce landmarks that look entirely reasonable and are silently wrong - the
failure mode this project has already met twice.

Recovery walks the whole chain back::

    world click -> display index -> source voxel index -> patient mm

Only the middle step depends on the orientation, and it is exact: interpolation
degrades values, never positions. The last step uses the volume's own DICOM geometry,
which the display never touches, so a decimated volume recovers as truthfully as a
full-resolution one.

Landmarks are stored in **patient millimetres** - the one frame that survives
decimation, re-orientation and reloading.

Pure: no napari, no Qt. A session file replays without a viewer, which is what keeps
the solver out of the GUI.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

from .geometry import SeriesGeometry, apply, voxel_to_world
from .oblique import ObliqueView


class LandmarkError(RuntimeError):
    """A session is inconsistent, or a session file cannot be read."""


# Patient coordinates are written rounded, purely so the file stays readable. A
# nanometre is orders of magnitude below anything these scanners resolve.
STORED_DECIMALS = 6


def _matrix(affine) -> np.ndarray:
    """Extract a plain 4x4 from an affine.

    Args:
        affine: A 4x4 array, or a napari Affine wrapping one.

    Returns:
        The 4x4 as a float array.
    """
    return np.asarray(getattr(affine, "affine_matrix", affine), dtype=float)


@dataclass
class Placement:
    """Where a volume is on screen, and how to get back from there to its data.

    Takes the layer's *actual* affine rather than rebuilding one. Before the
    orientation controls are first touched a layer carries its DICOM affine, which
    may have negative axes and a fraction of a degree of tilt; afterwards it carries
    the `ObliqueView` affine, which is positive and axis-aligned. Reading the real
    one is correct in both states.

    Attributes:
        geometry: The volume's own DICOM geometry, which the display never touches.
        affine: The layer's current display affine, display index -> world mm.
        view: The oblique view in force, or None for a layer never re-oriented, in
            which case the display index *is* the source voxel index.
    """

    geometry: SeriesGeometry
    affine: np.ndarray
    view: Optional[ObliqueView] = None

    def __post_init__(self):
        self.affine = _matrix(self.affine)

    # --- the recovery chain ------------------------------------------------

    def world_to_patient(self, point) -> np.ndarray:
        """Recover the true coordinate of a point picked on screen.

        Args:
            point: World (z, y, x) mm position, as clicked.

        Returns:
            Patient (x, y, z) mm of the voxel under it.
        """
        display_index = apply(np.linalg.inv(self.affine), point)
        voxel = (
            display_index
            if self.view is None
            else self.view.to_source_index(display_index)
        )
        return apply(voxel_to_world(self.geometry), voxel)[::-1]

    def patient_to_world(self, point) -> np.ndarray:
        """Find where a stored landmark should currently be drawn.

        Exact inverse of `world_to_patient`.

        Args:
            point: Patient (x, y, z) mm.

        Returns:
            World (z, y, x) mm position on screen.
        """
        world = np.asarray(point, dtype=float)[::-1]
        voxel = apply(np.linalg.inv(voxel_to_world(self.geometry)), world)
        display_index = (
            voxel if self.view is None else self.view.to_display_index(voxel)
        )
        return apply(self.affine, display_index)

    def contains(self, patient_point, margin: float = 0.5) -> bool:
        """Test whether a coordinate falls inside this volume's own grid.

        Args:
            patient_point: Patient (x, y, z) mm.
            margin: Tolerance in voxels beyond the outermost voxel centres.

        Returns:
            True if the point is inside the volume.
        """
        world = np.asarray(patient_point, dtype=float)[::-1]
        voxel = apply(np.linalg.inv(voxel_to_world(self.geometry)), world)
        limits = np.array(self.geometry.shape, dtype=float) - 1.0
        return bool(np.all(voxel >= -margin) and np.all(voxel <= limits + margin))


def placement_for(layer) -> Optional[Placement]:
    """Build a `Placement` from a napari layer.

    Duck-typed on purpose: reads three attributes and imports nothing, so this
    module stays testable without a viewer.

    Args:
        layer: A napari layer, or anything with `.affine` and `.metadata`.

    Returns:
        The `Placement`, or None if the layer holds no volume.
    """
    metadata = getattr(layer, "metadata", None) or {}
    volume = metadata.get("volume")
    if volume is None:
        return None
    return Placement(
        geometry=volume.geometry,
        affine=layer.affine,
        view=metadata.get("view"),
    )


@dataclass(frozen=True)
class Landmark:
    """One picked point, in its own volume's patient frame.

    Attributes:
        id: The pairing key across volumes. Deliberately not the row index.
        volume: Name of the volume it was picked in.
        patient: (x, y, z) mm in that volume's patient frame.
        label: Free text for the user's own notes; it pairs nothing.
    """

    id: int
    volume: str
    patient: Tuple[float, float, float]
    label: str = ""

    def as_array(self) -> np.ndarray:
        """Express the patient coordinate as an array.

        Returns:
            (3,) float array, (x, y, z) in mm.
        """
        return np.asarray(self.patient, dtype=float)


@dataclass
class Session:
    """Which volumes are being registered, in what order, and what was picked.

    Attributes:
        volumes: Volume name -> its series directory.
        chain: Registration order, moving to fixed; the last never moves.
        landmarks: Every point picked, across every volume.
        issued: Highest id ever handed out, including for points since deleted.
            Counting from the *current* landmarks instead would hand a deleted id to
            the next point picked, which would then pair with the other volume's old
            point of that id - a mismatch the fit absorbs silently. Persisted for
            the same reason: reloading must not reset it.
    """

    volumes: Dict[str, Path] = field(default_factory=dict)
    chain: Tuple[str, ...] = ()
    landmarks: List[Landmark] = field(default_factory=list)
    issued: int = 0

    # --- queries ------------------------------------------------------------

    def for_volume(self, name: str) -> List[Landmark]:
        """List the landmarks picked in one volume.

        Args:
            name: Volume name.

        Returns:
            Its landmarks, in the order they were added.
        """
        return [mark for mark in self.landmarks if mark.volume == name]

    def next_id(self) -> int:
        """Allocate the next landmark id.

        Returns:
            An id never used before, so a deleted point cannot be re-paired.
        """
        return 1 + max(self.issued, *(mark.id for mark in self.landmarks), 0)

    def active_chain(self) -> Tuple[str, ...]:
        """List the chain members that actually carry landmarks.

        A volume can be loaded purely for context - a second scan already sharing
        the fixed frame, say - and have nothing to register.

        Returns:
            Chain names holding at least one landmark, in chain order.
        """
        marked = {mark.volume for mark in self.landmarks}
        return tuple(name for name in self.chain if name in marked)

    def skipped(self) -> Tuple[str, ...]:
        """List the chain members passed over.

        Returns:
            Chain names with nothing picked in them, in chain order.
        """
        active = set(self.active_chain())
        return tuple(name for name in self.chain if name not in active)

    def links(self) -> Tuple[Tuple[str, str], ...]:
        """List the registrations to solve.

        Returns:
            Consecutive (source, target) pairs of the *active* chain.
        """
        active = self.active_chain()
        return tuple(zip(active, active[1:]))

    def paired(self, source: str, target: str):
        """Collect the landmarks two volumes have in common.

        Matched by id rather than by row order: deleting one point from one volume
        would otherwise re-pair every later point, and the fit would still succeed -
        just wrongly.

        Args:
            source: Name of the moving volume.
            target: Name of the volume it registers to.

        Returns:
            (ids, source_points, target_points), the two arrays (N, 3) in patient mm
            and row-aligned.
        """
        indexed: Dict[str, Dict[int, Landmark]] = {}
        for mark in self.landmarks:
            indexed.setdefault(mark.volume, {})[mark.id] = mark

        left = indexed.get(source, {})
        right = indexed.get(target, {})
        ids = tuple(sorted(set(left) & set(right)))

        def points(store):
            return np.array(
                [store[i].patient for i in ids], dtype=float
            ).reshape(-1, 3)

        return ids, points(left), points(right)

    # --- editing ------------------------------------------------------------

    def add(self, volume: str, patient, label: str = "", id: Optional[int] = None):
        """Record a picked point.

        Args:
            volume: Volume name it was picked in.
            patient: (x, y, z) mm in that volume's patient frame.
            label: Optional free text.
            id: Explicit id; a fresh one is allocated if omitted.

        Returns:
            The new `Landmark`.
        """
        mark = Landmark(
            id=self.next_id() if id is None else int(id),
            volume=volume,
            patient=tuple(float(v) for v in np.asarray(patient, dtype=float)),
            label=label,
        )
        self.landmarks.append(mark)
        self.issued = max(self.issued, mark.id)
        return mark

    def renumber(self, old: int, new: int) -> None:
        """Declare two landmarks the same feature, by merging their ids.

        Args:
            old: The id to replace, in every volume holding it.
            new: The id to merge into.

        Raises:
            LandmarkError: If some volume already holds both, since one volume
                cannot hold one feature twice and the merge would drop a pick.
        """
        if old == new:
            return
        holding_old = {mark.volume for mark in self.landmarks if mark.id == old}
        holding_new = {mark.volume for mark in self.landmarks if mark.id == new}
        clash = holding_old & holding_new
        if clash:
            raise LandmarkError(
                f"landmark {new} is already picked in {', '.join(sorted(clash))}; "
                "one volume cannot hold the same feature twice."
            )
        self.landmarks = [
            mark if mark.id != old else replace(mark, id=new)
            for mark in self.landmarks
        ]
        self.issued = max(self.issued, new)

    def unlink(self, volume: str, id: int):
        """Give one volume's landmark a fresh id, so it no longer pairs.

        The opposite of `renumber`, and deliberately one-sided: it detaches this
        pick from the feature without disturbing the others that shared the id.

        Args:
            volume: Volume holding the landmark.
            id: Its current id.

        Returns:
            The landmark under its new id, or None if there was no such landmark.
        """
        fresh = self.next_id()
        found = None
        updated = []
        for mark in self.landmarks:
            if mark.volume == volume and mark.id == id:
                found = replace(mark, id=fresh)
                updated.append(found)
            else:
                updated.append(mark)
        if found is not None:
            self.landmarks = updated
            self.issued = max(self.issued, fresh)
        return found

    def remove(self, volume: str, id: int) -> None:
        """Delete one landmark.

        Args:
            volume: Volume name.
            id: Landmark id. Its id is not returned to the pool.
        """
        self.landmarks = [
            mark
            for mark in self.landmarks
            if not (mark.volume == volume and mark.id == id)
        ]

    def validate(self) -> None:
        """Check the session is internally consistent.

        Raises:
            LandmarkError: If the chain names an unknown volume, a landmark belongs
                to an unknown volume, or one volume holds an id twice.
        """
        unknown = [name for name in self.chain if name not in self.volumes]
        if unknown:
            raise LandmarkError(
                f"chain names no volume called {', '.join(unknown)}; "
                f"known volumes are {', '.join(sorted(self.volumes)) or '(none)'}"
            )
        orphans = sorted(
            {mark.volume for mark in self.landmarks} - set(self.volumes)
        )
        if orphans:
            raise LandmarkError(
                f"landmarks refer to unknown volumes: {', '.join(orphans)}"
            )
        seen = set()
        for mark in self.landmarks:
            key = (mark.volume, mark.id)
            if key in seen:
                raise LandmarkError(
                    f"duplicate landmark id {mark.id} in volume {mark.volume!r}"
                )
            seen.add(key)


# --- the session file -----------------------------------------------------


def _portable(directory, base: Path) -> str:
    """Express a path so the session file travels with its data.

    Args:
        directory: The path to store.
        base: Directory the session file sits in.

    Returns:
        A forward-slashed relative path, or an absolute one when no relative path
        exists - on Windows, when the data is on another drive.
    """
    # os.path.relpath rather than Path.relative_to, which cannot produce `..` before
    # 3.12 - the shortcut that produced a real bug in Image2Dicom's job files.
    try:
        return os.path.relpath(Path(directory), base).replace("\\", "/")
    except ValueError:
        return str(Path(directory)).replace("\\", "/")


def save_session(path, session: Session) -> Path:
    """Write a session to YAML, with volume paths relative to it.

    Args:
        path: File to write.
        session: The session to store.

    Returns:
        The path written.

    Raises:
        LandmarkError: If the session does not validate.
    """
    session.validate()
    path = Path(path)
    base = path.parent

    document = {
        "volumes": {
            name: _portable(directory, base) for name, directory in session.volumes.items()
        },
        "chain": list(session.chain),
        "issued": int(session.issued),
        "landmarks": [
            {
                "id": mark.id,
                "volume": mark.volume,
                "patient": [round(float(v), STORED_DECIMALS) for v in mark.patient],
                "label": mark.label,
            }
            for mark in session.landmarks
        ],
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return path


def load_session(path) -> Session:
    """Read a session file.

    Volume paths resolve relative to *the file*, not the working directory - the
    distinction that produced a real bug in Image2Dicom's job files.

    Args:
        path: The session file.

    Returns:
        The `Session`, validated.

    Raises:
        LandmarkError: If the file is unreadable, not valid YAML, malformed, or
            describes an inconsistent session.
    """
    path = Path(path)
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except OSError as exc:
        raise LandmarkError(f"Cannot read session file {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise LandmarkError(f"{path} is not valid YAML: {exc}") from exc

    if not isinstance(document, dict):
        raise LandmarkError(f"{path} should hold a mapping, not {type(document).__name__}")

    base = path.parent
    volumes = {}
    for name, directory in (document.get("volumes") or {}).items():
        volumes[str(name)] = (base / str(directory)).resolve()

    landmarks = []
    for index, entry in enumerate(document.get("landmarks") or []):
        if not isinstance(entry, dict):
            raise LandmarkError(f"{path}: landmark {index} is not a mapping")
        missing = [key for key in ("id", "volume", "patient") if key not in entry]
        if missing:
            raise LandmarkError(
                f"{path}: landmark {index} is missing {', '.join(missing)}"
            )
        patient = entry["patient"]
        if len(patient) != 3:
            raise LandmarkError(
                f"{path}: landmark {entry['id']} has {len(patient)} coordinates, not 3"
            )
        landmarks.append(
            Landmark(
                id=int(entry["id"]),
                volume=str(entry["volume"]),
                patient=tuple(float(v) for v in patient),
                label=str(entry.get("label", "")),
            )
        )

    session = Session(
        volumes=volumes,
        chain=tuple(str(name) for name in (document.get("chain") or [])),
        landmarks=landmarks,
        # Falls back to the highest surviving id for a file written by hand or by an
        # older version, which is the best that can be recovered from one.
        issued=int(document.get("issued") or 0),
    )
    session.validate()
    return session
