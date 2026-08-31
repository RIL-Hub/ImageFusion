"""Picking landmarks, and keeping them glued to the data rather than to the screen.

One napari Points layer per volume, which is most of why napari was chosen: adding,
dragging and deleting points are all free, and each layer carries its own colour and
its own id labels.

The Session is the source of truth and the Points layer is a rendering of it. Every
change the user makes is absorbed back into patient millimetres immediately, through
`Placement.world_to_patient`; every change to the *display* redraws the points from
those stored millimetres, through `Placement.patient_to_world`. So a landmark stays
on the feature it was picked on however the volume is later rotated or nudged, and
the number written to file never depends on where the volume happened to be shown.

Ids live in the layer's `features` table rather than in row order. napari renumbers
rows on deletion, so pairing by position would silently re-pair every later landmark
against the wrong one in the other volume - a mismatch the fit absorbs without
complaint.
"""

from __future__ import annotations

import numpy as np

from ..geometry import extent_mm
from ..landmarks import Session, placement_for

# One per volume, in layer order. Distinct hues rather than a colormap, since these
# only ever need telling apart.
POINT_COLORS = ("#ff3b30", "#32d74b", "#0a84ff", "#ffd60a", "#bf5af2")

# Point radius as a fraction of the volume's longest side, so a 2 mm Zeiss subregion
# and a 200 mm PET field both get a usable marker.
SIZE_FRACTION = 0.012

SUFFIX = " landmarks"

# How close a row must be to a stored landmark to count as the same one, in mm.
# The conversion round-trips to floating-point exactness, so this only has to be
# below the smallest deliberate drag - which is many orders of magnitude larger.
MATCH_TOLERANCE_MM = 1e-6


class LandmarkPicker:
    """Keeps one Points layer per volume in step with a `Session`."""

    def __init__(self, viewer, session: Session = None):
        self._viewer = viewer
        self.session = Session() if session is None else session
        self._points = {}  # volume name -> points layer
        self._images = {}  # volume name -> image layer
        self._busy = False  # suppresses absorb while we are the ones writing
        # Called with no arguments whenever the landmarks change. A plain list of
        # callables rather than a Qt signal, so this module stays importable - and
        # testable - without Qt.
        self.listeners = []

    # --- setup --------------------------------------------------------------

    def attach(self, image_layer, index: int = 0):
        """Give one image layer its own landmark layer.

        Args:
            image_layer: The image layer to attach to.
            index: Position in the load order, which picks its colour.

        Returns:
            The new Points layer.
        """
        volume = image_layer.metadata["volume"]
        name = volume.name
        size = SIZE_FRACTION * max(extent_mm(volume.geometry))

        points = self._viewer.add_points(
            np.empty((0, 3), dtype=float),
            ndim=3,
            name=f"{name}{SUFFIX}",
            size=size,
            face_color=POINT_COLORS[index % len(POINT_COLORS)],
            features={"id": np.zeros(0, dtype=int)},
            text={"string": "{id}", "size": 9, "color": "white"},
            # Matching the image layers. napari compares units across every layer
            # and silently stops using them for rendering if any disagree.
            units=("mm", "mm", "mm"),
            axis_labels=("z", "y", "x"),
        )
        # Points sit in world millimetres directly, so no affine of their own: the
        # display transform is already baked into where we put them.
        #
        # Landmarks are worth seeing from a slice or two away, since a pick rarely
        # lands exactly on a displayed plane. napari 0.9 replaced
        # out_of_slice_display with projection_mode, so prefer the new spelling and
        # fall back rather than emit a deprecation warning on every layer.
        if hasattr(points, "projection_mode"):
            _try(points, "projection_mode", "all")
        else:
            _try(points, "out_of_slice_display", True)
        _try(points.feature_defaults, "id", 0)

        self._images[name] = image_layer
        self._points[name] = points
        self.session.volumes.setdefault(name, volume.path)
        if name not in self.session.chain:
            # Load order is the chain order: plug -> pot -> PET, with the last one
            # fixed. The panel can reorder it; this is only the starting guess.
            self.session.chain = self.session.chain + (name,)

        points.events.data.connect(lambda *_: self.absorb(name))
        self._notify()
        return points

    def names(self):
        """List the volumes being picked in.

        Returns:
            Their names, in attach order.
        """
        return tuple(self._images)

    def image(self, name: str):
        """Find the image layer a volume name belongs to.

        Args:
            name: Volume name.

        Returns:
            Its image layer, or None if it is not open.
        """
        return self._images.get(name)

    def _notify(self) -> None:
        """Tell every listener the landmarks have changed."""
        for listener in list(self.listeners):
            listener()

    # --- session <- screen --------------------------------------------------

    def absorb(self, name: str) -> None:
        """Take whatever is now in one points layer into the session, in mm.

        Covers adding, dragging and deleting with one path, rather than three that
        could disagree. Does nothing while we are the ones writing to the layer.

        Args:
            name: Volume name whose layer changed.
        """
        if self._busy:
            return
        layer, image = self._points.get(name), self._images.get(name)
        if layer is None or image is None:
            return
        place = placement_for(image)
        if place is None:
            return

        positions = np.asarray(layer.data, dtype=float).reshape(-1, 3)
        picked = [place.world_to_patient(row) for row in positions]
        ids = self._identify(name, layer, picked)

        self.session.landmarks = [
            mark for mark in self.session.landmarks if mark.volume != name
        ]
        for point, identifier in zip(picked, ids):
            self.session.add(name, point, id=int(identifier))

        self._write(layer, positions, ids)
        self._notify()

    def add_at(self, name: str, world):
        """Add a landmark to one volume at a world position.

        The same path a click takes: the point goes into the layer, whose data
        event absorbs it back into patient millimetres and issues it an id. One
        route in, so a click and this cannot disagree about what a position means.

        Args:
            name: Volume to add to.
            world: World (z, y, x) mm position.

        Returns:
            (landmark, reason): the new `Landmark` and an empty string, or None and
            why not - the volume is not open, or the point is outside it.
        """
        layer, image = self._points.get(name), self._images.get(name)
        if layer is None or image is None:
            return None, f"{name} is not open"
        place = placement_for(image)
        if place is None:
            return None, f"{name} has no geometry to place a point in"

        patient = place.world_to_patient(world)
        if not place.contains(patient):
            return None, f"that point is outside {name}"

        before = {mark.id for mark in self.session.for_volume(name)}
        layer.data = np.vstack(
            [
                np.asarray(layer.data, dtype=float).reshape(-1, 3),
                np.asarray(world, dtype=float).reshape(1, 3),
            ]
        )
        fresh = [
            mark for mark in self.session.for_volume(name) if mark.id not in before
        ]
        return (fresh[0] if fresh else None), ""

    def delete(self, name: str, id: int) -> bool:
        """Remove one landmark from a volume and from its layer.

        Args:
            name: Volume holding it.
            id: Landmark id.

        Returns:
            True if something was removed.
        """
        if name not in self._points:
            return False
        before = len(self.session.for_volume(name))
        self.session.remove(name, int(id))
        if len(self.session.for_volume(name)) == before:
            return False
        self.reposition(self._images.get(name))
        self._notify()
        return True

    def unlink(self, name: str, id: int):
        """Detach one volume's landmark from the feature it shares an id with.

        Args:
            name: Volume holding it.
            id: Landmark id.

        Returns:
            The landmark under its new id, or None if there was no such landmark.
        """
        mark = self.session.unlink(name, int(id))
        if mark is None:
            return None
        self.reposition(self._images.get(name))
        self._notify()
        return mark

    def _identify(self, name: str, layer, picked) -> np.ndarray:
        """Work out which landmark each row is: by position, then by id, then new.

        Position comes first because it is the only evidence that cannot go stale.
        napari keeps the features table in step when points are deleted through the
        UI, but assigning `data` wholesale truncates it instead, which would hand a
        surviving point its neighbour's id - and a mismatched pair fits silently.
        Nothing moves during a deletion, so matching on position survives both.

        Args:
            name: Volume name.
            layer: Its Points layer, whose features may carry existing ids.
            picked: The rows' patient coordinates, in layer order.

        Returns:
            An int array of ids, one per row. New rows take fresh ids from the
            session counter, so a deleted id is never reissued.
        """
        known = {
            mark.id: mark.as_array() for mark in self.session.for_volume(name)
        }
        declared = _feature(layer, "id")
        ids = np.zeros(len(picked), dtype=int)
        claimed = set()

        # 1. A row sitting exactly where a landmark was is that landmark.
        for row, point in enumerate(picked):
            for identifier, stored in known.items():
                if identifier in claimed:
                    continue
                if np.allclose(point, stored, atol=MATCH_TOLERANCE_MM, rtol=0.0):
                    ids[row] = identifier
                    claimed.add(identifier)
                    break

        # 2. A row that has moved keeps the id the layer still declares for it,
        #    which is right for a drag: no resize happens, so nothing has shifted.
        for row in range(len(picked)):
            if ids[row] or declared is None or row >= len(declared):
                continue
            candidate = int(declared[row])
            if candidate > 0 and candidate not in claimed:
                ids[row] = candidate
                claimed.add(candidate)

        # 3. Anything left is newly picked and gets an id of its own. Pairing is a
        #    separate, deliberate act - the user retypes an id in the panel to say
        #    two picks are the same feature - rather than something a mode has to be
        #    in the right state for while picking.
        for row in range(len(picked)):
            if not ids[row]:
                # Straight off the session's counter, so an id is never reissued
                # even after the point that held it has been deleted.
                self.session.issued += 1
                ids[row] = self.session.issued
        return ids

    # --- screen <- session --------------------------------------------------

    def reposition(self, image_layer=None) -> None:
        """Redraw stored landmarks wherever the display now puts them.

        The stored patient coordinates are untouched; only their screen positions
        move.

        Args:
            image_layer: The layer that moved, or None to redraw every volume.
        """
        if image_layer is None:
            names = tuple(self._images)
        else:
            volume = getattr(image_layer, "metadata", {}).get("volume")
            if volume is None or volume.name not in self._images:
                return
            names = (volume.name,)

        for name in names:
            place = placement_for(self._images[name])
            if place is None:
                continue
            marks = self.session.for_volume(name)
            positions = np.array(
                [place.patient_to_world(mark.patient) for mark in marks], dtype=float
            ).reshape(-1, 3)
            self._write(
                self._points[name],
                positions,
                np.array([mark.id for mark in marks], dtype=int),
            )

    def use(self, session: Session) -> None:
        """Adopt a loaded session and redraw everything from it.

        Args:
            session: The session to take over from the current one. Open volumes it
                does not name are added to it.
        """
        self.session = session
        for name, image in self._images.items():
            session.volumes.setdefault(name, image.metadata["volume"].path)
            if name not in session.chain:
                session.chain = session.chain + (name,)
        self.reposition()
        self._notify()

    # --- writing ------------------------------------------------------------

    def _write(self, layer, positions, ids) -> None:
        """Push positions and ids into a layer without absorbing them straight back.

        Args:
            layer: The Points layer to write.
            positions: (N, 3) world coordinates.
            ids: (N,) landmark ids, row-aligned with `positions`.
        """
        self._busy = True
        try:
            layer.data = np.asarray(positions, dtype=float).reshape(-1, 3)
            layer.features = {"id": np.asarray(ids, dtype=int)}
        finally:
            self._busy = False


def _feature(layer, column: str):
    """Read one column of a Points layer's features table.

    Args:
        layer: A Points layer.
        column: Column name.

    Returns:
        The column as a 1-D int array, or None if the layer has no such column.
    """
    features = getattr(layer, "features", None)
    if features is None or column not in features:
        return None
    return np.asarray(features[column], dtype=int).reshape(-1)


def _try(target, attribute: str, value) -> None:
    """Set an attribute that only some napari versions have.

    These are cosmetic - out-of-slice display and the default id for a newly added
    row - so an older napari loses the polish rather than the program.

    Args:
        target: The object or mapping to set on.
        attribute: Attribute or key name.
        value: The value to set. Failures are ignored.
    """
    try:
        if isinstance(target, dict) or hasattr(target, "__setitem__"):
            target[attribute] = value
        else:
            setattr(target, attribute, value)
    except (AttributeError, KeyError, TypeError, ValueError):
        pass
