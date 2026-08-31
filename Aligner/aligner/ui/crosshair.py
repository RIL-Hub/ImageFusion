"""A crosshair drawn as a napari overlay rather than as data.

The distinction is the whole reason this works. A Shapes layer is *data*, so napari
slices it: a line at z = 12.3 mm does not exist while the slider sits at 12.0, and
making it appear means pinning it to the slice, which is a fight not worth having. An
overlay is not data. It is parented straight into the vispy scene under the same
transform as the layers and drawn with no slicing step, so it renders at any slice and
in any view. Its rays run far past the data, so they cross any view at any zoom.

What it holds is one world position, split by what is on screen. The two coordinates
*in* the plane are what a click sets and they stay put. The one *out* of the plane
follows the slider: scrub from z = 0 to z = 10 and the mark's z becomes 10, because the
mark is where you are looking, not where you last clicked.

Changing view then carries the point across. Going from z to y puts the y slider onto
the held y, so the new view opens on the marked feature and the crosshair draws at the
z and x it had; going back to z opens on z = 10, not on the z the mark was born at.
Tracking is suspended while the sliders are being moved for that purpose, or the stale
slider value would overwrite the held one on the way through.

Two napari internals are needed - the overlay-to-visual registry and the viewer's
overlay dict - and both are private. Installation is guarded end to end: if either has
moved, `Crosshair.available` is False and everything downstream carries on without a
crosshair rather than without a program.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Tuple

import numpy as np

NAME = "aligner_crosshair"

COLOR = (0.0, 0.9, 1.0, 0.85)  # cyan: unused by the landmark palette
WIDTH = 1.0

# How far the rays run. Ten metres is far beyond any specimen, so the lines always
# leave the view and read as axes rather than segments, and far short of the ranges
# where single-precision scene coordinates start to grind.
DEFAULT_REACH = 1.0e4

# Named colours the panel offers, as RGBA.
COLORS = {
    "cyan": (0.0, 0.9, 1.0, 0.85),
    "yellow": (1.0, 0.85, 0.0, 0.85),
    "magenta": (1.0, 0.2, 0.8, 0.85),
    "green": (0.2, 1.0, 0.35, 0.85),
    "white": (1.0, 1.0, 1.0, 0.85),
}


def scene_vertices(world, displayed, reach: float = DEFAULT_REACH) -> np.ndarray:
    """Build the crosshair's line segments, in vispy scene coordinates.

    napari's scene holds world coordinates in *reversed displayed* order: with
    `displayed` of (0, 2), vispy x is world axis 2 and vispy y is world axis 0.
    Getting that backwards draws a crosshair that tracks the mouse convincingly and
    sits on the wrong feature.

    Args:
        world: The marked point, (z, y, x) mm.
        displayed: `viewer.dims.displayed`, the axes currently on screen.
        reach: How far each ray runs from the point, in mm.

    Returns:
        (2N, ndisplay) endpoints, one segment per displayed axis, for a Line drawn
        with ``connect="segments"``.
    """
    world = np.asarray(world, dtype=float).reshape(-1)
    axes = tuple(int(a) for a in displayed)[::-1]
    centre = np.array([world[a] for a in axes], dtype=float)

    points = []
    for index in range(len(centre)):
        start, end = centre.copy(), centre.copy()
        start[index] -= reach
        end[index] += reach
        points.extend([start, end])
    return np.asarray(points, dtype=float)


def capture(viewer, position=None):
    """Read the world point a click marks.

    `viewer.cursor.position` already combines the two halves - in-plane from the
    mouse, depth from the slider - so this is mostly about tolerating a viewer that
    cannot answer.

    Args:
        viewer: The napari viewer.
        position: An explicit position to use instead of the cursor's.

    Returns:
        (z, y, x) mm, or None if no position is available.
    """
    if position is None:
        position = getattr(getattr(viewer, "cursor", None), "position", None)
    if position is None or len(position) < 3:
        return None
    return np.asarray(position, dtype=float).reshape(-1)[-3:]


def drawable(placed, visible, ndisplay) -> bool:
    """Decide whether the crosshair should be drawn at all.

    Every one of the three has to re-check when *any* of them changes: marking a
    point while the crosshair was switched off once left it blanked for good,
    because turning it back on restored the visibility of a node whose geometry had
    been thrown away.

    Args:
        placed: Whether a point has been marked.
        visible: Whether the user has it switched on.
        ndisplay: How many dimensions napari is displaying.

    Returns:
        True only when marked, shown, and in a 2D view - in 3D it marks a point on
        a slice and there is no slice.
    """
    try:
        return bool(placed) and bool(visible) and int(ndisplay) == 2
    except (TypeError, ValueError):
        return False


def follow(position, point, displayed) -> np.ndarray:
    """Update a marked point from the sliders, keeping what the click set.

    The whole rule, in one place: the axes on screen are where you clicked; the axis
    on the slider is wherever you have scrubbed to.

    Args:
        position: The marked point, (z, y, x) mm.
        point: `viewer.dims.point`, the current slider positions in world mm.
        displayed: `viewer.dims.displayed`, the axes currently on screen.

    Returns:
        The updated point, (z, y, x) mm.
    """
    updated = list(np.asarray(position, dtype=float).reshape(-1))
    on_screen = {int(a) for a in displayed}
    point = list(np.asarray(point, dtype=float).reshape(-1))
    for axis in range(len(updated)):
        if axis not in on_screen and axis < len(point):
            updated[axis] = point[axis]
    return np.asarray(updated, dtype=float)


def look_at(viewer, world) -> bool:
    """Centre the camera on a world position, leaving the zoom alone.

    napari reads the *last* ``ndisplay`` entries of ``camera.center`` and reverses
    them for vispy, so those entries are the displayed axes' world coordinates in
    display order. Writing it is well defined; reading it back and guessing what the
    entries meant is what went wrong the first time this was attempted.

    Zoom is untouched on purpose: every axis here is millimetres, so a millimetre is
    the same size whichever pair of axes is on screen.

    Args:
        viewer: The napari viewer.
        world: (z, y, x) mm to centre on, or None to do nothing.

    Returns:
        True if the camera moved.
    """
    if world is None:
        return False
    try:
        displayed = tuple(int(a) for a in viewer.dims.displayed)
        world = np.asarray(world, dtype=float).reshape(-1)
        centre = [float(world[a]) for a in displayed if 0 <= a < len(world)]
        while len(centre) < 3:
            centre.insert(0, 0.0)
        viewer.camera.center = tuple(centre[-3:])
    except (AttributeError, IndexError, TypeError, ValueError):
        return False
    return True


@contextmanager
def steady_zoom(viewer):
    """Hold the zoom steady across a block that changes the axis order.

    napari connects ``dims.events.order`` straight to ``fit_to_view``, so simply not
    calling ``reset_view`` does not preserve the zoom - changing which axes are on
    screen refits the camera on its own. The zoom is noted beforehand and restored
    afterwards, in a `finally`, so an exception mid-block does not leave the view at
    whatever the refit chose.

    Args:
        viewer: The napari viewer.
    """
    try:
        zoom = float(viewer.camera.zoom)
    except (AttributeError, TypeError, ValueError):
        zoom = None
    try:
        yield
    finally:
        if zoom is not None:
            try:
                viewer.camera.zoom = zoom
            except (AttributeError, TypeError, ValueError):
                pass


def overlay_store(viewer):
    """Find where scene overlays have to be installed.

    napari 0.9 moved them off the viewer: `viewer._overlays` became
    `viewer.scene.overlays`, an evented dict that still takes arbitrary keys. Both
    spellings are tried, so this works either side of that change.

    Args:
        viewer: The napari viewer.

    Returns:
        The overlay store, or None if neither spelling exists.
    """
    store = getattr(getattr(viewer, "scene", None), "overlays", None)
    if store is not None:
        return store
    return getattr(viewer, "_overlays", None)


def recall(viewer, world) -> bool:
    """Put every slider back on a stored point, so a new view lands on it.

    Args:
        viewer: The napari viewer.
        world: (z, y, x) mm to move to, or None to do nothing.

    Returns:
        True if the sliders moved.
    """
    if world is None:
        return False
    try:
        for axis, value in enumerate(np.asarray(world, dtype=float).reshape(-1)):
            viewer.dims.set_point(axis, float(value))
    except (AttributeError, IndexError, TypeError, ValueError):
        return False
    return True


# --- the napari overlay ---------------------------------------------------

try:
    from napari._vispy.overlays.base import ViewerOverlayMixin, VispySceneOverlay
    from napari._vispy.utils.visual import overlay_to_visual
    from napari.components.overlays.base import SceneOverlay
    from vispy.scene.visuals import Line

    class CrosshairOverlay(SceneOverlay):
        """One world position, plus how it should be drawn."""

        position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        placed: bool = False
        color: Tuple[float, float, float, float] = COLOR
        width: float = WIDTH
        reach: float = DEFAULT_REACH

    class VispyCrosshair(ViewerOverlayMixin, VispySceneOverlay):
        """Draws the overlay, and redraws it whenever the view changes."""

        def __init__(self, **kwargs):
            # Forwarded rather than named: napari builds these itself and has added
            # arguments between versions (font_info in 0.9). Passing them through
            # untouched means a new one does not break us.
            super().__init__(
                node=Line(connect="segments", color=COLOR, width=WIDTH, method="gl"),
                **kwargs,
            )
            # `visible` included: napari's own handler only flips the node's
            # visibility, which is not enough when hiding also blanks the geometry.
            for field in ("position", "placed", "visible", "color", "width", "reach"):
                event = getattr(self.overlay.events, field, None)
                if event is not None:
                    event.connect(self._redraw)
            # The scene axes change meaning when the view does, so the geometry has
            # to be rebuilt rather than merely re-transformed.
            dims = getattr(getattr(self, "viewer", None), "dims", None)
            if dims is not None:
                dims.events.order.connect(self._redraw)
                dims.events.ndisplay.connect(self._redraw)
            self._redraw()

        def _hide(self, ndisplay: int = 2) -> None:
            """Blank the geometry as well as the visibility.

            Leaving ten metres of line parented in the scene while it is merely
            invisible is what makes the 3D camera unhappy. A zero-length segment
            cannot - and unlike an empty array, every vispy version will draw it.
            """
            self.node.set_data(pos=np.zeros((2, max(int(ndisplay), 2)), dtype=float))
            self.node.visible = False

        def _redraw(self, event=None) -> None:
            dims = getattr(getattr(self, "viewer", None), "dims", None)
            if dims is None:
                self._hide()
                return

            # Two dimensions only. In a 3D view the crosshair means nothing - it
            # marks a point on a slice, and there is no slice - and rays this long
            # wreck the perspective camera's depth range.
            ndisplay = int(getattr(dims, "ndisplay", 2))
            if not drawable(self.overlay.placed, self.overlay.visible, ndisplay):
                self._hide(ndisplay)
                return

            self.node.set_data(
                pos=scene_vertices(
                    self.overlay.position, dims.displayed, self.overlay.reach
                ),
                color=self.overlay.color,
                width=self.overlay.width,
            )
            self.node.visible = True

    AVAILABLE = True
    UNAVAILABLE_BECAUSE = ""
except Exception as exc:  # napari moved its overlay machinery
    AVAILABLE = False
    # Kept, not swallowed. "napari moved its overlay machinery" is useless on its own;
    # which import or which field is what says whether it is a one-line fix.
    UNAVAILABLE_BECAUSE = f"{type(exc).__name__}: {exc}"


class Crosshair:
    """Installs the overlay and drives it. A no-op if napari has moved.

    Attributes:
        reason: Why the overlay could not be installed, or "" if it could.
    """

    def __init__(self, viewer):
        self._viewer = viewer
        self._overlay = None
        self._frozen = False
        self.reason = UNAVAILABLE_BECAUSE
        if not AVAILABLE:
            return
        try:
            store = overlay_store(viewer)
            if store is None:
                raise AttributeError("no scene.overlays or _overlays on this viewer")
            overlay_to_visual[CrosshairOverlay] = VispyCrosshair
            self._overlay = CrosshairOverlay(visible=True)
            store[NAME] = self._overlay
        except Exception as exc:
            self._overlay = None
            self.reason = f"{type(exc).__name__}: {exc}"
            return

        # Deliberately not connected to the order event: changing view swaps which
        # axis is out of plane, and reading the slider *then* would take the stale
        # value before `recall` has moved it onto the held one.
        for name in ("current_step", "point"):
            event = getattr(viewer.dims.events, name, None)
            if event is not None:
                event.connect(self._follow)

    # --- following the slider -----------------------------------------------

    @contextmanager
    def held(self):
        """Suspend slider-following, for while the sliders are being set."""
        previous, self._frozen = self._frozen, True
        try:
            yield
        finally:
            self._frozen = previous

    def _follow(self, event=None) -> None:
        """Take the out-of-plane coordinate from the sliders, on a slider change.

        Args:
            event: The napari event, ignored.
        """
        if self._frozen or self._overlay is None or not self._overlay.placed:
            return
        try:
            displayed = self._viewer.dims.displayed
            point = self._viewer.dims.point
        except (AttributeError, TypeError, ValueError):
            return
        updated = follow(self._overlay.position, point, displayed)
        if not np.allclose(updated, self._overlay.position):
            self._overlay.position = tuple(float(v) for v in updated)

    @property
    def available(self) -> bool:
        """Test whether the overlay installed.

        Returns:
            False if napari has moved its overlay machinery; `reason` says how.
        """
        return self._overlay is not None

    @property
    def position(self):
        """Read the marked point.

        Returns:
            (z, y, x) mm, or None if nothing has been marked.
        """
        if self._overlay is None or not self._overlay.placed:
            return None
        return np.asarray(self._overlay.position, dtype=float)

    def get(self, field, default=None):
        """Read one overlay field.

        Args:
            field: Overlay field name.
            default: Returned if there is no overlay or no such field.

        Returns:
            The field's value, or `default`.
        """
        return getattr(self._overlay, field, default) if self._overlay else default

    def set(self, field, value) -> None:
        """Set one drawing parameter.

        Args:
            field: Overlay field name.
            value: Its new value. Ignored if there is no overlay.
        """
        if self._overlay is not None:
            try:
                setattr(self._overlay, field, value)
            except (AttributeError, TypeError, ValueError):
                pass

    def connect(self, callback) -> None:
        """Subscribe to changes of the marked point.

        Args:
            callback: Called with no arguments whenever the point moves or is
                placed or cleared.
        """
        if self._overlay is not None:
            try:
                self._overlay.events.position.connect(lambda *_: callback())
                self._overlay.events.placed.connect(lambda *_: callback())
            except AttributeError:
                pass

    def move_to(self, world) -> bool:
        """Mark a world position.

        Args:
            world: (z, y, x) mm, or None to take the cursor's position.

        Returns:
            True if a point was marked.
        """
        world = capture(self._viewer, world)
        if world is None or self._overlay is None:
            return False
        self._overlay.position = tuple(float(v) for v in world)
        self._overlay.placed = True
        return True

    def mark_here(self, *args) -> bool:
        """Mark wherever the pointer is.

        Args:
            *args: Ignored. This is wired to both a key binding and a mouse
                callback, which call back differently.

        Returns:
            True if a point was marked.
        """
        return self.move_to(None)

    def recall(self) -> bool:
        """Move the sliders onto the marked point, for after a change of view.

        Returns:
            True if the sliders moved.
        """
        with self.held():
            return recall(self._viewer, self.position)

    def look(self) -> bool:
        """Centre the camera on the marked point, at whatever zoom is set.

        Returns:
            True if the camera moved.
        """
        return look_at(self._viewer, self.position)

    def clear(self) -> None:
        """Unset the marked point, leaving the appearance settings alone."""
        if self._overlay is not None:
            self._overlay.placed = False
