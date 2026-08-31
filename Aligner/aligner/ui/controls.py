"""Orientation controls for the selected layer.

There is one piece of rotation state: a 3x3 matrix mapping source physical
directions to display physical directions. Every rotation control multiplies a
delta onto it from the left, which is what makes them all *relative to what is
currently displayed*.

Coarse and fine are not different mechanisms - the distinction that caused most of
this panel's history. Both change the matrix, and `ObliqueView` samples whichever
planes the viewer asks for. A quarter turn simply happens to land on exact grid
coordinates, so it costs nothing in accuracy. Nothing is resampled as a whole.

Translation is separate and independent, a pure affine offset with its own reset.
It is stored in mm and merely *displayed* in whichever unit is selected.
"""

from __future__ import annotations

import numpy as np
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..geometry import centre_world
from ..oblique import ObliqueView, axis_rotation, flip, quarter_turn

# The panel reads x, y, z top to bottom; internally the world order is (z, y, x).
AXES = (("x", 2), ("y", 1), ("z", 0))

# Matrix columns are the data axes, labelled the way Amide does: i is the column
# index (x), j the row index (y), k the slice index (z).
DATA_AXES = (("i", 2), ("j", 1), ("k", 0))

# Left to right. Clockwise is positive in our convention, so a quarter turn
# clockwise and +90 degrees are the same matrix.
COARSE_STEPS = (
    ("90° ↻", lambda axis: quarter_turn(axis, clockwise=True)),
    ("15° ↻", lambda axis: axis_rotation(axis, 15.0)),
    ("invert", lambda axis: flip(axis)),
    ("15° ↺", lambda axis: axis_rotation(axis, -15.0)),
    ("90° ↺", lambda axis: quarter_turn(axis, clockwise=False)),
)

UNITS = ("mm", "voxels")


class DeltaSlider(QSlider):
    """A slider that reports only once the user has let go.

    Neither available signal distinguishes "still adjusting" from "done".
    valueChanged fires continuously while dragging, and a groove click emits it
    with isSliderDown() still False, because that flag is only set when the
    *handle* is grabbed - so a plain click would commit mid-gesture. Committing on
    the mouse or key release covers drags, groove clicks and arrow keys alike.
    """

    committed = Signal(int)

    def mouseReleaseEvent(self, event):
        """Commit on a mouse release, covering drags and groove clicks alike.

        Args:
            event: The Qt mouse event.
        """
        super().mouseReleaseEvent(event)
        self.committed.emit(self.value())

    def keyReleaseEvent(self, event):
        """Commit on a key release, covering the arrow keys.

        Args:
            event: The Qt key event.
        """
        super().keyReleaseEvent(event)
        self.committed.emit(self.value())

    def wheelEvent(self, event):
        """Ignore the wheel, so scrolling the panel cannot rotate a volume.

        Args:
            event: The Qt wheel event, passed back unhandled.
        """
        event.ignore()


def _group(title: str) -> tuple:
    """Build a titled frame.

    Args:
        title: The frame's heading.

    Returns:
        (box, layout): the group box, and the vertical layout to fill it with.
    """
    box = QGroupBox(title)
    box.setStyleSheet("QGroupBox { font-weight: bold; margin-top: 8px; }")
    layout = QVBoxLayout(box)
    layout.setContentsMargins(8, 8, 8, 8)
    layout.setSpacing(4)
    return box, layout


def _subsection(text: str) -> QLabel:
    """Build a greyed subheading.

    Args:
        text: The heading.

    Returns:
        A QLabel.
    """
    label = QLabel(text)
    label.setStyleSheet("color: gray; margin-top: 4px;")
    return label


def _form() -> QFormLayout:
    """Build a label-and-field form layout.

    Returns:
        A QFormLayout whose fields grow with the panel.
    """
    layout = QFormLayout()
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(4)
    layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
    return layout


class OrientationControls(QWidget):
    """Acts on whichever layer is selected in napari's layer list."""

    # Emitted after a layer's display has moved, with that layer. Landmarks are
    # stored in the data's frame, so anything drawing them has to redraw. Signalled
    # rather than called directly, so this panel stays ignorant of the picker.
    layer_changed = Signal(object)

    def __init__(self, viewer):
        super().__init__()
        self._viewer = viewer
        self._busy = False
        self._sliders = {}
        self._angle_boxes = {}
        self._offset_boxes = {}
        self._matrix_cells = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        rotation, inside = _group("Rotation Controls")
        inside.addLayout(self._matrix_grid())

        reset_rotation = QPushButton("reset")
        reset_rotation.clicked.connect(self._reset_rotation)
        inside.addWidget(reset_rotation)

        inside.addWidget(_subsection("Fine"))
        inside.addLayout(self._fine_form())
        inside.addWidget(_subsection("Coarse"))
        inside.addLayout(self._coarse_grid())
        layout.addWidget(rotation)

        translation, inside = _group("Translation Controls")
        inside.addLayout(self._units_row())
        inside.addLayout(self._translation_form())

        reset_translation = QPushButton("reset")
        reset_translation.clicked.connect(self._reset_translation)
        inside.addWidget(reset_translation)
        layout.addWidget(translation)

        layout.addStretch(1)

        viewer.layers.selection.events.changed.connect(self._sync_from_layer)
        # Which axis is on the slider decides how the lazy data must be chunked.
        viewer.dims.events.order.connect(self._view_changed)

    # --- construction -----------------------------------------------------

    def _matrix_grid(self) -> QGridLayout:
        """Build the orientation readout: where each data axis points on screen.

        Returns:
            The grid layout, rows for display axes and columns for data axes.
        """
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(2)
        grid.setColumnStretch(0, 0)

        for column, (name, _) in enumerate(DATA_AXES, start=1):
            header = QLabel(name)
            header.setAlignment(Qt.AlignCenter)
            header.setStyleSheet("color: gray;")
            grid.addWidget(header, 0, column)
            grid.setColumnStretch(column, 1)

        for row, (name, world_axis) in enumerate(AXES, start=1):
            label = QLabel(name)
            label.setStyleSheet("color: gray;")
            grid.addWidget(label, row, 0)
            for column, (_, data_axis) in enumerate(DATA_AXES, start=1):
                cell = QLabel("-")
                cell.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
                cell.setStyleSheet("font-family: monospace;")
                self._matrix_cells[(world_axis, data_axis)] = cell
                grid.addWidget(cell, row, column)
        return grid

    def _fine_form(self) -> QFormLayout:
        """Build the per-axis rotation sliders and their editable angle boxes.

        Returns:
            The form layout.
        """
        form = _form()
        for name, axis in AXES:
            slider = DeltaSlider(Qt.Orientation.Horizontal)
            slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            slider.setRange(-180, 180)
            slider.setSingleStep(1)
            slider.setPageStep(15)

            # Editable, so an exact angle can be typed rather than dragged for.
            # keyboardTracking off means typing commits on Enter or focus-out.
            box = QSpinBox()
            box.setRange(-180, 180)
            box.setSuffix("°")
            box.setKeyboardTracking(False)
            box.setAlignment(Qt.AlignRight)
            box.setFixedWidth(70)

            slider.valueChanged.connect(lambda v, a=axis: self._show_pending(a, v))
            slider.committed.connect(lambda _v, a=axis: self._commit_slider(a))
            box.valueChanged.connect(lambda v, a=axis: self._commit_typed(a, v))

            self._sliders[axis] = slider
            self._angle_boxes[axis] = box

            row = QWidget()
            line = QHBoxLayout(row)
            line.setContentsMargins(0, 0, 0, 0)
            line.setSpacing(4)
            line.addWidget(slider, 1)
            line.addWidget(box, 0)
            form.addRow(QLabel(name), row)
        return form

    def _coarse_grid(self) -> QGridLayout:
        """Build the quarter-turn, 15 degree and invert buttons.

        Returns:
            The grid layout, one row per display axis.
        """
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(2)
        grid.setColumnStretch(0, 0)
        for column in range(1, len(COARSE_STEPS) + 1):
            grid.setColumnStretch(column, 1)

        for row, (name, axis) in enumerate(AXES):
            grid.addWidget(QLabel(name), row, 0)
            for column, (text, delta) in enumerate(COARSE_STEPS, start=1):
                button = QPushButton(text)
                button.setMinimumWidth(1)  # let the grid decide, not the label
                button.clicked.connect(
                    lambda _=False, d=delta, a=axis: self._apply(d(a))
                )
                grid.addWidget(button, row, column)
        return grid

    def _units_row(self) -> QHBoxLayout:
        """Build the mm-or-voxels chooser for the translation boxes.

        Returns:
            The row layout.
        """
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(QLabel("units"))
        self._units = QComboBox()
        self._units.addItems(UNITS)
        self._units.currentTextChanged.connect(lambda *_: self._show_offsets())
        row.addWidget(self._units, 1)
        return row

    def _translation_form(self) -> QFormLayout:
        """Build the per-axis offset boxes and their resets.

        Returns:
            The form layout.
        """
        form = _form()
        for name, axis in AXES:
            box = QDoubleSpinBox()
            box.setRange(-1e5, 1e5)
            box.setSingleStep(1.0)
            box.setDecimals(3)
            box.setKeyboardTracking(False)
            box.valueChanged.connect(lambda *_: self._commit_offsets())
            self._offset_boxes[axis] = box

            reset = QPushButton("reset")
            reset.setFixedWidth(56)
            reset.clicked.connect(lambda _=False, a=axis: self._reset_offset(a))

            row = QWidget()
            line = QHBoxLayout(row)
            line.setContentsMargins(0, 0, 0, 0)
            line.setSpacing(4)
            line.addWidget(box, 1)
            line.addWidget(reset, 0)
            form.addRow(QLabel(name), row)
        return form

    # --- state ------------------------------------------------------------

    def _selected(self):
        """Find the layer these controls act on.

        Returns:
            The selected layer if it holds a volume, else None - so selecting
            a landmarks layer disables the controls rather than misapplying
            them.
        """
        layer = self._viewer.layers.selection.active
        if layer is None or "volume" not in getattr(layer, "metadata", {}):
            return None
        return layer

    def _display_spacing(self, layer) -> np.ndarray:
        """Read the voxel size of the *displayed* grid, which orientation permutes.

        Args:
            layer: The layer, or None.

        Returns:
            (z, y, x) display voxel size in mm, or ones if there is no view yet.
        """
        view = layer.metadata.get("view") if layer is not None else None
        if view is None:
            return np.ones(3)
        return np.asarray(view.spacing, dtype=float)

    # --- rotation ---------------------------------------------------------

    def _apply(self, delta) -> None:
        """Compose a rotation onto the selected layer's orientation.

        Args:
            delta: 3x3 rotation in the *display* frame. It multiplies from the
                left, which is what makes every control relative to what is on
                screen rather than to the data's original axes.
        """
        layer = self._selected()
        if layer is None:
            return
        current = layer.metadata["matrix"]
        layer.metadata["matrix"] = np.asarray(delta, dtype=float) @ current
        self._refresh(layer)

    def _show_pending(self, axis: int, degrees: int) -> None:
        """Mirror a drag into its angle box. Nothing is applied until release.

        Args:
            axis: World axis being dragged.
            degrees: The slider's current value.
        """
        if self._busy:
            return
        self._busy = True
        try:
            self._angle_boxes[axis].setValue(int(degrees))
        finally:
            self._busy = False

    def _commit_slider(self, axis: int) -> None:
        """Apply a slider's angle once the user lets go.

        Args:
            axis: World axis the slider rotates about.
        """
        if not self._busy:
            self._commit_angle(axis, float(self._sliders[axis].value()))

    def _commit_typed(self, axis: int, degrees: int) -> None:
        """Apply an angle typed into a box.

        Args:
            axis: World axis to rotate about.
            degrees: The angle typed.
        """
        if not self._busy:
            self._commit_angle(axis, float(degrees))

    def _commit_angle(self, axis: int, degrees: float) -> None:
        """Rotate by an angle, then return the control to zero.

        The controls are relative, so they read zero at rest and the
        orientation matrix carries the accumulated state.

        Args:
            axis: World axis to rotate about.
            degrees: The angle to apply.
        """
        self._zero_angle(axis)
        if abs(degrees) >= 1e-9:
            self._apply(axis_rotation(axis, degrees))

    def _zero_angle(self, axis: int) -> None:
        """Return one axis's slider and box to zero without applying anything.

        Args:
            axis: World axis.
        """
        self._busy = True
        try:
            self._sliders[axis].setValue(0)
            self._angle_boxes[axis].setValue(0)
        finally:
            self._busy = False

    def _reset_rotation(self, *_) -> None:
        """Orientation only. Translation has its own reset."""
        layer = self._selected()
        for axis in (0, 1, 2):
            self._zero_angle(axis)
        if layer is None:
            return
        layer.metadata["matrix"] = np.eye(3)
        self._refresh(layer)

    # --- translation ------------------------------------------------------

    def _commit_offsets(self, *_) -> None:
        """Apply the three offset boxes to the selected layer, converting to mm."""
        if self._busy:
            return
        layer = self._selected()
        if layer is None:
            return

        shown = np.array(
            [self._offset_boxes[axis].value() for axis in (0, 1, 2)], dtype=float
        )
        if self._units.currentText() == "voxels":
            shown = shown * self._display_spacing(layer)
        layer.metadata["nudge"] = shown
        self._refresh(layer)

    def _reset_offset(self, axis: int) -> None:
        """Zero one axis's offset.

        Args:
            axis: World axis.
        """
        self._busy = True
        try:
            self._offset_boxes[axis].setValue(0.0)
        finally:
            self._busy = False
        self._commit_offsets()

    def _reset_translation(self, *_) -> None:
        """Zero every offset, leaving the orientation alone."""
        self._busy = True
        try:
            for axis in (0, 1, 2):
                self._offset_boxes[axis].setValue(0.0)
        finally:
            self._busy = False
        self._commit_offsets()

    def _show_offsets(self) -> None:
        """Redisplay the stored millimetre offset in the selected unit."""
        layer = self._selected()
        nudge = (
            np.zeros(3) if layer is None else np.asarray(layer.metadata["nudge"])
        )
        voxels = self._units.currentText() == "voxels"
        shown = nudge / self._display_spacing(layer) if voxels else nudge

        self._busy = True
        try:
            for axis in (0, 1, 2):
                self._offset_boxes[axis].setDecimals(2 if voxels else 3)
                self._offset_boxes[axis].setValue(float(shown[axis]))
        finally:
            self._busy = False

    # --- placing a layer outright -----------------------------------------

    def set_placement(self, layer, matrix, nudge) -> None:
        """Place a layer as a solved transform says it should sit.

        Display only, exactly like every other control here: it writes the same
        two pieces of metadata the buttons and sliders write, and neither the
        data nor the DICOM geometry is touched.

        Args:
            layer: The image layer to place; ignored if it holds no volume.
            matrix: 3x3 orientation, source physical to display physical.
            nudge: (z, y, x) mm offset for the display affine.
        """
        if layer is None or "volume" not in getattr(layer, "metadata", {}):
            return
        layer.metadata["matrix"] = np.asarray(matrix, dtype=float)
        layer.metadata["nudge"] = np.asarray(nudge, dtype=float)
        self._refresh(layer)
        if layer is self._selected():
            self._sync_from_layer()

    # --- rendering --------------------------------------------------------

    def _scrub_axis(self) -> int:
        """Find the axis napari has on the slider.

        Returns:
            The first axis of `dims.order`, since napari displays the *last* two
            and puts sliders on the rest. 0 if the order cannot be read.
        """
        try:
            order = tuple(int(a) for a in self._viewer.dims.order)
        except (AttributeError, TypeError, ValueError):
            return 0
        return order[0] if len(order) == 3 else 0

    def _view_changed(self, *_) -> None:
        """Re-chunk every re-oriented volume when the slice axis changes.

        Only the chunking changes - the same planes, sampled perpendicular to the
        new slider axis instead of the old one. Volumes still showing their original
        array are plain numpy and slice fast along every axis, so they are left be.
        """
        axis = self._scrub_axis()
        for layer in self._viewer.layers:
            view = (getattr(layer, "metadata", None) or {}).get("view")
            if view is not None:
                layer.data = view.as_dask(axis)

    def _refresh(self, layer) -> None:
        """Resample and re-place one layer from its orientation and nudge.

        Args:
            layer: The image layer to redraw.
        """
        volume = layer.metadata["volume"]
        geometry = volume.geometry

        view = ObliqueView(
            source=volume.data,
            source_spacing=geometry.spacing,
            matrix=layer.metadata["matrix"],
            fill=(volume.value_range or (0.0, 0.0))[0],
        )
        # Kept for coordinate recovery: a landmark picked on screen must be
        # resolved through the view, never from the displayed position.
        layer.metadata["view"] = view

        world_centre = centre_world(layer.metadata["base_affine"], volume.shape)
        layer.data = view.as_dask(self._scrub_axis())
        layer.affine = view.display_affine(world_centre, layer.metadata["nudge"])
        self._show_matrix(layer)
        # After the affine, so listeners read the placement the volume now has.
        self.layer_changed.emit(layer)

    def _sync_from_layer(self, *_) -> None:
        """Redisplay the controls for whichever layer is now selected."""
        layer = self._selected()
        for axis in (0, 1, 2):
            self._zero_angle(axis)
        self._show_offsets()
        self._show_matrix(layer)

    def _show_matrix(self, layer) -> None:
        """Redisplay the orientation readout.

        Args:
            layer: The layer to read, or None to blank the grid.
        """
        if layer is None:
            for cell in self._matrix_cells.values():
                cell.setText("-")
            return
        matrix = layer.metadata["matrix"]
        for (world_axis, data_axis), cell in self._matrix_cells.items():
            value = matrix[world_axis, data_axis]
            cell.setText(f"{0.0 if abs(value) < 5e-4 else value:.3f}")
