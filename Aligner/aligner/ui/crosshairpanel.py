"""Controls for the crosshair, and a readout of where it is.

The crosshair is an overlay rather than a layer, which is what lets it draw at any
slice - but it also means it has no entry in the layer list and no layer controls. This
is that missing panel.

The position boxes are editable both ways: they show where a Shift+click put the
crosshair, and typing into them moves it. So a coordinate read off one volume can be
typed in to find the same place again.
"""

from __future__ import annotations

import numpy as np
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .crosshair import COLORS
from .widgets import form, group, note

# The panel reads z, y, x top to bottom, which is the world order.
AXES = (("z", 0), ("y", 1), ("x", 2))


class CrosshairPanel(QWidget):
    """Drives a `Crosshair`. Explains itself when there isn't one."""

    def __init__(self, crosshair=None, viewer=None):
        super().__init__()
        self._crosshair = crosshair
        self._viewer = viewer
        self._busy = False
        self._boxes = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        if crosshair is None or not crosshair.available:
            reason = getattr(crosshair, "reason", "") if crosshair else ""
            layout.addWidget(note("No crosshair on this napari version."))
            if reason:
                layout.addWidget(note(reason))
            layout.addStretch(1)
            return

        layout.addWidget(self._position_group())
        layout.addWidget(self._style_group())
        layout.addStretch(1)

        crosshair.connect(self.refresh)
        self.refresh()

    # --- construction -------------------------------------------------------

    def _position_group(self) -> QGroupBox:
        """Build the position readout, which is also how a position is typed in.

        Returns:
            The group box.
        """
        box, inside = group("Crosshair")
        inside.addWidget(note("Shift+click, or T, to mark a point."))

        fields = form()
        for name, axis in AXES:
            spin = QDoubleSpinBox()
            spin.setRange(-1e6, 1e6)
            spin.setDecimals(3)
            spin.setSuffix(" mm")
            spin.setKeyboardTracking(False)
            spin.valueChanged.connect(lambda *_: self._typed())
            self._boxes[axis] = spin
            fields.addRow(QLabel(name), spin)
        inside.addLayout(fields)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        for text, slot in (("go to", self._recall), ("clear", self._clear)):
            button = QPushButton(text)
            button.clicked.connect(slot)
            row.addWidget(button)
        holder = QWidget()
        holder.setLayout(row)
        inside.addWidget(holder)

        self._state = note("")
        inside.addWidget(self._state)
        return box

    def _style_group(self) -> QGroupBox:
        """Build the appearance controls.

        Returns:
            The group box.
        """
        box, inside = group("Appearance")
        fields = form()

        self._visible = QCheckBox()
        self._visible.setChecked(True)
        self._visible.stateChanged.connect(
            lambda *_: self._push("visible", self._visible.isChecked())
        )
        fields.addRow(QLabel("show"), self._visible)

        self._color = QComboBox()
        self._color.addItems(list(COLORS))
        self._color.currentTextChanged.connect(
            lambda name: self._push("color", COLORS.get(name, COLORS["cyan"]))
        )
        fields.addRow(QLabel("colour"), self._color)

        self._width = QDoubleSpinBox()
        self._width.setRange(0.5, 10.0)
        self._width.setSingleStep(0.5)
        self._width.setDecimals(1)
        self._width.setKeyboardTracking(False)
        self._width.valueChanged.connect(lambda v: self._push("width", float(v)))
        fields.addRow(QLabel("width"), self._width)

        self._reach = QDoubleSpinBox()
        self._reach.setRange(1.0, 1e6)
        self._reach.setDecimals(0)
        self._reach.setSuffix(" mm")
        self._reach.setKeyboardTracking(False)
        self._reach.setToolTip(
            "How far the lines run from the marked point. Long enough to leave the "
            "view is the point; much longer costs scene precision."
        )
        self._reach.valueChanged.connect(lambda v: self._push("reach", float(v)))
        fields.addRow(QLabel("length"), self._reach)

        inside.addLayout(fields)
        return box

    # --- crosshair -> panel -------------------------------------------------

    def refresh(self) -> None:
        """Redisplay the crosshair's position and appearance."""
        crosshair = self._crosshair
        position = crosshair.position

        self._busy = True
        try:
            if position is not None:
                for _, axis in AXES:
                    self._boxes[axis].setValue(float(position[axis]))
            self._visible.setChecked(bool(crosshair.get("visible", True)))
            self._width.setValue(float(crosshair.get("width", 1.0)))
            self._reach.setValue(float(crosshair.get("reach", 1e4)))

            # Length-checked before comparing: np.allclose raises on mismatched
            # shapes, and an overlay that reports no colour would take the panel
            # down with it.
            current = tuple(crosshair.get("color", ()) or ())
            if len(current) == 4:
                for name, value in COLORS.items():
                    if np.allclose(value, current, atol=1e-6):
                        self._color.setCurrentText(name)
                        break
        finally:
            self._busy = False

        self._state.setText(
            "not placed yet" if position is None else "marking one point"
        )

    # --- panel -> crosshair -------------------------------------------------

    def _push(self, field: str, value) -> None:
        """Set one drawing parameter on the crosshair.

        Args:
            field: Overlay field name.
            value: Its new value. Ignored while the panel is refreshing itself.
        """
        if not self._busy:
            self._crosshair.set(field, value)

    def _typed(self) -> None:
        """Typing a coordinate moves the crosshair there."""
        if self._busy:
            return
        world = [self._boxes[axis].value() for _, axis in AXES]
        self._crosshair.move_to(world)

    def _recall(self) -> None:
        """Go to the marked point without changing view: sliders, then camera."""
        self._crosshair.recall()
        self._crosshair.look()

    def _clear(self) -> None:
        """Unset the marked point, leaving the appearance settings alone."""
        self._crosshair.clear()
        self.refresh()
