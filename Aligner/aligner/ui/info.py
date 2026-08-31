"""Read-only description of the selected layer."""

from __future__ import annotations

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QLabel, QVBoxLayout, QWidget


def _fmt(values, spec="g") -> str:
    """Format a sequence of numbers for display.

    Args:
        values: The numbers.
        spec: A format spec applied to each.

    Returns:
        Them joined by " x ".
    """
    return " x ".join(format(float(v), spec) for v in values)


class LayerInfo(QWidget):
    """Everything worth knowing about the selected volume, in one place."""

    FIELDS = (
        "name",
        "modality",
        "units",
        "grid",
        "voxel",
        "extent",
        "values",
        "resolution",
        "memory",
        "source",
    )

    def __init__(self, viewer):
        super().__init__()
        self._viewer = viewer

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        self._text = QLabel("select a volume")
        self._text.setWordWrap(True)
        self._text.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._text.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        # Monospace so the label column lines up without a table.
        self._text.setStyleSheet("font-family: monospace;")
        layout.addWidget(self._text)
        layout.addStretch(1)

        viewer.layers.selection.events.changed.connect(self.refresh)
        viewer.layers.events.inserted.connect(self.refresh)
        viewer.layers.events.removed.connect(self.refresh)

    def refresh(self, *_) -> None:
        """Redescribe whichever layer is selected, or say nothing is."""
        layer = self._viewer.layers.selection.active
        if layer is None or "volume" not in getattr(layer, "metadata", {}):
            self._text.setText("select a volume")
            return

        volume = layer.metadata["volume"]
        geometry = volume.geometry
        spacing = (
            geometry.slice_spacing,
            geometry.pixel_spacing[0],
            geometry.pixel_spacing[1],
        )
        extent = tuple(n * s for n, s in zip(volume.shape, spacing))
        low, high = volume.value_range or (float("nan"), float("nan"))
        resolution = (
            "full"
            if volume.decimation == 1
            else f"{volume.decimation}x decimated from {_fmt(volume.full_shape, '.0f')}"
        )

        values = {
            "name": volume.name,
            "modality": volume.modality,
            "units": volume.units,
            "grid": f"{_fmt(layer.data.shape, '.0f')} voxels",
            "voxel": f"{_fmt(spacing, '.5g')} mm",
            "extent": f"{_fmt(extent, '.4g')} mm",
            "values": f"{low:.4g} to {high:.4g}",
            "resolution": resolution,
            "memory": f"{np.asarray(layer.data).nbytes / 1024**2:.0f} MB",
            "source": str(volume.path),
        }
        width = max(len(name) for name in self.FIELDS)
        self._text.setText(
            "\n".join(f"{name:<{width}}  {values[name]}" for name in self.FIELDS)
        )
