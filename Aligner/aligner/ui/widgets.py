"""Small Qt builders the panels share, so they look the same and stay short."""

from __future__ import annotations

from qtpy.QtWidgets import QFormLayout, QGroupBox, QLabel, QVBoxLayout


def group(title: str) -> tuple:
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


def note(text: str = "") -> QLabel:
    """Build a greyed explanatory or status label.

    Args:
        text: The note.

    Returns:
        A word-wrapped QLabel.
    """
    label = QLabel(text)
    label.setStyleSheet("color: gray;")
    label.setWordWrap(True)
    return label


def subsection(text: str) -> QLabel:
    """Build a greyed subheading inside a group.

    Args:
        text: The heading.

    Returns:
        A QLabel.
    """
    label = QLabel(text)
    label.setStyleSheet("color: gray; margin-top: 4px;")
    return label


def form() -> QFormLayout:
    """Build a label-and-field form layout.

    Returns:
        A QFormLayout whose fields grow with the panel.
    """
    layout = QFormLayout()
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(4)
    layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
    return layout
