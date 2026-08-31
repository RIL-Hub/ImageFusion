"""Rebuilding napari's viewer button row for this application.

napari ships six buttons in one undifferentiated line. Three of them are not
useful here - rolling and transposing the axes are both worse ways of choosing a
view than naming it, and the IPython console is not part of the workflow - and the
three that remain read better grouped.

The result is::

    Slice [Z] [Y] [X]
    Mode  [reset] [grid] [2D/3D]

napari shows the *last* two axes of `viewer.dims.order` and puts sliders on the
rest, so a view is chosen by ordering the axes so the pair you want to see comes
last.

This reaches into private API (`_qt_viewer._viewerButtons`). Removed buttons are
hidden rather than deleted, because napari holds references to them and connects
signals to them; destroying them would leave those connections pointing at freed
C++ objects. If a future napari moves the row, `rebuild_button_row` returns False
rather than raising.
"""

from __future__ import annotations

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QGridLayout, QLabel, QPushButton

from .crosshair import steady_zoom

# name, dims order, shortcut. World axes are (z, y, x) = (0, 1, 2).
#
# Shift+letter: the letter names the axis, and napari's own bindings are plain
# letters and Ctrl combinations. Ctrl+Alt was tried and is AltGr on many Windows
# layouts, where it types a character instead of firing.
VIEWS = (
    ("Z", (0, 1, 2), "Shift-Z"),  # display (y, x) - looking down z
    ("Y", (1, 0, 2), "Shift-Y"),  # display (z, x) - looking down y
    ("X", (2, 0, 1), "Shift-X"),  # display (z, y) - looking down x
)

# napari's own buttons worth keeping, in the order they should appear.
KEEP = ("resetViewButton", "gridViewButton", "ndisplayButton")


def set_view(viewer, order, crosshair=None) -> None:
    """Look down one axis.

    Args:
        viewer: The napari viewer.
        order: Dimension order to set; its first axis lands on the slider.
        crosshair: The crosshair, if there is one. With a point marked, the new view
            opens *on* it - sliders onto the mark, camera centred there at the zoom
            you already had. Without one there is nothing to aim at, so the view is
            reset to fit, which is also what stops a rotated volume opening off
            screen.
    """
    marked = crosshair is not None and crosshair.position is not None
    if not marked:
        viewer.dims.order = tuple(order)
        viewer.reset_view()
        return

    # `held` because changing the order swaps which axis is out of plane, and the
    # crosshair would otherwise adopt that axis's stale slider value in the moment
    # before `recall` moves it onto the marked one. `steady_zoom` because napari
    # refits the camera on an order change whether we ask it to or not.
    with crosshair.held(), steady_zoom(viewer):
        viewer.dims.order = tuple(order)
        crosshair.recall()
        crosshair.look()


def _group_label(text: str) -> QLabel:
    """Build the greyed row label for a group of buttons.

    Args:
        text: The label.

    Returns:
        A right-aligned QLabel.
    """
    label = QLabel(text)
    label.setStyleSheet("color: gray; margin-right: 4px;")
    label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
    return label


def pretty(key: str) -> str:
    """Render a key binding the way a person writes it.

    Args:
        key: A napari key string, such as "Shift-Z".

    Returns:
        The readable form, such as "Shift+Z".
    """
    return key.replace("Control", "Ctrl").replace("-", "+")


def bind_view_keys(viewer, crosshair=None) -> list:
    """Install the view shortcuts.

    Bound without overwriting, so a napari that has claimed one of these keeps it
    and we lose a shortcut rather than one of its documented behaviours. They fire
    only while the canvas has focus - after typing in a panel, click the image
    first.

    Args:
        viewer: The napari viewer.
        crosshair: Passed to `set_view`, so the shortcuts behave as the buttons do.

    Returns:
        (key, description) for each shortcut that bound. A short list means napari
        has claimed one; the reason is printed.
    """
    bound = []
    for name, order, key in VIEWS:
        try:
            viewer.bind_key(
                key,
                lambda v, o=order: set_view(v, o, crosshair),
                overwrite=False,
            )
        except (AttributeError, TypeError, ValueError) as exc:
            print(f"note: {pretty(key)} not bound ({exc})")
            continue
        bound.append((key, f"look down {name.lower()}"))
    return bound


def _view_button(viewer, name: str, order, key: str, crosshair) -> QPushButton:
    """Build one axis button.

    Args:
        viewer: The napari viewer.
        name: Axis name shown on the button.
        order: Dimension order it sets.
        key: Its keyboard shortcut, for the tooltip.
        crosshair: Passed to `set_view`.

    Returns:
        The button, already connected.
    """
    button = QPushButton(name)
    button.setToolTip(f"Look down the {name.lower()} axis  ({pretty(key)})")
    button.setFixedWidth(26)
    button.clicked.connect(
        lambda _=False, o=order: set_view(viewer, o, crosshair)
    )
    return button


def rebuild_button_row(viewer, crosshair=None) -> bool:
    """Replace napari's button row with ours.

    Args:
        viewer: The napari viewer.
        crosshair: Passed to the axis buttons.

    Returns:
        False if napari has moved the row, in which case it is left as shipped.
    """
    qt_viewer = getattr(viewer.window, "_qt_viewer", None)
    row = getattr(qt_viewer, "_viewerButtons", None)
    layout = row.layout() if row is not None else None
    if layout is None:
        return False

    # Empty the layout, hiding rather than destroying: the widgets stay children
    # of the frame so napari's references and signal connections remain valid.
    while layout.count():
        item = layout.takeAt(0)
        widget = item.widget()
        if widget is not None:
            widget.hide()

    # A grid nested inside the frame's existing QHBoxLayout. Qt will not let a
    # widget be given a second layout, so the row layout is reused rather than
    # replaced, and the two rows come from the grid inside it.
    grid = QGridLayout()
    grid.setContentsMargins(0, 0, 0, 0)
    grid.setSpacing(2)

    grid.addWidget(_group_label("Slice"), 0, 0)
    for column, (name, order, key) in enumerate(VIEWS, start=1):
        grid.addWidget(_view_button(viewer, name, order, key, crosshair), 0, column)

    grid.addWidget(_group_label("Mode"), 1, 0)
    for column, name in enumerate(KEEP, start=1):
        button = getattr(row, name, None)
        if button is not None:
            button.show()
            grid.addWidget(button, 1, column)

    layout.addLayout(grid)
    layout.addStretch(1)
    return True
