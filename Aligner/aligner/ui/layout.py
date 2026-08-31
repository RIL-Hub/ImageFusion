"""Deterministic arrangement of napari's dock widgets.

napari docks are Qt QDockWidgets in one of four dock areas. Left to itself, napari
restores whatever arrangement was saved from the last session, so panels land
wherever they were last dragged. This pins them instead.

Panels are described as *groups*: each group is one row in the dock area, and the
panels within a group share it as tabs. Adding a panel later means adding its title
to PANEL_GROUPS - either into an existing group to tab it alongside, or as a new
tuple to give it its own row.

Titles are matched case-insensitively against each dock's window title, and
anything not found is skipped rather than raising, since napari renames its
built-ins occasionally.
"""

from __future__ import annotations

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QDockWidget

# One tuple per row, top to bottom. Panels inside a tuple are tabbed together.
PANEL_GROUPS = (
    ("layer list", "layer controls", "layer info", "crosshair"),
    ("orientation", "landmarks", "registration"),
)

AREAS = {
    "left": Qt.LeftDockWidgetArea,
    "right": Qt.RightDockWidgetArea,
    "top": Qt.TopDockWidgetArea,
    "bottom": Qt.BottomDockWidgetArea,
}


def find_docks(viewer) -> dict:
    """Find every dock widget in the window.

    Args:
        viewer: The napari viewer.

    Returns:
        Dock widgets keyed by lowercased title; untitled docks are skipped.
    """
    window = viewer.window._qt_window
    found = {}
    for dock in window.findChildren(QDockWidget):
        title = (dock.windowTitle() or "").strip().lower()
        if title:
            found[title] = dock
    return found


def arrange(viewer, groups=PANEL_GROUPS, area: str = "left", verbose: bool = False):
    """Lay the named docks out as rows of tab groups.

    Args:
        viewer: The napari viewer.
        groups: One tuple per row; panels within a tuple are tabbed together.
        area: Dock area to use - left, right, top or bottom.
        verbose: Print the dock titles found, for diagnosing a rename.

    Returns:
        The titles actually placed. Anything not found is skipped, so a short list
        means napari has renamed a built-in panel.
    """
    window = viewer.window._qt_window
    docks = find_docks(viewer)

    if verbose:
        print(f"dock widgets found: {sorted(docks)}")

    # Drop anything missing, and any group left empty by that.
    resolved = [
        [(title, docks[title]) for title in group if title in docks]
        for group in groups
    ]
    resolved = [group for group in resolved if group]
    if not resolved:
        return []

    dock_area = AREAS[area]

    # Everything into the target area first; split and tabify only work on docks
    # that already share one.
    for group in resolved:
        for _, dock in group:
            dock.setFloating(False)
            window.addDockWidget(dock_area, dock)
            dock.setVisible(True)

    # Each group's leader defines a row, stacked under the previous leader.
    leaders = [group[0][1] for group in resolved]
    for upper, lower in zip(leaders, leaders[1:]):
        window.splitDockWidget(upper, lower, Qt.Vertical)

    # Remaining members of a group become tabs on its leader.
    for group in resolved:
        leader = group[0][1]
        for _, dock in group[1:]:
            window.tabifyDockWidget(leader, dock)
        leader.raise_()  # show the first tab rather than the last one added

    return [title for group in resolved for title, _ in group]


def freeze_geometry(save: bool = False) -> None:
    """Stop napari restoring the previous session's window arrangement.

    Without this, our arrangement is applied and then overwritten by whatever was
    saved last time.

    Args:
        save: Whether napari should save and restore window geometry.
    """
    try:
        from napari.settings import get_settings

        get_settings().application.save_window_geometry = save
    except Exception:
        pass
