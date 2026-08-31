"""What has been picked in each image, and which picks are the same feature.

Points are added and deleted through napari's own Points layer - this panel does not
duplicate that. What it adds is the part napari cannot know about: which picks in
different images are the *same* feature, which is the pairing the whole registration
rests on. One tab per image, one row per landmark in it.

Linking is a merge, not a stored pair: two landmarks are the same feature exactly when
they share a number, so linking renumbers one to match the other. That keeps the
relation transitive for free - link a to b and b to c, and a and c are linked too.
Unlinking is one-sided: it gives this pick a fresh number and leaves the others alone.

The controls act on the selected row rather than living inside the cells. In a dock
this narrow a combo and a spin box per row left neither usable.

The chain, the session file and solving live in `registrationpanel`.
"""

from __future__ import annotations

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..landmarks import LandmarkError, placement_for
from ..solve import MIN_POINTS
from .widgets import group, note

ID_WIDTH = 34
LABEL_WIDTH = 80


class LandmarkPanel(QWidget):
    """One tab per image: its landmarks, and what they link to."""

    def __init__(self, picker, viewer=None, crosshair=None):
        super().__init__()
        self._picker = picker
        self._viewer = viewer
        self._crosshair = crosshair
        self._tables = {}  # volume name -> its table
        self._tab_names = []  # parallel to the tabs; see _fill_tabs
        self._busy = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)
        layout.addWidget(self._landmark_group(), 1)

        picker.listeners.append(self.refresh)
        if crosshair is not None:
            crosshair.connect(self._update_buttons)
        self.refresh()

    # --- construction -------------------------------------------------------

    def _landmark_group(self) -> QGroupBox:
        """Build the tabs, the action controls and the status lines.

        Returns:
            The group box.
        """
        box, inside = group("Landmarks")
        inside.addWidget(note("Select a landmark, then act on it below."))

        self._tabs = QTabWidget()
        self._tabs.setMinimumHeight(160)
        self._tabs.currentChanged.connect(lambda *_: self._update_buttons())
        inside.addWidget(self._tabs, 1)

        inside.addLayout(self._link_row())
        inside.addLayout(self._action_row())

        self._tally = note()
        inside.addWidget(self._tally)
        self._status = note()
        inside.addWidget(self._status)
        return box

    def _link_row(self) -> QHBoxLayout:
        """Build the link controls: which image, which number.

        Returns:
            The row layout.
        """
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(QLabel("link to"))

        self._target = QComboBox()
        row.addWidget(self._target, 1)

        row.addWidget(QLabel("as"))
        self._as = QSpinBox()
        self._as.setRange(1, 10_000)
        self._as.setKeyboardTracking(False)
        self._as.setFixedWidth(64)
        self._as.setToolTip("The number the two should share")
        row.addWidget(self._as)

        self._link_button = QPushButton("link")
        self._link_button.clicked.connect(self._link)
        row.addWidget(self._link_button)
        return row

    def _action_row(self) -> QHBoxLayout:
        """Build the add, unlink and delete buttons.

        Returns:
            The row layout.
        """
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)

        self._add_button = QPushButton("add at crosshair")
        self._add_button.clicked.connect(self._add_at_crosshair)
        row.addWidget(self._add_button)

        self._unlink_button = QPushButton("unlink")
        self._unlink_button.setToolTip(
            "Give this pick its own number, leaving the others it was paired with"
        )
        self._unlink_button.clicked.connect(self._unlink)
        row.addWidget(self._unlink_button)

        self._delete_button = QPushButton("delete")
        self._delete_button.setToolTip("Remove this pick from this image")
        self._delete_button.clicked.connect(self._delete)
        row.addWidget(self._delete_button)
        return row

    # --- one tab per image --------------------------------------------------

    def _fill_tabs(self, session, volumes) -> None:
        """Rebuild every tab, keeping whichever was open.

        Args:
            session: The current session.
            volumes: Every volume, in chain order.
        """
        # Remembered by volume name, not by tab label: the label carries the landmark
        # count, so adding a point changed it and the open tab jumped to the first.
        current = self._tabs.currentIndex()
        open_name = (
            self._tab_names[current] if 0 <= current < len(self._tab_names) else None
        )
        while self._tabs.count():
            self._tabs.removeTab(0)
        self._tab_names = []
        self._tables = {}

        for name in volumes:
            marks = sorted(session.for_volume(name), key=lambda m: m.id)
            table = self._volume_table(session, name, marks)
            self._tables[name] = table
            self._tabs.addTab(table, f"{name} ({len(marks)})" if marks else name)
            self._tab_names.append(name)

        if open_name in self._tab_names:
            self._tabs.setCurrentIndex(self._tab_names.index(open_name))

        self._target.clear()
        self._target.addItems([v for v in volumes if v != self._open_volume()])

    def _volume_table(self, session, name, marks) -> QTableWidget:
        """Build one image's landmark table.

        Args:
            session: The current session.
            name: Volume this tab is for.
            marks: Its landmarks, already sorted by id.

        Returns:
            The table.
        """
        table = QTableWidget(len(marks), 3)
        table.setAlternatingRowColors(True)
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.setSelectionMode(QAbstractItemView.SingleSelection)
        table.verticalHeader().setVisible(False)
        table.setHorizontalHeaderLabels(["id", "label", "also in"])

        header = table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Fixed)
        header.setSectionResizeMode(1, QHeaderView.Interactive)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        table.setColumnWidth(0, ID_WIDTH)
        table.setColumnWidth(1, LABEL_WIDTH)

        for row, mark in enumerate(marks):
            number = QTableWidgetItem(str(mark.id))
            number.setData(Qt.UserRole, mark.id)
            table.setItem(row, 0, number)
            table.setItem(row, 1, QTableWidgetItem(mark.label))

            elsewhere = sorted(
                other.volume
                for other in session.landmarks
                if other.id == mark.id and other.volume != name
            )
            shared = QTableWidgetItem(", ".join(elsewhere))
            shared.setFlags(shared.flags() & ~Qt.ItemIsEditable)
            table.setItem(row, 2, shared)

        table.itemChanged.connect(lambda item, v=name: self._cell_edited(v, item))
        table.itemSelectionChanged.connect(self._selection_changed)
        table.cellClicked.connect(lambda r, c, v=name, t=table: self._jump(v, t, r, c))
        return table

    # --- what is selected ---------------------------------------------------

    def _open_volume(self):
        """Name the image whose tab is showing.

        Returns:
            The volume name, or None if there are no tabs.
        """
        index = self._tabs.currentIndex()
        return self._tab_names[index] if 0 <= index < len(self._tab_names) else None

    def _selected(self):
        """Find the landmark the controls act on.

        Returns:
            (volume, id), or (None, None) if nothing is selected.
        """
        name = self._open_volume()
        table = self._tables.get(name)
        if table is None:
            return None, None
        rows = table.selectionModel().selectedRows() if table.selectionModel() else []
        if not rows:
            return name, None
        number = table.item(rows[0].row(), 0)
        return name, (number.data(Qt.UserRole) if number is not None else None)

    def _selection_changed(self) -> None:
        """Re-enable the controls, and offer the selected landmark's own number."""
        if self._busy:
            return
        self._update_buttons()
        _, identifier = self._selected()
        if identifier is not None:
            was, self._busy = self._busy, True
            try:
                self._as.setValue(int(identifier))
            finally:
                self._busy = was

    def _update_buttons(self) -> None:
        """Enable each control only when it has something to act on."""
        name, identifier = self._selected()
        picked = identifier is not None
        placed = self._crosshair is not None and self._crosshair.position is not None

        for button in (self._link_button, self._unlink_button, self._delete_button):
            try:
                button.setEnabled(picked)
            except RuntimeError:
                pass
        try:
            self._add_button.setEnabled(placed and name is not None)
            self._add_button.setToolTip(
                f"Place a landmark in {name} where the crosshair is"
                if placed
                else "Mark a point with the crosshair first (Shift+click, or T)"
            )
            # The current image is never a link target.
            others = [v for v in self._tab_names if v != name]
            if [self._target.itemText(i) for i in range(self._target.count())] != others:
                self._target.clear()
                self._target.addItems(others)
        except RuntimeError:
            pass

    # --- acting on it -------------------------------------------------------

    def _link(self) -> None:
        """Declare the selected landmark the same feature as one in another image."""
        _, identifier = self._selected()
        if identifier is None:
            self._say("select a landmark first")
            return
        self._repair(identifier, str(self._as.value()))

    def _unlink(self) -> None:
        """Give the selected landmark its own number, leaving its partners alone."""
        name, identifier = self._selected()
        if identifier is None:
            self._say("select a landmark first")
            return
        mark = self._picker.unlink(name, identifier)
        self._say(
            f"landmark {identifier} in {name} is now {mark.id}"
            if mark
            else "nothing to unlink"
        )

    def _delete(self) -> None:
        """Remove the selected landmark from this image."""
        name, identifier = self._selected()
        if identifier is None:
            self._say("select a landmark first")
            return
        if self._picker.delete(name, identifier):
            self._say(f"deleted landmark {identifier} from {name}")
        else:
            self._say("nothing to delete")

    def _add_at_crosshair(self) -> None:
        """Commit the crosshair as a landmark in the image whose tab is open."""
        name = self._open_volume()
        if name is None:
            return
        if self._crosshair is None or self._crosshair.position is None:
            self._say("no crosshair placed")
            return
        mark, reason = self._picker.add_at(name, self._crosshair.position)
        if mark is None:
            self._say(reason or f"could not add a landmark to {name}")
            return
        self._say(f"landmark {mark.id} added to {name}")

    # --- display ------------------------------------------------------------

    def refresh(self) -> None:
        """Rebuild the tabs and the tally from the session."""
        session = self._picker.session
        volumes = list(session.chain) or sorted(session.volumes)

        self._busy = True
        try:
            self._fill_tabs(session, volumes)
        finally:
            self._busy = False

        self._tally.setText(self._summarise(session))
        self._update_buttons()

    def _summarise(self, session) -> str:
        """Count what is linked, and what is not yet.

        Args:
            session: The current session.

        Returns:
            The tally shown under the controls.
        """
        if not session.landmarks:
            return "nothing picked yet"
        parts = []
        for source, target in session.links():
            shared = len(session.paired(source, target)[0])
            enough = "" if shared >= MIN_POINTS else f" (needs {MIN_POINTS})"
            parts.append(f"{source}-{target} {shared}{enough}")
        lines = ["linked  " + "   ".join(parts)] if parts else []

        seen = {}
        for mark in session.landmarks:
            seen.setdefault(mark.id, set()).add(mark.volume)
        lonely = sum(1 for elsewhere in seen.values() if len(elsewhere) < 2)
        if lonely:
            lines.append(f"{lonely} not yet linked")
        return "\n".join(lines)

    def _say(self, message: str) -> None:
        """Show one line of feedback under the tally.

        Args:
            message: What happened.
        """
        self._status.setText(message)

    # --- editing in the table -----------------------------------------------

    def _cell_edited(self, volume: str, item) -> None:
        """Apply an edit to an id or a label.

        Args:
            volume: Volume whose tab was edited.
            item: The changed cell. Column 0 renumbers, column 1 relabels.
        """
        if self._busy or item.column() > 1:
            return
        table = item.tableWidget()
        number = table.item(item.row(), 0) if table is not None else None
        if number is None:
            return
        identifier = number.data(Qt.UserRole)

        if item.column() == 1:
            session = self._picker.session
            session.landmarks = [
                mark if mark.id != identifier else _relabelled(mark, item.text())
                for mark in session.landmarks
            ]
            return

        self._repair(identifier, item.text())

    def _repair(self, old, text: str) -> None:
        """Merge one landmark into another, pairing the two.

        Args:
            old: The landmark id being changed.
            text: The id to merge into, as text.
        """
        try:
            new = int(text)
        except ValueError:
            self._say(f"'{text}' is not a landmark number")
            self.refresh()
            return
        try:
            self._picker.session.renumber(int(old), new)
        except LandmarkError as exc:
            self._say(str(exc))
            self.refresh()
            return
        # The ids live in the points layers too, so they have to be redrawn.
        self._picker.reposition()
        self._say(f"landmark {old} is now {new}")
        self.refresh()

    # --- finding a landmark again -------------------------------------------

    def _jump(self, volume: str, table, row: int, column: int) -> None:
        """Bring one landmark into view, in the image whose tab it is on.

        Args:
            volume: Volume whose tab was clicked.
            table: That tab's table.
            row: Row clicked.
            column: Column clicked; unused, any column navigates.
        """
        number = table.item(row, 0)
        if number is None:
            return

        identifier = number.data(Qt.UserRole)
        mark = next(
            (
                candidate
                for candidate in self._picker.session.landmarks
                if candidate.id == identifier and candidate.volume == volume
            ),
            None,
        )
        if mark is None:
            return

        layer = self._picker.image(volume)
        place = placement_for(layer) if layer is not None else None
        if place is not None:
            self._look_at(place.patient_to_world(mark.patient))

    def _look_at(self, world) -> None:
        """Move the sliders and the camera to a world position.

        Args:
            world: (z, y, x) mm to go to.
        """
        viewer = self._viewer
        if viewer is None:
            return
        try:
            for axis, value in enumerate(world):
                viewer.dims.set_point(axis, float(value))
        except (AttributeError, IndexError, TypeError, ValueError):
            pass
        try:
            viewer.camera.center = tuple(float(value) for value in world)
        except (AttributeError, TypeError, ValueError):
            pass


def _relabelled(mark, label: str):
    """Relabel a landmark.

    Args:
        mark: The landmark; it is frozen, so a copy is made.
        label: The new label.

    Returns:
        A new `Landmark`, identical but for the label.
    """
    from ..landmarks import Landmark

    return Landmark(id=mark.id, volume=mark.volume, patient=mark.patient, label=label)
