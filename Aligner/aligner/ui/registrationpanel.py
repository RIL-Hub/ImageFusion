"""The chain, the session file, and turning landmarks into a registration.

Solving happens here but the solver does not. Pressing the button reads the session
and calls `aligner.solve`, which knows nothing about Qt; the same session file solves
identically from a terminal.

Picking and linking live in `landmarkpanel`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from qtpy.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QListWidget,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..apply import ApplyError, apply_session, save_transforms
from ..geometry import display_state
from ..landmarks import LandmarkError, load_session, save_session
from ..solve import MIN_POINTS, Rigid, SolveError, describe, solve_session
from .widgets import group, note

SESSION_FILTER = "Aligner session (*.yaml *.yml);;All files (*)"


class RegistrationPanel(QWidget):
    """Order the chain, save the session, solve it, and write the result out."""

    def __init__(self, picker, controls=None):
        super().__init__()
        self._picker = picker
        self._controls = controls  # drives "show fit"; this panel places no layers

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)
        layout.addWidget(self._chain_group())
        layout.addWidget(self._session_group())
        layout.addWidget(self._solve_group(), 1)

        picker.listeners.append(self.refresh)
        self.refresh()

    # --- construction -------------------------------------------------------

    def _chain_group(self) -> QGroupBox:
        """Build the chain list and its reorder buttons.

        Returns:
            The group box.
        """
        box, inside = group("Chain")
        inside.addWidget(note("Each image registers to the one below. The last is fixed."))

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)

        self._chain = QListWidget()
        self._chain.setMaximumHeight(90)
        row.addWidget(self._chain, 1)

        buttons = QVBoxLayout()
        buttons.setContentsMargins(0, 0, 0, 0)
        buttons.setSpacing(2)
        for text, offset in (("up", -1), ("down", 1)):
            button = QPushButton(text)
            button.setFixedWidth(50)
            button.clicked.connect(lambda _=False, o=offset: self._move_in_chain(o))
            buttons.addWidget(button)
        buttons.addStretch(1)
        row.addLayout(buttons, 0)

        holder = QWidget()
        holder.setLayout(row)
        inside.addWidget(holder)
        return box

    def _session_group(self) -> QGroupBox:
        """Build the save and load controls.

        Returns:
            The group box.
        """
        box, inside = group("Session")
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        for text, slot in (("save ...", self._save), ("load ...", self._load)):
            button = QPushButton(text)
            button.clicked.connect(slot)
            row.addWidget(button)
        holder = QWidget()
        holder.setLayout(row)
        inside.addWidget(holder)

        self._session_path = note("not saved yet")
        inside.addWidget(self._session_path)
        return box

    def _solve_group(self) -> QGroupBox:
        """Build the solve, preview, undo and export controls, and the report box.

        Returns:
            The group box.
        """
        box, inside = group("Registration")

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        for text, slot, tip in (
            ("solve", self._solve, "Fit each link and report the residuals"),
            (
                "show fit",
                self._preview,
                "Move every volume on screen as the fit says. Display only - "
                "nothing is written and no data is resampled on disk.",
            ),
            ("undo fit", self._clear, "Put every volume back where it started"),
        ):
            button = QPushButton(text)
            button.setToolTip(tip)
            button.clicked.connect(slot)
            row.addWidget(button)
        holder = QWidget()
        holder.setLayout(row)
        inside.addWidget(holder)

        export = QPushButton("export registered series ...")
        export.setToolTip(
            "Write each moving volume into the fixed volume's frame, plus the "
            "transforms. Geometry is rewritten; voxels are copied untouched."
        )
        export.clicked.connect(self._export)
        inside.addWidget(export)

        self._report = QPlainTextEdit()
        self._report.setReadOnly(True)
        self._report.setStyleSheet("font-family: monospace;")
        self._report.setPlaceholderText(
            f"Link at least {MIN_POINTS} landmarks between each neighbouring pair, "
            "then solve."
        )
        inside.addWidget(self._report, 1)
        return box

    # --- display ------------------------------------------------------------

    def refresh(self) -> None:
        """Redisplay the chain, marking the fixed and passed-over images."""
        session = self._picker.session
        volumes = list(session.chain) or sorted(session.volumes)
        active = session.active_chain()

        self._chain.clear()
        for name in volumes:
            if name not in active:
                self._chain.addItem(f"{name}    (passed over)")
            elif name == active[-1]:
                self._chain.addItem(f"{name}    (fixed)")
            else:
                self._chain.addItem(name)

    def _move_in_chain(self, offset: int) -> None:
        """Move the selected image up or down the chain.

        Args:
            offset: -1 to move it earlier, +1 later.
        """
        session = self._picker.session
        order = list(session.chain)
        row = self._chain.currentRow()
        target = row + offset
        if row < 0 or not (0 <= target < len(order)):
            return
        order[row], order[target] = order[target], order[row]
        session.chain = tuple(order)
        self.refresh()
        self._chain.setCurrentRow(target)

    # --- the session file ---------------------------------------------------

    def _save(self) -> None:
        """Ask for a file and write the session to it."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Save session", "session.yaml", SESSION_FILTER
        )
        if not path:
            return
        try:
            save_session(path, self._picker.session)
        except (LandmarkError, OSError) as exc:
            self._show([f"could not save: {exc}"])
            return
        self._session_path.setText(str(path))

    def _load(self) -> None:
        """Ask for a file, adopt the session in it, and report anything unopenable."""
        path, _ = QFileDialog.getOpenFileName(self, "Load session", "", SESSION_FILTER)
        if not path:
            return
        try:
            session = load_session(path)
        except LandmarkError as exc:
            self._show([f"could not load: {exc}"])
            return

        self._picker.use(session)
        self._session_path.setText(str(path))

        # Landmarks for volumes that are not open cannot be drawn, and would be
        # written straight back out on the next save. Say so rather than lose them
        # quietly.
        absent = sorted(
            {m.volume for m in session.landmarks} - set(self._picker.names())
        )
        if absent:
            self._show(
                [
                    f"loaded {Path(path).name}",
                    "",
                    "these volumes hold landmarks but are not open:",
                ]
                + [f"   {name}" for name in absent]
                + ["", "they are kept in the session but cannot be shown or edited."]
            )
        else:
            self._show([f"loaded {Path(path).name}"])

    # --- solving ------------------------------------------------------------

    def _solve(self) -> None:
        """Fit every link and show the residuals. Nothing is moved or written."""
        try:
            solution = solve_session(self._picker.session)
        except (SolveError, LandmarkError) as exc:
            self._show([str(exc)])
            return
        self._show(describe(solution))

    def _preview(self) -> None:
        """Move every volume on screen as the fit says, without touching the data.

        The display already holds a rotation about the volume's centre plus a
        millimetre offset, which is exactly six degrees of freedom - so a solved
        transform goes straight in. Landmarks are stored in each volume's own frame
        and redraw with it, so the marks converging is a genuine check rather than a
        restatement of the fit.
        """
        if self._controls is None:
            self._show(["no orientation controls to drive"])
            return
        try:
            solution = solve_session(self._picker.session)
        except (SolveError, LandmarkError) as exc:
            self._show([str(exc)])
            return

        placements = dict(solution.to_fixed)
        # A passed-over image is shown truthfully rather than left wherever it was
        # dragged: it is not registered, so the honest thing is its own geometry.
        for name in solution.skipped:
            placements.setdefault(name, Rigid.identity())

        moved = []
        for name, transform in placements.items():
            layer = self._picker.image(name)
            if layer is None:
                continue
            matrix, nudge = display_state(
                layer.metadata["volume"].geometry,
                transform.rotation,
                transform.translation,
            )
            self._controls.set_placement(layer, matrix, nudge)
            moved.append(name)

        self._show(
            describe(solution)
            + [
                "",
                "moved on screen: " + ", ".join(moved),
                "display only - nothing written, no data resampled.",
            ]
        )

    def _clear(self) -> None:
        """Back to how each volume arrived. Landmarks are unaffected either way."""
        if self._controls is None:
            return
        for name in self._picker.names():
            self._controls.set_placement(
                self._picker.image(name), np.eye(3), np.zeros(3)
            )
        self._show(["display reset; landmarks and the session are unchanged"])

    def _export(self) -> None:
        """Write the registered series and the transforms to a chosen directory."""
        session = self._picker.session
        try:
            solution = solve_session(session)
        except (SolveError, LandmarkError) as exc:
            self._show([str(exc)])
            return

        directory = QFileDialog.getExistingDirectory(self, "Export into")
        if not directory:
            return

        try:
            written = apply_session(session, solution, directory)
            transforms = save_transforms(
                Path(directory) / "transforms.yaml", session, solution
            )
        except (ApplyError, OSError) as exc:
            self._show([f"could not export: {exc}"])
            return

        self._show(
            [
                f"{record.slices} slices -> {record.destination.name}"
                f"   (rotated {record.rotation_degrees:.2f}deg, "
                f"shifted {record.shift_mm:.2f} mm)"
                for record in written
            ]
            + [
                f"transforms -> {transforms.name}",
                "",
                f"{solution.fixed} was not copied: it is the fixed frame.",
                "Geometry rewritten, voxels copied untouched, nothing resampled.",
            ]
        )

    def _show(self, lines) -> None:
        """Replace the report box's contents.

        Args:
            lines: Text to show, one line each.
        """
        self._report.setPlainText("\n".join(lines))
