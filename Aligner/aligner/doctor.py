"""What is installed, and whether the parts we reach into are where we expect.

Aligner reaches into a handful of napari internals, because the things it needs -
drawing an overlay that ignores slicing, replacing the viewer button row - have no
public API. Those internals move: between 0.8 and 0.9 the overlay store moved from
`viewer._overlays` to `viewer.scene.overlays`, `VispyBaseOverlay` gained a required
argument, and `dims.events.order` was wired to `fit_to_view`. Each of those looked, from
the outside, like "the crosshair doesn't work".

So the first thing to run when something is wrong is this. It names versions and says
which hooks resolved, which turns a bug report into a diagnosis.

Checks are import-level on purpose: no viewer is constructed, so this is quick and
works over SSH. The few things that can only be seen with a live viewer are reported at
launch instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

# napari 0.9 is not a preference. The overlay store, the vispy overlay constructor and
# the automatic refit on an axis-order change are all 0.9 shapes.
NAPARI_MINIMUM = (0, 9)

# pydicom 3 dropped `Dataset.is_little_endian` and changed how `save_as` decides a
# transfer syntax, which is what Image2Dicom's writer and our applier are built on.
PYDICOM_CEILING = (3,)

PACKAGES = ("napari", "numpy", "scipy", "dask", "pydicom", "yaml", "typer", "vispy")


@dataclass(frozen=True)
class Check:
    """One thing that either holds or does not.

    Attributes:
        name: Short label for the thing tested.
        ok: Whether it holds.
        detail: What was actually found, or why it failed.
        essential: False for something the program degrades without, such as the
            crosshair; True for something it cannot start without.
    """

    name: str
    ok: bool
    detail: str
    essential: bool = True


def _version(name: str) -> Optional[str]:
    """Read an installed package's version.

    Args:
        name: Importable module name.

    Returns:
        Its version string, a placeholder if it declares none, or None if it is not
        installed.
    """
    try:
        module = __import__(name)
    except Exception:
        return None
    for attribute in ("__version__", "VERSION", "version"):
        value = getattr(module, attribute, None)
        if isinstance(value, str):
            return value
    return "(installed, version unknown)"


def _parsed(text: Optional[str]) -> Tuple[int, ...]:
    """Parse a version string into something comparable.

    Args:
        text: A version string, or None.

    Returns:
        Its leading numeric components, so '0.9.1rc2' gives (0, 9, 1) and anything
        unparseable gives (). String comparison would rank 0.10 below 0.9.
    """
    if not text:
        return ()
    parts = []
    for chunk in text.split("."):
        digits = ""
        for character in chunk:
            if not character.isdigit():
                break
            digits += character
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def versions() -> List[Tuple[str, str]]:
    """Collect everything worth naming in a bug report.

    Returns:
        (label, value) pairs covering Python, the platform, Qt and every
        dependency.
    """
    import platform
    import sys

    found = [
        ("python", platform.python_version()),
        ("platform", f"{platform.system()} {platform.release()}"),
        ("executable", sys.executable),
    ]

    try:
        import qtpy

        found.append(("qt binding", getattr(qtpy, "API_NAME", "?")))
        found.append(("qt version", getattr(qtpy, "QT_VERSION", "?")))
    except Exception as exc:
        found.append(("qt binding", f"NOT IMPORTABLE - {exc}"))

    for name in PACKAGES:
        found.append((name, _version(name) or "NOT INSTALLED"))
    return found


def checks() -> List[Check]:
    """Test every dependency and every napari internal the program relies on.

    Returns:
        A `Check` per test, essential ones first. Nothing here constructs a viewer.
    """
    results: List[Check] = []

    for name, why in (
        ("napari", "the viewer"),
        ("numpy", "everything"),
        ("scipy", "oblique plane sampling"),
        ("dask", "lazy slice loading"),
        ("pydicom", "reading and writing series"),
        ("yaml", "session and job files"),
    ):
        version = _version(name)
        results.append(
            Check(
                f"{name} installed",
                version is not None,
                version or f"required for {why}",
            )
        )

    napari_version = _version("napari")
    parsed = _parsed(napari_version)
    results.append(
        Check(
            f"napari >= {'.'.join(str(v) for v in NAPARI_MINIMUM)}",
            bool(parsed) and parsed >= NAPARI_MINIMUM,
            napari_version or "napari not installed",
        )
    )

    pydicom_version = _version("pydicom")
    parsed_pydicom = _parsed(pydicom_version)
    results.append(
        Check(
            "pydicom < 3",
            bool(parsed_pydicom) and parsed_pydicom < PYDICOM_CEILING,
            pydicom_version or "pydicom not installed",
        )
    )

    # --- the internals we reach into ---------------------------------------

    try:
        from .ui.crosshair import AVAILABLE, UNAVAILABLE_BECAUSE

        results.append(
            Check(
                "crosshair overlay classes",
                AVAILABLE,
                UNAVAILABLE_BECAUSE or "napari.components.overlays + _vispy resolved",
                essential=False,
            )
        )
    except Exception as exc:
        results.append(
            Check("crosshair overlay classes", False, str(exc), essential=False)
        )

    results.append(_scene_overlays())
    results.append(_projection_mode())
    return results


def _scene_overlays() -> Check:
    """Find where a scene overlay has to be installed.

    Returns:
        A `Check` naming the overlay store this napari has; it moved from
        `viewer._overlays` to `viewer.scene.overlays` in 0.9.
    """
    try:
        from napari.components.viewer_model import ViewerModel

        fields = getattr(ViewerModel, "model_fields", None) or getattr(
            ViewerModel, "__fields__", {}
        )
        if "scene" in fields:
            return Check(
                "overlay store", True, "viewer.scene.overlays (0.9)", essential=False
            )
        return Check(
            "overlay store",
            True,
            "viewer._overlays (pre-0.9); the crosshair falls back to this",
            essential=False,
        )
    except Exception as exc:
        return Check("overlay store", False, str(exc), essential=False)


def _projection_mode() -> Check:
    """Find how landmarks can be shown from a slice or two away.

    Returns:
        A `Check` naming the spelling this napari uses; `out_of_slice_display`
        became `projection_mode` in 0.9.
    """
    try:
        from napari.layers import Points

        if hasattr(Points, "projection_mode"):
            return Check(
                "points out-of-slice display",
                True,
                "projection_mode (0.9)",
                essential=False,
            )
        return Check(
            "points out-of-slice display",
            True,
            "out_of_slice_display (pre-0.9)",
            essential=False,
        )
    except Exception as exc:
        return Check("points out-of-slice display", False, str(exc), essential=False)


def report() -> Tuple[List[str], bool]:
    """Render the whole report.

    Returns:
        (lines, healthy): the text to print, and False if anything essential
        failed - in which case the viewer will not start.
    """
    lines = ["environment", ""]
    for name, value in versions():
        lines.append(f"  {name:<12} {value}")

    lines += ["", "checks", ""]
    healthy = True
    for check in checks():
        mark = "ok  " if check.ok else ("FAIL" if check.essential else "warn")
        healthy &= check.ok or not check.essential
        lines.append(f"  [{mark}] {check.name:<28} {check.detail}")

    lines.append("")
    lines.append(
        "all essential checks passed"
        if healthy
        else "something essential is missing; the viewer will not start"
    )
    return lines, healthy
