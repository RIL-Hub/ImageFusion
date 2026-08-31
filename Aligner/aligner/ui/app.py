"""Build the napari viewer, put the volumes in it, and wire the panels together.

The one thing this has to get right is that every volume lands in a shared world
space via its own affine, so volumes at 20 um and 490 um sit in correct physical
proportion without anything being resampled.

Ordering matters in `launch`: the landmark panel offers to place a point at the
crosshair, so the crosshair is installed before the panels are docked.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

import napari

from ..geometry import axis_tilt_degrees, tilt_displacement_mm, tilt_is_subvoxel
from ..loading import (
    DEFAULT_BUDGET_BYTES,
    Volume,
    load_series,
    sample_value_range,
)
from .controls import OrientationControls
from .crosshair import Crosshair
from .crosshairpanel import CrosshairPanel
from .info import LayerInfo
from .landmarkpanel import LandmarkPanel
from .layout import arrange, freeze_geometry
from .picking import LandmarkPicker
from .registrationpanel import RegistrationPanel
from .viewbuttons import bind_view_keys, pretty, rebuild_button_row

# Inverted grey: dense material dark on white, the convention these scans are
# usually read in. Per-layer colormaps are one click away in napari's layer
# controls if a fused look is wanted instead.
DEFAULT_COLORMAP = "gray_r"


def add_volume(viewer: "napari.Viewer", volume: Volume, index: int = 0):
    """Add one volume as an image layer placed by its own geometry.

    Args:
        viewer: The napari viewer.
        volume: The loaded volume.
        index: Position in the load order; the first is opaque, the rest blend.

    Returns:
        The image layer, with `volume`, `base_affine`, `matrix` and `nudge` in its
        metadata for the orientation controls to act on.
    """
    low, high = sample_value_range(volume)

    layer = viewer.add_image(
        volume.data,
        name=f"{volume.name} [{volume.modality}]",
        affine=volume.affine,
        # Units are per-layer; the viewer-level scale bar unit is deprecated.
        units=("mm", "mm", "mm"),
        axis_labels=("z", "y", "x"),
        contrast_limits=(low, high),
        colormap=DEFAULT_COLORMAP,
        blending="translucent" if index else "opaque",
        opacity=1.0 if index == 0 else 0.5,
    )
    # base_affine is the DICOM geometry and never changes; matrix and nudge are
    # display state the controls compose against it.
    layer.metadata["volume"] = volume
    layer.metadata["base_affine"] = volume.affine
    layer.metadata["nudge"] = np.zeros(3)
    layer.metadata["matrix"] = np.eye(3)
    return layer


def install_crosshair(viewer):
    """Add the crosshair and the two ways of setting it.

    Shift+click rather than a plain one, since a plain click on a landmarks layer
    is how a landmark gets added. `T` does the same for whoever prefers a key, and
    works whatever the selected layer is doing with the mouse.

    Args:
        viewer: The napari viewer.

    Returns:
        The `Crosshair`, or None if this napari has moved its overlay machinery, in
        which case the reason is printed and everything else carries on.
    """
    crosshair = Crosshair(viewer)
    if not crosshair.available:
        print(f"note: no crosshair -- {crosshair.reason or 'overlay machinery moved'}")
        return None

    def on_click(_source, event):
        if "Shift" in getattr(event, "modifiers", ()):
            crosshair.move_to(getattr(event, "position", None))

    viewer.mouse_drag_callbacks.append(on_click)
    try:
        viewer.bind_key("t", crosshair.mark_here, overwrite=False)
        print("  Shift+click, T mark a point the views keep")
    except (AttributeError, TypeError, ValueError):
        print("  Shift+click    mark a point the views keep")
    return crosshair


def _report_tilt(volumes) -> None:
    """Say what each volume's axis tilt costs, and mute napari's warning if it is nil.

    Scanners round their direction cosines, so a series can be a fraction of a
    degree off-axis. napari cannot slice obliquely and warns on every slice change.
    Where the resulting error is smaller than a voxel the warning is pure noise;
    where it is not, it matters and must stay visible.

    Args:
        volumes: The loaded volumes.
    """
    negligible = True
    for volume in volumes:
        tilt = axis_tilt_degrees(volume.affine)
        if tilt == 0:
            continue
        error = tilt_displacement_mm(volume.geometry)
        subvoxel = tilt_is_subvoxel(volume.geometry)
        negligible &= subvoxel
        verdict = "below one voxel" if subvoxel else "LARGER THAN ONE VOXEL"
        print(
            f"{volume.name}: axes tilted {tilt:.4f} deg; ignoring it displaces the "
            f"far corner by {error:.3f} mm ({verdict})."
        )

    if negligible:
        warnings.filterwarnings(
            "ignore", message="Non-orthogonal slicing", category=UserWarning
        )


def launch(paths, show: bool = True, budget_bytes: int = DEFAULT_BUDGET_BYTES):
    """Open one viewer holding every given series.

    Args:
        paths: Image2Dicom output directories, in chain order - moving first, fixed
            last.
        show: Enter napari's event loop. False builds the viewer and returns it,
            for inspection.
        budget_bytes: Memory per volume before it is decimated to fit.

    Returns:
        The napari viewer.

    Raises:
        LoadError: If a series will not load.
    """
    volumes = []
    for path in paths:
        print(f"loading {Path(path).name} ...")
        volume = load_series(Path(path), budget_bytes=budget_bytes)
        if volume.decimation > 1:
            full = " x ".join(str(v) for v in volume.full_shape)
            print(
                f"  decimated {volume.decimation}x to fit the budget: "
                f"{full} -> {' x '.join(str(v) for v in volume.shape)} "
                f"({volume.nbytes / 1024**2:.0f} MB)"
            )
        else:
            print(f"  full resolution ({volume.nbytes / 1024**2:.0f} MB)")
        volumes.append(volume)
    _report_tilt(volumes)

    freeze_geometry(False)

    viewer = napari.Viewer(title="ImageFusion Aligner")
    viewer.scale_bar.visible = True
    viewer.axes.visible = True

    images = [add_volume(viewer, volume, index) for index, volume in enumerate(volumes)]

    # Landmark layers go on last so they draw over every image.
    picker = LandmarkPicker(viewer)
    for index, image in enumerate(images):
        picker.attach(image, index)
    if images:
        viewer.layers.selection.active = images[0]

    controls = OrientationControls(viewer)
    # Landmarks live in the data's frame, so moving a volume moves where they show.
    controls.layer_changed.connect(picker.reposition)

    # Before the panels: the landmark panel offers to place a point at the crosshair,
    # so it needs one to hand.
    crosshair = install_crosshair(viewer)

    viewer.window.add_dock_widget(LayerInfo(viewer), name="layer info", area="left")
    viewer.window.add_dock_widget(controls, name="orientation", area="left")
    viewer.window.add_dock_widget(
        LandmarkPanel(picker, viewer, crosshair), name="landmarks", area="left"
    )
    viewer.window.add_dock_widget(
        RegistrationPanel(picker, controls), name="registration", area="left"
    )
    viewer.window.add_dock_widget(
        CrosshairPanel(crosshair, viewer), name="crosshair", area="left"
    )

    if not rebuild_button_row(viewer, crosshair):
        print("note: napari moved its viewer button row; leaving it as shipped")

    for key, description in bind_view_keys(viewer, crosshair):
        print(f"  {pretty(key):<14} {description}")

    # Pin the panel arrangement rather than inheriting last session's layout.
    arrange(viewer, area="left", verbose=True)

    viewer.reset_view()

    if show:
        napari.run()
    return viewer
