"""Independent read-back of a written series.

Verification deliberately does not trust the writer's own bookkeeping. It re-runs the
validator against the output directory - dogfooding the same code that inspects real
scanner data - and then compares physical values against the source.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pydicom

from . import discovery
from .writer import ConversionPlan


@dataclass
class VerifyResult:
    """What reading the written series back proved.

    Attributes:
        passed: False as soon as any check fails.
        checks: (name, ok, detail) for each check run.
        max_abs_error: Largest difference found between a source voxel and
            the value the written file reads back as.
        tolerance: The error budget that was allowed.
    """

    passed: bool = True
    checks: list = field(default_factory=list)
    max_abs_error: float = 0.0
    tolerance: float = 0.0

    def add(self, name: str, ok: bool, detail: str = ""):
        """Record one check, failing the whole result if it did not hold.

        Args:
            name: What was checked.
            ok: Whether it held.
            detail: What was actually found.
        """
        self.checks.append((name, ok, detail))
        if not ok:
            self.passed = False


def _spread(n: int, k: int) -> list:
    """Choose evenly spaced indices, for sampling slices rather than all.

    Args:
        n: How many there are.
        k: How many to sample.

    Returns:
        Up to `k` indices spread across ``range(n)``, or all of them if
        there are fewer than `k`.
    """
    if n <= k:
        return list(range(n))
    step = n / k
    return [min(int(i * step), n - 1) for i in range(k)]


def verify_series(
    plan: ConversionPlan, source, out_dir: Path, n_sample_slices: int = 5
) -> VerifyResult:
    """Read the written series back and check it says what it should.

    Args:
        plan: The conversion that was carried out.
        source: The original image source, for comparing voxels.
        out_dir: The directory just written.
        n_sample_slices: How many slices to compare voxel for voxel.

    Returns:
        A `VerifyResult` naming every check and whether it held.
    """
    result = VerifyResult()

    # --- structural: does our own validator see one clean volume? ---------
    probe = discovery.probe(out_dir)
    result.add(
        "resolves to exactly one volume",
        probe.n_candidates == 1,
        f"found {probe.n_candidates}",
    )
    if probe.n_candidates != 1:
        return result

    written = probe.candidates[0]
    c = written.characterization

    expected_shape = (plan.n_slices, plan.rows, plan.columns)
    result.add("shape round-trips", c.shape == expected_shape, f"{c.shape} vs {expected_shape}")

    result.add(
        "slice spacing round-trips",
        c.slice_spacing is not None
        and abs(c.slice_spacing - plan.slice_spacing) < 1e-6,
        f"{c.slice_spacing} vs {plan.slice_spacing}",
    )
    result.add(
        "pixel spacing round-trips",
        c.pixel_spacing is not None
        and abs(c.pixel_spacing[0] - plan.pixel_spacing[0]) < 1e-6
        and abs(c.pixel_spacing[1] - plan.pixel_spacing[1]) < 1e-6,
        f"{c.pixel_spacing} vs {plan.pixel_spacing}",
    )
    result.add("modality round-trips", c.modality == plan.modality, f"{c.modality}")
    result.add(
        "rescale mapping round-trips",
        c.rescale_slope is not None
        and abs(c.rescale_slope - plan.rescale_slope) < 1e-12
        and abs(c.rescale_intercept - plan.rescale_intercept) < 1e-9,
        f"slope {c.rescale_slope} vs {plan.rescale_slope}, "
        f"intercept {c.rescale_intercept} vs {plan.rescale_intercept}",
    )
    result.add(
        "ordering is geometric",
        c.order_source == "ImagePositionPatient",
        c.order_source,
    )
    if plan.geometry_source.startswith("centred"):
        first = pydicom.dcmread(str(written.files[0]), stop_before_pixels=True)
        last = pydicom.dcmread(str(written.files[-1]), stop_before_pixels=True)

        # ImagePositionPatient is the centre of each slice's CORNER voxel, so the
        # midpoint of the first and last only centres the slice axis. The in-plane
        # half-extents have to be added along the orientation cosines.
        iop = [float(v) for v in first.ImageOrientationPatient]
        row_dir = np.array(iop[:3])  # direction of increasing column index
        col_dir = np.array(iop[3:])  # direction of increasing row index
        row_mm, col_mm = (float(v) for v in first.PixelSpacing)

        slice_axis_centre = (
            np.array([float(v) for v in first.ImagePositionPatient])
            + np.array([float(v) for v in last.ImagePositionPatient])
        ) / 2.0
        centre = (
            slice_axis_centre
            + row_dir * ((int(first.Columns) - 1) / 2.0) * col_mm
            + col_dir * ((int(first.Rows) - 1) / 2.0) * row_mm
        )

        offset = float(np.max(np.abs(centre)))
        result.add(
            "volume centre is at the origin",
            offset < 1e-6,
            f"max |component| {offset:.3g} mm",
        )

    result.add(
        "frame of reference preserved",
        (written.detail.get("frame_of_reference") or None)
        == (plan.frame_of_reference_uid or None),
        written.detail.get("frame_of_reference", ""),
    )
    result.add(
        "no blocking issues on output",
        not written.errors,
        "; ".join(i.code for i in written.errors),
    )

    # --- numeric: do physical values survive? ----------------------------
    tolerance = (
        0.0 if not plan.requantized else plan.rescale_slope / 2.0 + 1e-9
    )
    result.tolerance = tolerance

    max_error = 0.0
    for index in _spread(plan.n_slices, n_sample_slices):
        original = np.asarray(source.raw(index), dtype=np.float64)
        if plan.requantized:
            original = np.nan_to_num(
                original,
                nan=plan.rescale_intercept,
                posinf=plan.rescale_intercept + plan.rescale_slope * 65535,
                neginf=plan.rescale_intercept,
            )
            original = np.clip(
                original,
                plan.rescale_intercept,
                plan.rescale_intercept + plan.rescale_slope * 65535,
            )
        else:
            original = original * plan.rescale_slope + plan.rescale_intercept

        ds = pydicom.dcmread(str(written.files[index]))
        recovered = (
            ds.pixel_array.astype(np.float64) * plan.rescale_slope
            + plan.rescale_intercept
        )

        if recovered.shape != original.shape:
            result.add(
                f"slice {index} shape",
                False,
                f"{recovered.shape} vs {original.shape}",
            )
            continue

        max_error = max(max_error, float(np.max(np.abs(recovered - original))))

    result.max_abs_error = max_error
    result.add(
        "physical values recover within tolerance",
        max_error <= tolerance,
        f"max |error| {max_error:.6g}, tolerance {tolerance:.6g}",
    )

    return result
