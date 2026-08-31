"""The PRODUCE phase: decide, then stream to a canonical DICOM series.

Two rules drive everything here:

1. **Integer input passes through untouched.** If the source already stores integers
   with a valid slope/intercept, the stored values are copied verbatim and the mapping
   is carried forward. There is nothing to gain by requantizing and everything to lose.
2. **Float input is quantized once, deliberately, and the mapping is recorded.** The
   slope is rounded to the precision DICOM can actually store *before* it is used, so
   the writer and any reader agree exactly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset

from .model import Characterization, VolumeCandidate

# SOP classes we emit.
SOP_CLASS_BY_MODALITY = {
    "CT": "1.2.840.10008.5.1.4.1.1.2",
    "PT": "1.2.840.10008.5.1.4.1.1.128",
    "MR": "1.2.840.10008.5.1.4.1.1.4",
    "NM": "1.2.840.10008.5.1.4.1.1.20",
}
DEFAULT_SOP_CLASS = "1.2.840.10008.5.1.4.1.1.2"

# DICOM Decimal String: at most 16 characters.
DS_MAX_LEN = 16

# Slices sampled when the value range is only needed for a display window.
SCAN_SAMPLE_SLICES = 64


def ds_decimal(value: float) -> tuple:
    """Format a float as a DICOM DS string.

    RescaleSlope is stored as a decimal *string*. Computing a slope at full
    float precision and writing a truncated string means readers recover a
    slightly different value than intended, so quantization must use the
    round-tripped one rather than the original.

    Args:
        value: The number to write.

    Returns:
        (text, round_tripped): the DS string of at most 16 characters, and the
        float a reader will recover from it.
    """
    for precision in range(15, 4, -1):
        text = f"{value:.{precision}g}"
        if len(text) <= DS_MAX_LEN:
            return text, float(text)
    text = f"{value:.5g}"
    return text, float(text)


@dataclass
class ConversionPlan:
    """Every decision, frozen before a single byte is written."""

    source_key: str
    source_label: str
    n_slices: int
    rows: int
    columns: int

    modality: str
    units: str
    output_dtype: str  # "uint16" or "int16"
    rescale_slope: float
    rescale_intercept: float
    rescale_slope_ds: str
    rescale_intercept_ds: str
    requantized: bool

    pixel_spacing: tuple  # (row_mm, col_mm)
    slice_spacing: float
    origin: tuple  # ImagePositionPatient of the first slice
    orientation: tuple  # ImageOrientationPatient
    geometry_source: str  # "centred" or "source"
    frame_of_reference_uid: Optional[str]

    # Full-range display window, in physical units, so viewers do not threshold.
    window_center: float
    window_width: float
    window_center_ds: str
    window_width_ds: str

    patient_name: str
    patient_id: str
    study_instance_uid: str = ""
    series_instance_uid: str = ""

    source_value_range: Optional[tuple] = None
    quantization_step: Optional[float] = None
    n_nonfinite: int = 0
    notes: list = field(default_factory=list)
    overrides: list = field(default_factory=list)  # {field, source, supplied, kind}

    @property
    def contradictions(self) -> list:
        """Pick out the overrides that replaced rather than filled.

        Returns:
            Overrides that replaced a value the source actually declared. These
            need explicit confirmation; filling a gap does not.
        """
        return [o for o in self.overrides if o["kind"] == "replace"]


def _centred_origin(
    orientation, n_slices, rows, columns, slice_spacing, row_mm, col_mm
) -> tuple:
    """Place the volume's centre exactly at the patient-space origin.

    Args:
        orientation: ImageOrientationPatient, six direction cosines.
        n_slices: Number of slices.
        rows: Rows per slice.
        columns: Columns per slice.
        slice_spacing: Distance between slices, in mm.
        row_mm: Distance between rows, in mm.
        col_mm: Distance between columns, in mm.

    Returns:
        ImagePositionPatient for the first slice, (x, y, z) in mm. It is the
        centre of the *first voxel*, so the volume centre sits (n - 1) / 2
        voxels along each axis - not n / 2, which would miss by half a voxel.
    """
    row_dir = np.array(orientation[:3], dtype=float)  # increasing column index
    col_dir = np.array(orientation[3:], dtype=float)  # increasing row index
    normal = np.cross(row_dir, col_dir)
    norm = np.linalg.norm(normal)
    normal = normal / norm if norm > 0 else np.array([0.0, 0.0, 1.0])

    offset = (
        row_dir * ((columns - 1) / 2.0) * col_mm
        + col_dir * ((rows - 1) / 2.0) * row_mm
        + normal * ((n_slices - 1) / 2.0) * slice_spacing
    )
    return tuple(float(-v) for v in offset)


def _scan_range(
    source, on_slice: Optional[Callable] = None, step: int = 1
) -> tuple:
    """Find the value range by streaming, never holding more than one slice.

    Args:
        source: The slice reader.
        on_slice: Called with each slice index, for progress.
        step: Sample every Nth slice. Only valid where the result is a display
            hint - a quantization mapping must see every voxel, or it clips.

    Returns:
        (vmin, vmax, n_nonfinite): the range, and how many voxels were NaN or
        infinite.

    Raises:
        ValueError: If the source holds no finite values at all.
    """
    vmin = math.inf
    vmax = -math.inf
    n_nonfinite = 0

    for index in range(0, len(source), step):
        data = np.asarray(source.raw(index))
        finite = np.isfinite(data)
        n_nonfinite += int(data.size - int(finite.sum()))
        if finite.any():
            vmin = min(vmin, float(data[finite].min()))
            vmax = max(vmax, float(data[finite].max()))
        if on_slice is not None:
            on_slice(index)

    if not math.isfinite(vmin) or not math.isfinite(vmax):
        raise ValueError("Source contains no finite values.")
    return vmin, vmax, n_nonfinite


def build_plan(
    candidate: VolumeCandidate,
    source,
    *,
    modality: Optional[str] = None,
    units: Optional[str] = None,
    spacing: Optional[tuple] = None,
    rescale_slope: Optional[float] = None,
    rescale_intercept: Optional[float] = None,
    patient_name: Optional[str] = None,
    patient_id: Optional[str] = None,
    center: bool = True,
    on_scan: Optional[Callable] = None,
) -> ConversionPlan:
    """Resolve every decision the conversion needs, before anything is written.

    Reads pixels only when a float source needs its range measured for
    quantization.

    Args:
        candidate: The volume to convert.
        source: Its slice reader.
        modality: Override for the DICOM Modality, e.g. CT or PT.
        units: Override for RescaleType, what the values mean.
        spacing: Override for voxel size, (k, j, i) in mm.
        rescale_slope: Override for the stored-to-physical slope.
        rescale_intercept: Override for its intercept. Both must be given
            together.
        patient_name: PatientName to write.
        patient_id: PatientID to write.
        center: Place the volume's centre at the origin, regenerating the frame
            of reference. False keeps the source's own geometry.
        on_scan: Called with each slice index while measuring a float range.

    Returns:
        The `ConversionPlan`, including any overrides and contradictions for the
        caller to confirm.

    Raises:
        ValueError: If something required is neither declared nor supplied, or a
            supplied rescale mapping cannot represent the data.
    """
    c: Characterization = candidate.characterization
    notes: list = []
    overrides: list = []

    def record(name, source_value, supplied):
        """Log a supplied value, distinguishing filling a gap from replacing one."""
        if supplied is None:
            return
        kind = "fill" if source_value in (None, "arbitrary") else "replace"
        if source_value is not None and str(source_value) == str(supplied):
            return  # identical; not an override at all
        overrides.append(
            {
                "field": name,
                "source": source_value,
                "supplied": supplied,
                "kind": kind,
            }
        )

    record("modality", c.modality, modality)
    resolved_modality = modality or c.modality
    if resolved_modality is None:
        raise ValueError("Modality could not be determined; set overrides.modality.")

    record("units", c.units, units)
    resolved_units = units or c.units
    if resolved_units is None:
        raise ValueError("Units could not be determined; set overrides.units.")

    if spacing is not None:
        source_spacing = (
            (c.slice_spacing, c.pixel_spacing[0], c.pixel_spacing[1])
            if c.has_spacing
            else None
        )
        record("spacing_mm", source_spacing, tuple(spacing))
        slice_spacing, row_mm, col_mm = spacing
        spacing_note = "supplied in the job file"
    elif c.has_spacing:
        slice_spacing = c.slice_spacing
        row_mm, col_mm = c.pixel_spacing
        spacing_note = f"from source ({c.spacing_source})"
    else:
        raise ValueError(
            "Voxel spacing is unknown; set overrides.spacing_mm = [z, y, x]."
        )
    notes.append(f"Spacing {spacing_note}.")

    n_slices = len(source)
    probe_slice = np.asarray(source.raw(0))
    rows, columns = int(probe_slice.shape[0]), int(probe_slice.shape[1])

    # --- geometry -------------------------------------------------------
    source_origin = candidate.detail.get("origin")
    orientation = candidate.detail.get("orientation") or (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    frame_uid = candidate.detail.get("frame_of_reference") or None

    if center or source_origin is None:
        origin = _centred_origin(
            orientation, n_slices, rows, columns, slice_spacing, row_mm, col_mm
        )
        geometry_source = "centred on (0, 0, 0)"
        if source_origin is None:
            notes.append(
                "Source carried no position; the volume was centred on the world "
                "origin and a new FrameOfReferenceUID generated."
            )
        else:
            notes.append(
                "Volume re-centred on (0, 0, 0); the source position was discarded. "
                "FrameOfReferenceUID regenerated, because re-centring breaks any "
                "co-registration the shared frame expressed. Set output.center: "
                "false to keep the scanner's own geometry instead."
            )
        frame_uid = pydicom.uid.generate_uid()
    else:
        origin = source_origin
        geometry_source = "source"
        notes.append(
            "Source geometry preserved verbatim, including FrameOfReferenceUID - "
            "series sharing a frame are already co-registered."
        )

    # --- pixel representation -------------------------------------------
    source_dtype = np.dtype(probe_slice.dtype)
    quantization_step = None
    is_integer = np.issubdtype(source_dtype, np.integer)

    # Float input needs an exact range to choose a quantization - a missed extreme
    # would clip real data. Integer input is passed through untouched, so its range
    # only feeds the display window, and sampling is enough. That matters: an exact
    # scan doubles the read cost on a 2000-slice volume.
    scan_step = max(1, len(source) // SCAN_SAMPLE_SLICES) if is_integer else 1
    vmin, vmax, n_nonfinite = _scan_range(source, on_slice=on_scan, step=scan_step)
    source_range = (vmin, vmax)

    supplied_mapping = rescale_slope is not None and rescale_intercept is not None
    if supplied_mapping:
        record("rescale_slope", c.rescale_slope, rescale_slope)
        record("rescale_intercept", c.rescale_intercept, rescale_intercept)

    if np.issubdtype(source_dtype, np.integer):
        requantized = False
        output_dtype = "uint16" if source_dtype.kind == "u" else "int16"
        if supplied_mapping:
            slope_ds, slope = ds_decimal(float(rescale_slope))
            intercept_ds, intercept = ds_decimal(float(rescale_intercept))
            notes.append(
                "Integer source: stored values copied verbatim with a supplied "
                "calibration. Only the interpretation changed, so this is still "
                "lossless."
            )
        else:
            slope = c.rescale_slope if c.rescale_slope is not None else 1.0
            intercept = c.rescale_intercept if c.rescale_intercept is not None else 0.0
            slope_ds, slope = ds_decimal(float(slope))
            intercept_ds, intercept = ds_decimal(float(intercept))
            notes.append(
                "Integer source: stored values copied verbatim, original rescale "
                "mapping carried forward. Lossless."
            )
    else:
        requantized = True
        output_dtype = "uint16"

        if supplied_mapping:
            slope_ds, slope = ds_decimal(float(rescale_slope))
            intercept_ds, intercept = ds_decimal(float(rescale_intercept))
            representable_max = intercept + slope * 65535
            if vmin < intercept or vmax > representable_max:
                raise ValueError(
                    f"The supplied mapping represents "
                    f"[{intercept:.6g}, {representable_max:.6g}] but the data spans "
                    f"[{vmin:.6g}, {vmax:.6g}]. Converting would clip real values. "
                    "Widen the mapping or remove it to fit the true range."
                )
            notes.append(
                f"Float source quantized using the supplied mapping; step "
                f"{slope:.6g} per level."
            )
        else:
            span = vmax - vmin
            raw_slope = span / 65535.0 if span > 0 else 1.0
            slope_ds, slope = ds_decimal(raw_slope)
            intercept_ds, intercept = ds_decimal(vmin)
            notes.append(
                f"Float source quantized to uint16 over its true range "
                f"[{vmin:.6g}, {vmax:.6g}]; step {slope:.6g} per level."
            )
        quantization_step = slope

        if n_nonfinite:
            notes.append(
                f"{n_nonfinite} non-finite voxel(s) were clamped to the minimum."
            )

    # Full-range display window, in physical units. Without these tags a viewer
    # picks its own range and the image comes up thresholded.
    if requantized:
        physical_min, physical_max = vmin, vmax
    else:
        physical_min = slope * vmin + intercept
        physical_max = slope * vmax + intercept

    window_width = max(physical_max - physical_min, 1e-6)
    window_center = (physical_min + physical_max) / 2.0
    window_center_ds, window_center = ds_decimal(window_center)
    window_width_ds, window_width = ds_decimal(window_width)
    if scan_step > 1:
        n_sampled = len(range(0, n_slices, scan_step))
        notes.append(
            f"Display window [{physical_min:.6g}, {physical_max:.6g}] "
            f"{resolved_units}, from {n_sampled} sampled slices of {n_slices}. "
            "The window is a display hint only; the voxels are untouched."
        )
    else:
        notes.append(
            f"Display window covers the full data range "
            f"[{physical_min:.6g}, {physical_max:.6g}] {resolved_units}."
        )

    return ConversionPlan(
        source_key=candidate.key,
        source_label=candidate.label,
        n_slices=n_slices,
        rows=rows,
        columns=columns,
        modality=resolved_modality,
        units=resolved_units,
        output_dtype=output_dtype,
        rescale_slope=slope,
        rescale_intercept=intercept,
        rescale_slope_ds=slope_ds,
        rescale_intercept_ds=intercept_ds,
        requantized=requantized,
        pixel_spacing=(row_mm, col_mm),
        slice_spacing=slice_spacing,
        origin=origin,
        orientation=orientation,
        geometry_source=geometry_source,
        frame_of_reference_uid=frame_uid,
        window_center=window_center,
        window_width=window_width,
        window_center_ds=window_center_ds,
        window_width_ds=window_width_ds,
        patient_name=patient_name or "Specimen",
        patient_id=patient_id or "UNKNOWN",
        study_instance_uid=pydicom.uid.generate_uid(),
        series_instance_uid=pydicom.uid.generate_uid(),
        source_value_range=source_range,
        quantization_step=quantization_step,
        n_nonfinite=n_nonfinite,
        notes=notes,
        overrides=overrides,
    )


def _encode(data: np.ndarray, plan: ConversionPlan) -> np.ndarray:
    """Turn one source slice into the integers that will be stored.

    Args:
        data: One 2D slice of source values.
        plan: The resolved plan, which says whether to requantize.

    Returns:
        The slice as int16 or uint16. An integer source is copied through
        unchanged; a float one is mapped through the plan's rescale pair.
    """
    if not plan.requantized:
        target = np.uint16 if plan.output_dtype == "uint16" else np.int16
        return np.asarray(data).astype(target, copy=False)

    values = np.asarray(data, dtype=np.float64)
    values = np.nan_to_num(
        values,
        nan=plan.rescale_intercept,
        posinf=plan.rescale_intercept + plan.rescale_slope * 65535,
        neginf=plan.rescale_intercept,
    )
    stored = np.rint((values - plan.rescale_intercept) / plan.rescale_slope)
    return np.clip(stored, 0, 65535).astype(np.uint16)


def _slice_position(plan: ConversionPlan, index: int) -> list:
    """Locate one slice in patient space.

    Args:
        plan: The resolved plan, holding the origin and orientation.
        index: Slice number.

    Returns:
        ImagePositionPatient, (x, y, z) in mm, stepping along the slice normal.
    """
    row = np.array(plan.orientation[:3], dtype=float)
    col = np.array(plan.orientation[3:], dtype=float)
    normal = np.cross(row, col)
    norm = np.linalg.norm(normal)
    if norm > 0:
        normal = normal / norm
    else:
        normal = np.array([0.0, 0.0, 1.0])
    position = np.array(plan.origin, dtype=float) + normal * (index * plan.slice_spacing)
    return [float(v) for v in position]


def write_series(
    plan: ConversionPlan,
    source,
    out_dir: Path,
    on_slice: Optional[Callable] = None,
) -> Path:
    """Write the volume out as one single-frame DICOM per slice.

    Streams: peak memory is one slice.

    Args:
        plan: The resolved plan.
        source: The slice reader.
        out_dir: Directory to write into; created if absent.
        on_slice: Called with each slice index, for progress.

    Returns:
        The directory written.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sop_class = SOP_CLASS_BY_MODALITY.get(plan.modality, DEFAULT_SOP_CLASS)
    signed = plan.output_dtype == "int16"

    for index in range(plan.n_slices):
        stored = _encode(source.raw(index), plan)

        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = sop_class
        meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
        meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
        meta.ImplementationClassUID = pydicom.uid.PYDICOM_IMPLEMENTATION_UID

        path = out_dir / f"{plan.modality}_{index + 1:05d}.dcm"
        ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)

        # pydicom 2.x requires these explicitly: without them it writes implicit VR
        # while the file meta declares explicit, producing subtly corrupt files.
        # pydicom 3.x infers them from the transfer syntax and ignores these.
        ds.is_little_endian = True
        ds.is_implicit_VR = False

        ds.SOPClassUID = sop_class
        ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
        ds.StudyInstanceUID = plan.study_instance_uid
        ds.SeriesInstanceUID = plan.series_instance_uid
        ds.FrameOfReferenceUID = plan.frame_of_reference_uid

        ds.PatientName = plan.patient_name
        ds.PatientID = plan.patient_id
        ds.Modality = plan.modality
        ds.SeriesNumber = 1
        ds.InstanceNumber = index + 1
        ds.SeriesDescription = plan.source_label[:64]
        ds.ImageType = ["DERIVED", "SECONDARY", "AXIAL"]
        ds.ConversionType = "WSD"

        ds.ImagePositionPatient = _slice_position(plan, index)
        ds.ImageOrientationPatient = [float(v) for v in plan.orientation]
        ds.PixelSpacing = [float(plan.pixel_spacing[0]), float(plan.pixel_spacing[1])]
        ds.SliceThickness = float(plan.slice_spacing)
        ds.SpacingBetweenSlices = float(plan.slice_spacing)

        ds.Rows = plan.rows
        ds.Columns = plan.columns
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 1 if signed else 0

        ds.RescaleSlope = plan.rescale_slope_ds
        ds.RescaleIntercept = plan.rescale_intercept_ds
        ds.RescaleType = plan.units

        # Without an explicit window, viewers invent a narrow one and the image
        # opens thresholded. This covers the full data range.
        ds.WindowCenter = plan.window_center_ds
        ds.WindowWidth = plan.window_width_ds
        ds.WindowCenterWidthExplanation = "Full range"
        ds.VOILUTFunction = "LINEAR"

        # SmallestImagePixelValue / LargestImagePixelValue are deliberately not
        # written. They are per-image by definition, so an edge slice of mostly air
        # reports a range like 0-24; a viewer that takes a volume's range from the
        # first slice's tags would window on that instead of the real range.

        ds.PixelData = stored.tobytes()

        ds.save_as(str(path))

        if on_slice is not None:
            on_slice(index)

    return out_dir
