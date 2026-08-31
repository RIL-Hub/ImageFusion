"""Fitting a rigid transform to paired landmarks, and composing the chain.

Arithmetic on millimetre points. It imports nothing from the viewer and knows nothing
about voxels: give it a session file and it re-runs every registration with napari
uninstalled, which is the point.

The fit is Umeyama's closed form - the SVD of the cross-covariance, with a sign
correction that keeps the rotation proper. Without that correction a noisy or
near-degenerate set can produce a reflection, which fits the points beautifully and
maps the volume to its mirror image.

**Scale is measured and reported but never applied.** Both volumes are already
calibrated in true millimetres by their DICOM spacing, so the correct scale is exactly
1. A fit that wants 1.04 is evidence of a spacing or units error upstream, and
absorbing it into the transform would hide the very thing worth seeing.

Errors accumulate along the chain, so per-link and end-to-end residuals are reported
separately. The composed transform is an estimate, not a measurement, unless a landmark
is visible in both end volumes.

Pure: no napari, no Qt.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .landmarks import Session

# Three non-collinear points are the minimum that fix a rotation.
MIN_POINTS = 3

# Ratio of second to largest spread of the source points. Below this they are
# effectively on one line, and rotation about that line is unconstrained: the fit
# still succeeds and still reports a tiny residual, which is what makes it dangerous.
COLLINEAR_TOLERANCE = 1e-3

# How far the measured scale may sit from 1 before it is worth saying so.
SCALE_TOLERANCE = 1e-3


class SolveError(RuntimeError):
    """A registration cannot be fitted from what has been picked."""


@dataclass(frozen=True)
class Rigid:
    """A rotation and a translation, in patient millimetres.

    Attributes:
        rotation: 3x3 proper orthogonal matrix.
        translation: (3,) offset in mm, applied after the rotation.
    """

    rotation: np.ndarray
    translation: np.ndarray

    @staticmethod
    def identity() -> "Rigid":
        """Build the transform that moves nothing.

        Returns:
            A `Rigid` with an identity rotation and zero translation.
        """
        return Rigid(np.eye(3), np.zeros(3))

    @property
    def matrix(self) -> np.ndarray:
        """Express the transform in homogeneous form.

        Returns:
            4x4 matrix, for composing with DICOM geometry.
        """
        matrix = np.eye(4)
        matrix[:3, :3] = self.rotation
        matrix[:3, 3] = self.translation
        return matrix

    def apply(self, points) -> np.ndarray:
        """Move points by this transform.

        Args:
            points: (N, 3) array in mm, or a single (3,) point.

        Returns:
            The moved points, in the same shape as `points`.
        """
        points = np.asarray(points, dtype=float)
        single = points.ndim == 1
        moved = np.atleast_2d(points) @ self.rotation.T + self.translation
        return moved[0] if single else moved

    def inverse(self) -> "Rigid":
        """Build the transform that undoes this one.

        Returns:
            A `Rigid` such that applying both leaves points where they started.
        """
        rotation = self.rotation.T
        return Rigid(rotation, -rotation @ self.translation)

    def then(self, other: "Rigid") -> "Rigid":
        """Compose this transform with one applied after it.

        Args:
            other: The transform to apply second.

        Returns:
            A single `Rigid` equivalent to applying this one, then `other`.
        """
        return Rigid(
            other.rotation @ self.rotation,
            other.rotation @ self.translation + other.translation,
        )

    @property
    def angle_degrees(self) -> float:
        """Measure how far the rotation turns, about whatever axis it turns about.

        Returns:
            The angle in degrees, 0 to 180.
        """
        cosine = (np.trace(self.rotation) - 1.0) / 2.0
        return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


@dataclass(frozen=True)
class Fit:
    """One solved link, with everything needed to judge whether to trust it.

    Attributes:
        transform: The fitted rigid transform, source frame to target frame.
        ids: Landmark ids used, in the order the point arrays hold them.
        residuals: (N,) per-point distance in mm remaining after fitting.
        scale: The similarity scale the points imply. Measured, never applied.
        spread: Singular values of the centred source points, largest first - how
            far the landmarks reach along their own principal directions, in mm.
    """

    transform: Rigid
    ids: Tuple[int, ...]
    residuals: np.ndarray
    scale: float
    spread: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    @property
    def rmsd(self) -> float:
        """The headline accuracy number.

        Returns:
            Root-mean-square residual in mm, or 0 with no points.
        """
        return float(np.sqrt(np.mean(self.residuals**2))) if len(self.residuals) else 0.0

    @property
    def worst(self) -> float:
        """Find the single worst-fitting landmark, which a good average can hide.

        Returns:
            Largest residual in mm, or 0 with no points.
        """
        return float(np.max(self.residuals)) if len(self.residuals) else 0.0

    @property
    def collinear(self) -> bool:
        """Test whether the landmarks lie on one line, leaving rotation about it free.

        Returns:
            True if the fit should not be trusted despite its residual.
        """
        # The *second* spread against the first, never the third. N centred points
        # span at most N-1 dimensions, so three points always give a third singular
        # value of exactly zero. Coplanar is fine; collinear is not.
        return self.spread[0] <= 0.0 or (self.spread[1] / self.spread[0]) < COLLINEAR_TOLERANCE

    @property
    def scale_is_suspicious(self) -> bool:
        """Test whether the measured scale points at a bug upstream.

        Returns:
            True if it differs from 1 by more than `SCALE_TOLERANCE`; both volumes
            are already in true mm, so anything else is a spacing or units error.
        """
        return abs(self.scale - 1.0) > SCALE_TOLERANCE


def kabsch(source, target) -> Fit:
    """Fit the rigid transform taking one point set onto another.

    Args:
        source: (N, 3) points in mm.
        target: (N, 3) points in mm, paired row for row with `source`. Pairing is
            the caller's job; `Session.paired` does it by landmark id.

    Returns:
        A `Fit` with an empty `ids`, which `solve_link` fills in.

    Raises:
        SolveError: If the two sets differ in length, or hold fewer than
            `MIN_POINTS` points.
    """
    source = np.asarray(source, dtype=float).reshape(-1, 3)
    target = np.asarray(target, dtype=float).reshape(-1, 3)

    if source.shape != target.shape:
        raise SolveError(
            f"{len(source)} source points against {len(target)} target points; "
            "landmarks must be paired before fitting."
        )
    if len(source) < MIN_POINTS:
        raise SolveError(
            f"{len(source)} paired landmark(s); at least {MIN_POINTS} "
            "non-collinear points are needed to fix a rotation."
        )

    source_centre = source.mean(axis=0)
    target_centre = target.mean(axis=0)
    centred_source = source - source_centre
    centred_target = target - target_centre

    covariance = (centred_target.T @ centred_source) / len(source)
    left, singular, right = np.linalg.svd(covariance)

    # Keep the result a rotation. Allowing a reflection would fit the points just as
    # well and map the volume to its mirror image.
    correction = np.eye(3)
    if np.linalg.det(left) * np.linalg.det(right) < 0:
        correction[2, 2] = -1.0

    rotation = left @ correction @ right
    translation = target_centre - rotation @ source_centre
    transform = Rigid(rotation, translation)

    variance = float(np.mean(np.sum(centred_source**2, axis=1)))
    scale = (
        float(np.trace(np.diag(singular) @ correction) / variance)
        if variance > 0
        else 1.0
    )

    residuals = np.linalg.norm(transform.apply(source) - target, axis=1)
    spread = tuple(float(v) for v in np.linalg.svd(centred_source, compute_uv=False))

    return Fit(
        transform=transform,
        ids=(),
        residuals=residuals,
        scale=scale,
        spread=spread,
    )


@dataclass
class ChainSolution:
    """Every link of the chain solved, plus what the composition costs.

    Attributes:
        fixed: The volume nothing moves relative to; PET.
        links: The (source, target) pairs solved, in chain order.
        skipped: Volumes loaded but carrying no landmarks, so not registered.
        fits: Each link's `Fit`, keyed by its pair.
        to_fixed: Each volume's composed transform into the fixed frame.
        end_to_end_ids: Landmark ids visible in both the first volume and `fixed`.
        end_to_end_residuals: (N,) mm errors for those, or None if there are none.
    """

    fixed: str
    links: List[Tuple[str, str]] = field(default_factory=list)
    skipped: Tuple[str, ...] = ()
    fits: Dict[Tuple[str, str], Fit] = field(default_factory=dict)
    to_fixed: Dict[str, Rigid] = field(default_factory=dict)
    end_to_end_ids: Tuple[int, ...] = ()
    end_to_end_residuals: Optional[np.ndarray] = None

    @property
    def end_to_end_rmsd(self) -> Optional[float]:
        """Measure the composed accuracy, where it can be measured at all.

        Returns:
            Root-mean-square mm error over the shared landmarks, or None if no
            landmark spans both ends - in which case the composition is an estimate.
        """
        if self.end_to_end_residuals is None or not len(self.end_to_end_residuals):
            return None
        return float(np.sqrt(np.mean(self.end_to_end_residuals**2)))

    @property
    def worst_link(self) -> Optional[Tuple[str, str]]:
        """Identify the link fitting least well.

        Returns:
            Its (source, target) pair, or None if nothing was solved.
        """
        if not self.fits:
            return None
        return max(self.fits, key=lambda link: self.fits[link].rmsd)


def describe(solution: "ChainSolution") -> List[str]:
    """Render a solution as plain text.

    Text rather than Qt so the same report reaches a panel and a terminal, and so a
    headless run can say what it found.

    Args:
        solution: A solved chain.

    Returns:
        Lines to print, including warnings for collinear landmarks and a suspicious
        scale.
    """
    lines: List[str] = []
    for source, target in solution.links:
        fit = solution.fits[(source, target)]
        shift = float(np.linalg.norm(fit.transform.translation))
        lines.append(f"{source} -> {target}")
        lines.append(
            f"   {len(fit.ids)} landmarks    rms {fit.rmsd:.3f} mm"
            f"    worst {fit.worst:.3f} mm"
        )
        lines.append(
            f"   rotates {fit.transform.angle_degrees:.2f}deg    shifts {shift:.2f} mm"
        )
        if fit.collinear:
            lines.append(
                "   WARNING  landmarks lie on one line; rotation about it is "
                "unconstrained and the small residual means nothing"
            )
        if fit.scale_is_suspicious:
            lines.append(
                f"   WARNING  measured scale {fit.scale:.4f}, expected 1.0000; "
                "both volumes should already be in true mm"
            )
        lines.append("")

    if solution.skipped:
        lines.append(
            "passed over (nothing picked in them): "
            + ", ".join(solution.skipped)
        )
        lines.append("")

    end = solution.end_to_end_rmsd
    if end is None:
        first = solution.links[0][0] if solution.links else "?"
        lines.append(
            f"end to end   not measurable - no landmark picked in both {first} "
            f"and {solution.fixed}, so the composed transform is an estimate"
        )
    else:
        lines.append(
            f"end to end   rms {end:.3f} mm over "
            f"{len(solution.end_to_end_ids)} shared landmarks"
        )
    return lines


def solve_link(session: Session, source: str, target: str) -> Fit:
    """Fit one link of the chain from a session's landmarks.

    Args:
        session: The session holding the landmarks.
        source: Name of the moving volume.
        target: Name of the volume it registers to.

    Returns:
        The `Fit`, with the landmark ids used.

    Raises:
        SolveError: If fewer than `MIN_POINTS` landmarks are shared. Landmarks pair
            by id, so a point picked in only one volume does not count.
    """
    ids, source_points, target_points = session.paired(source, target)
    if len(ids) < MIN_POINTS:
        raise SolveError(
            f"{source} -> {target}: {len(ids)} landmark(s) picked in both volumes; "
            f"at least {MIN_POINTS} are needed. Landmarks are paired by id, so a "
            "point picked in only one volume does not count."
        )
    fit = kabsch(source_points, target_points)
    return Fit(
        transform=fit.transform,
        ids=ids,
        residuals=fit.residuals,
        scale=fit.scale,
        spread=fit.spread,
    )


def solve_session(session: Session) -> ChainSolution:
    """Solve every link, and compose each volume's transform into the fixed frame.

    Volumes carrying no landmarks are passed over rather than blocking the solve, so
    the last volume *holding landmarks* is the fixed frame.

    Args:
        session: The session to solve.

    Returns:
        The `ChainSolution`.

    Raises:
        SolveError: If fewer than two volumes hold landmarks, or any link cannot be
            fitted.
        LandmarkError: If the session does not validate.
    """
    session.validate()
    active = session.active_chain()
    if len(active) < 2:
        raise SolveError(
            "A chain needs at least two volumes holding landmarks; this session has "
            f"{len(active)} of {len(session.chain)}."
        )

    links = list(session.links())
    fits = {link: solve_link(session, *link) for link in links}

    # Walk backwards from the fixed volume, accumulating as we go.
    fixed = active[-1]
    to_fixed: Dict[str, Rigid] = {fixed: Rigid.identity()}
    running = Rigid.identity()
    for source, target in reversed(links):
        running = fits[(source, target)].transform.then(running)
        to_fixed[source] = running

    solution = ChainSolution(
        fixed=fixed,
        links=links,
        fits=fits,
        to_fixed=to_fixed,
        skipped=session.skipped(),
    )

    # End-to-end accuracy is only measurable where the same landmark was picked in
    # both the moving end and the fixed end.
    first = active[0]
    ids, moving_points, fixed_points = session.paired(first, fixed)
    if len(ids):
        predicted = to_fixed[first].apply(moving_points)
        solution.end_to_end_ids = ids
        solution.end_to_end_residuals = np.linalg.norm(
            predicted - fixed_points, axis=1
        )

    return solution
