"""Reconfiguration correlations and exponential times from cached trajectories.

No DCD reading and no mdsim import is needed here. Result objects are duck-typed.
Times are nanoseconds. Python >=3.9; NumPy, SciPy and Matplotlib are required.

Definitions
-----------
Scalar: C(t) = <delta x(s) delta x(s+t)> / <delta x**2>.
Torsion: apply the same connected vector correlation to (cos(theta), sin(theta)).
Means are estimated separately for each chain/residue over the selected window.
Combined torsions average their separately normalized correlations (equal weight
by default). This is a local feature-memory measure, NOT a joint-state survival
probability, a first-passage time, or an exhaustive conformational mixing time.

Each chain/residue is fitted separately. Residue times are averaged within each
chain, then chain means are averaged; residues are never counted as independent
replicates for the overall SEM. Chain SEM assumes independent chains and may be
optimistic for interacting chains in a single condensate.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from scipy.fft import irfft, next_fast_len, rfft
from scipy.ndimage import median_filter
from scipy.optimize import least_squares
from scipy.special import gamma, gammaincc

__all__ = [
    "ReconfigurationCorrelationResult",
    "ReconfigurationDecayFit",
    "ReconfigurationTimeResult",
    "calculate_reconfiguration_correlations",
    "fit_reconfiguration_times",
    "plot_reconfiguration_correlations",
    "plot_reconfiguration_times",
]

_ANGLES = frozenset(("phi", "psi", "omega", "chi1", "chi2"))


def _field(obj: Any, key: str, default: Any = None) -> Any:
    return obj.get(key, default) if isinstance(obj, Mapping) else getattr(obj, key, default)


def _integer(value: Any, name: str, minimum: Optional[int] = None) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer")
    try:
        number = float(value)
    except (ValueError, TypeError) as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if not np.isfinite(number) or not number.is_integer():
        raise ValueError(f"{name} must be an integer")
    result = int(number)
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def _time(value: Any, name: str) -> float:
    if hasattr(value, "value_in_unit"):
        from openmm.unit import nanoseconds

        value = value.value_in_unit(nanoseconds)
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _left_frame(time: float, dt: float) -> int:
    ratio = time / dt
    nearest = round(ratio)
    return (
        int(nearest)
        if math.isclose(ratio, nearest, rel_tol=1e-12, abs_tol=1e-12)
        else math.ceil(ratio)
    )


def _stats(
    values: np.ndarray, axis: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Finite-only mean/SD/SEM/count, with undefined uncertainty for N<2."""
    array = np.asarray(values, dtype=float)
    good = np.isfinite(array)
    count = good.sum(axis=axis)
    mean = np.full(np.shape(count), np.nan, dtype=float)
    np.divide(np.where(good, array, 0.0).sum(axis=axis), count, out=mean, where=count > 0)
    deviations = np.where(good, array - np.expand_dims(mean, axis), 0.0)
    variance = np.full_like(mean, np.nan)
    np.divide((deviations**2).sum(axis=axis), count - 1, out=variance, where=count > 1)
    sd = np.sqrt(variance)
    sem = np.full_like(mean, np.nan)
    np.divide(sd, np.sqrt(count), out=sem, where=count > 1)
    return mean, sd, sem, count


@dataclass(frozen=True)
class ReconfigurationCorrelationResult:
    """ACFs with axes (chain, residue/site, lag).

    Scalar/Rg results have one site and residue_numbers=None. Torsion results
    use the union of selected PDB residue numbers; missing sites remain NaN.
    component_variance has axes (chain, site, component); for torsions this is
    1-|<exp(i theta)>|**2, evaluated stably from centered sin/cos coordinates.
    circular_plateau is |<exp(i theta)>|**2, not a kinetic plateau estimate.
    """

    lag_time_ns: np.ndarray
    correlation: np.ndarray
    pair_counts: np.ndarray
    chain_labels: tuple[str, ...]
    residue_numbers: Optional[np.ndarray]
    observables: tuple[str, ...]
    kind: str
    component_variance: np.ndarray
    circular_plateau: np.ndarray
    component_status: np.ndarray
    dt_ns: float
    kinetic_scale: float
    n_frames: int
    frame_start: int
    frame_stop: int
    time_origin_ns: float
    max_contiguous_time_ns: float
    torsion_weighting: str
    require_all_angles: bool
    min_pairs: int
    contiguous_time_per_site_ns: Optional[np.ndarray] = None
    dataset_label: str = "data"

    @property
    def n_chains(self) -> int:
        return len(self.chain_labels)

    @property
    def correlation_per_chain(self) -> np.ndarray:
        return _stats(self.correlation, axis=1)[0]

    @property
    def mean(self) -> np.ndarray:
        return _stats(self.correlation_per_chain, axis=0)[0]

    @property
    def std(self) -> np.ndarray:
        return _stats(self.correlation_per_chain, axis=0)[1]

    @property
    def sem(self) -> np.ndarray:
        return _stats(self.correlation_per_chain, axis=0)[2]


@dataclass(frozen=True)
class ReconfigurationDecayFit:
    """One fit; numerical estimates and quality flags are retained separately.

    tau_ns is the amplitude-normalized fitted integral. It is tau for a single
    exponential, fast_fraction*tau_fast_ns+(1-fast_fraction)*tau_slow_ns for a
    double exponential, and tau_scale_ns*Gamma(1+1/stretch_beta) for a stretched
    exponential. A stretched fit has NaN fast/slow times and fast_fraction.
    flags includes cautions; rejection_flags identifies summary-excluding flags.
    window_sensitivity is a fit-window diagnostic, not a sampling uncertainty.
    The model is A*[f exp(-t/tau_fast)+(1-f) exp(-t/tau_slow)], with no offset.
    A defaults to 1. A fitted A is an extrapolated tail amplitude, not a new
    normalization of the measured ACF. No independent-data covariance error
    is presented as a trustworthy uncertainty on correlated ACF samples.
    """

    model: str
    tau_ns: float
    tau_fast_ns: float
    tau_slow_ns: float
    fast_fraction: float
    amplitude: float
    r_squared: float
    rmse: float
    n_points: int
    fit_tmin_ns: float
    fit_tmax_ns: float
    optimizer_success: bool
    accepted: bool
    flags: tuple[str, ...]
    tau_scale_ns: float = float("nan")
    stretch_beta: float = 1.0
    weighted_rmse: float = float("nan")
    selection_score: float = float("nan")
    tail_fraction: float = float("nan")
    window_sensitivity: float = float("nan")
    window_tmax_ns: tuple[float, ...] = ()
    window_tau_ns: tuple[float, ...] = ()
    candidate_models: tuple[str, ...] = ()
    candidate_scores: tuple[float, ...] = ()
    rejection_flags: tuple[str, ...] = ()
    window_method: str = "manual"

    @property
    def status(self) -> str:
        """Numerical status; 'ok' does not establish trajectory convergence."""
        if not self.optimizer_success:
            return "failed"
        if not self.accepted:
            return "unresolved"
        return "warning" if self.flags else "ok"

    def predict(self, time_ns: Any) -> np.ndarray:
        time = np.asarray(time_ns, dtype=float)
        if not self.optimizer_success:
            return np.full_like(time, np.nan)
        if self.model == "stretched":
            return self.amplitude * np.exp(
                -np.power(np.maximum(time, 0.0) / self.tau_scale_ns, self.stretch_beta)
            )
        return self.amplitude * (
            self.fast_fraction * np.exp(-time / self.tau_fast_ns)
            + (1.0 - self.fast_fraction) * np.exp(-time / self.tau_slow_ns)
        )


@dataclass(frozen=True)
class ReconfigurationTimeResult:
    """Per-chain/residue fits plus descriptive chain-level uncertainty.

    tau_per_chain_residue_ns contains estimates admitted by the selected policy. Consult fits for
    every numerical estimate and its flags. mean_tau_ns averages the chain
    means, NOT a fit to the ensemble-averaged ACF. SEM is NaN for one chain.
    With residue_averaging='common', the overall torsion summary uses only
    residue numbers accepted in EVERY selected chain. The residue profile
    instead uses all accepted chains at each residue and records their counts.
    """

    fits: dict[str, tuple[ReconfigurationDecayFit, ...]]
    chain_labels: tuple[str, ...]
    residue_numbers: Optional[np.ndarray]
    observables: tuple[str, ...]
    kind: str
    model: str
    tau_per_chain_residue_ns: np.ndarray
    accepted: np.ndarray
    per_chain_tau_ns: np.ndarray
    mean_tau_ns: float
    std_tau_ns: float
    sem_tau_ns: float
    n_chains: int
    n_total_chains: int
    residue_mean_tau_ns: np.ndarray
    residue_std_tau_ns: np.ndarray
    residue_sem_tau_ns: np.ndarray
    residue_n_chains: np.ndarray
    summary_residue_numbers: Optional[np.ndarray]
    summary_n_residues_per_chain: np.ndarray
    residue_averaging: str
    n_fitted: int
    n_accepted: int
    kinetic_scale: float
    source_frame_start: int
    source_frame_stop: int
    source_dt_ns: float
    dataset_label: str = "data"

    @property
    def acceptance_fraction(self) -> float:
        return self.n_accepted / self.n_fitted if self.n_fitted else float("nan")

    def report(self, *, max_rows: Optional[int] = None) -> str:
        """Format all fitted component parameters and quality diagnostics.

        No thresholding or rejection is performed here. The reported tau is
        the area of the fitted normalized correlation, not necessarily a
        well-constrained experimental relaxation time. ``max_rows`` optionally
        abbreviates large torsion datasets; None includes every site.
        """
        if max_rows is not None:
            max_rows = _integer(max_rows, "max_rows", 1)
        lines = [
            f"Dataset: {self.dataset_label} | observables: {', '.join(self.observables)}",
            f"Fits admitted to summary: {self.n_accepted}/{self.n_fitted} | "
            f"mean tau = {self.mean_tau_ns:.6g} ns | "
            f"across-chain SEM = {self.sem_tau_ns:.6g} ns",
        ]
        shown = 0
        total = sum(len(items) for items in self.fits.values())
        for chain in self.chain_labels:
            for ri, fit in enumerate(self.fits[chain]):
                if max_rows is not None and shown >= max_rows:
                    lines.append(
                        f"... {total - shown} more site fits (increase max_rows to display)"
                    )
                    return "\n".join(lines)
                residue = (
                    "Rg"
                    if self.residue_numbers is None
                    else f"residue {int(self.residue_numbers[ri])}"
                )
                lines.append(
                    f"{chain} {residue}: model={fit.model} status={fit.status} "
                    f"accepted={fit.accepted} success={fit.optimizer_success}"
                )
                if fit.optimizer_success:
                    lines.append(
                        f"  effective tau={fit.tau_ns:.6g} ns; " f"amplitude A={fit.amplitude:.5g}"
                    )
                    if fit.model == "double":
                        lines.append(
                            f"  tau_fast={fit.tau_fast_ns:.6g} ns; "
                            f"tau_slow={fit.tau_slow_ns:.6g} ns; "
                            f"fast fraction f={fit.fast_fraction:.5g}; "
                            f"slow fraction={1.0 - fit.fast_fraction:.5g}"
                        )
                    elif fit.model == "stretched":
                        lines.append(
                            f"  tau_scale={fit.tau_scale_ns:.6g} ns; "
                            f"beta={fit.stretch_beta:.5g}"
                        )
                    lines.append(
                        f"  fit window={fit.fit_tmin_ns:.6g}–{fit.fit_tmax_ns:.6g} ns; "
                        f"R²={fit.r_squared:.4g}; RMSE={fit.rmse:.5g}; "
                        f"weighted RMSE={fit.weighted_rmse:.5g}; "
                        f"robust score={fit.selection_score:.5g}"
                    )
                    lines.append(
                        f"  extrapolated area={fit.tail_fraction:.1%}; "
                        f"window sensitivity={fit.window_sensitivity:.1%}; "
                        f"sampled fit points={fit.n_points}"
                    )
                lines.append(f"  cautions: {', '.join(fit.flags) if fit.flags else 'none'}")
                if fit.rejection_flags:
                    lines.append(f"  rejection reasons: {', '.join(fit.rejection_flags)}")
                shown += 1
        return "\n".join(lines)


def _chain_indices(
    labels: tuple[str, ...],
    chain_indices: Any,
    include: Any,
    exclude: Any,
    strict: bool,
    labeled: bool,
) -> np.ndarray:
    if not isinstance(strict, (bool, np.bool_)):
        raise TypeError("strict_chain_selection must be a boolean")
    if include is not None and exclude is not None:
        raise ValueError("include_chains and exclude_chains are mutually exclusive")
    if len(set(labels)) != len(labels):
        raise ValueError("chain_labels must be unique")
    if chain_indices is None:
        indices = list(range(len(labels)))
    else:
        if isinstance(chain_indices, (str, bytes)):
            raise TypeError("chain_indices must be a sequence")
        indices = [_integer(i, "chain index", 0) for i in chain_indices]
        if len(set(indices)) != len(indices) or any(i >= len(labels) for i in indices):
            raise ValueError("chain_indices contains duplicates or out-of-range values")
    spec = include if include is not None else exclude
    if spec is not None:
        if not labeled:
            raise ValueError("named selection requires physical chain_labels; use chain_indices")
        if isinstance(spec, (str, bytes)):
            raise TypeError("include_chains/exclude_chains must be a sequence of segment IDs")
        wanted = tuple(str(x) for x in spec)
        if len(set(wanted)) != len(wanted):
            raise ValueError("chain selection contains duplicate IDs")
        unknown = sorted(set(wanted) - set(labels))
        if unknown and strict:
            raise ValueError(f"unknown chains {unknown}; available: {list(labels)}")
        indices = [i for i in indices if (labels[i] in wanted) == (include is not None)]
    if not indices:
        raise ValueError("chain selection leaves no chains")
    return np.asarray(indices, dtype=np.int64)


def _residue_numbers(
    numbers: np.ndarray, resnum: Any, residue_range: Any, residues: Any
) -> np.ndarray:
    if sum(x is not None for x in (resnum, residue_range, residues)) > 1:
        raise ValueError("resnum, residue_range, and residues are mutually exclusive")
    if resnum is not None:
        return np.asarray([_integer(resnum, "resnum")], dtype=np.int64)
    if residue_range is not None:
        if len(residue_range) != 2:
            raise ValueError("residue_range must be (first, last), inclusive")
        lo, hi = [_integer(x, "residue range endpoint") for x in residue_range]
        if hi < lo:
            raise ValueError("residue_range must be increasing")
        return numbers[(numbers >= lo) & (numbers <= hi)]
    if residues is not None:
        if isinstance(residues, (str, bytes)):
            raise TypeError("residues must be a sequence of PDB residue numbers")
        selected = [_integer(x, "residue number") for x in residues]
        if not selected or len(set(selected)) != len(selected):
            raise ValueError("residues must be nonempty and unique")
        return np.asarray(selected, dtype=np.int64)
    return numbers.copy()


def _torsion_names(torsions: Any) -> tuple[str, ...]:
    if isinstance(torsions, str):
        text = torsions.strip().lower().replace("/", "+").replace(",", "+").replace(" ", "")
        if text in {"phipsi", "phi_psi", "backbone"}:
            text = "phi+psi"
        names = tuple(text.split("+"))
    else:
        names = tuple(str(x).strip().lower() for x in torsions)
    if not names or len(set(names)) != len(names) or any(x not in _ANGLES for x in names):
        raise ValueError("torsions must contain unique phi, psi, omega, chi1, or chi2 names")
    return names


def _runs(valid: np.ndarray, breaks: np.ndarray) -> list[tuple[int, int]]:
    """Maximal valid runs; never connect a NaN gap or independent trajectory boundary."""
    starts = np.flatnonzero(valid & (np.r_[True, ~valid[:-1]] | breaks))
    ends = np.flatnonzero(valid & np.r_[~valid[1:] | breaks[1:], True]) + 1
    return list(zip(starts.tolist(), ends.tolist()))


def _connected_acf(
    features: np.ndarray,
    breaks: np.ndarray,
    max_lag: int,
    min_pairs: int,
    min_variance: float,
) -> tuple[np.ndarray, np.ndarray, float, str, float]:
    """FFT estimate with N_k normalization and no circular FFT wraparound.

    features is (frames, vector_components). Nonfinite frames separate runs.
    The mean is estimated over all valid frames, independently for this series.
    A lag's denominator is the number of valid within-run pairs, not N or N-k
    across gaps. min_variance is absolute in the squared feature units.
    """
    valid = np.all(np.isfinite(features), axis=1)
    out = np.full(max_lag + 1, np.nan)
    counts = np.zeros(max_lag + 1, dtype=np.int64)
    if not np.any(valid):
        return out, counts, float("nan"), "missing", 0.0
    average = features[valid].mean(axis=0)
    centered = features - average
    variance = float(np.mean(np.sum(centered[valid] ** 2, axis=1)))
    segments = _runs(valid, breaks)
    longest = max(stop - start - 1 for start, stop in segments)
    if int(valid.sum()) < min_pairs:
        return out, counts, variance, "insufficient_samples", float(longest)
    if not np.isfinite(variance) or variance <= min_variance:
        return out, counts, variance, "constant", float(longest)
    sums = np.zeros(max_lag + 1, dtype=float)
    for start, stop in segments:
        length = stop - start
        limit = min(max_lag, length - 1)
        nfft = next_fast_len(2 * length - 1)
        fft = rfft(centered[start:stop], n=nfft, axis=0)
        numerator = irfft(np.sum(fft.conj() * fft, axis=1).real, n=nfft)[: limit + 1]
        sums[: limit + 1] += numerator
        counts[: limit + 1] += length - np.arange(limit + 1)
    good = counts >= min_pairs
    out[good] = sums[good] / counts[good] / variance
    out[0] = 1.0
    return out, counts, variance, "ok", float(longest)


def calculate_reconfiguration_correlations(
    data: Any,
    delta_t: Any,
    *,
    observable: str = "auto",
    torsions: Union[str, Sequence[str]] = "phi+psi",
    resnum: Optional[int] = None,
    residue_range: Optional[tuple[int, int]] = None,
    residues: Optional[Sequence[int]] = None,
    include_chains: Any = None,
    exclude_chains: Any = None,
    chain_indices: Optional[Sequence[int]] = None,
    strict_chain_selection: bool = True,
    mintime: Any = None,
    maxtime: Any = None,
    time_start: Any = None,
    time_stop: Any = None,
    frame_start: Optional[int] = None,
    frame_stop: Optional[int] = None,
    time_origin: Any = 0.0,
    max_lag: Any = None,
    max_lag_frames: Optional[int] = None,
    lag_stride: int = 1,
    kinetic_scale: float = 1.0,
    torsion_weighting: str = "equal",
    require_all_angles: bool = True,
    min_pairs: int = 20,
    min_variance: float = 1e-12,
    segment_ids: Any = None,
    correlation_dtype: str = "float32",
    max_output_gib: float = 2.0,
) -> Any:
    """Compute connected Rg/scalar or circular torsion autocorrelations.

    Input may be one cached Rg/torsion result, a 1D/2D scalar array, a mapping of
    fields, or a flat mapping {dataset_key: result}; output preserves that shape.
    ``observable='auto'`` detects peptide torsions, otherwise uses
    ``rg_per_chain_nm``. Other scalar fields are accepted through observable.

    ``torsions`` accepts 'phi', 'psi', 'phi+psi', or a sequence of supported
    angle names. Equal weighting averages the separately normalized angle ACFs;
    'variance' weights them by their fluctuation variances (the normalized dot
    product ACF of the concatenated sin/cos feature vector). Neither is a
    joint-state transition/survival correlation. By default, every requested
    angle must be present and fluctuating at a site; terminals or incomplete
    sites remain NaN. Set require_all_angles=False to use available components.

    Select PDB numbers using exactly one of resnum/residue_range/residues;
    ranges are inclusive, and omission selects all residues. Repeated residue
    numbers within a chain are rejected. Chain-ID selections can be global
    sequences or mappings {dataset_key: sequence}; indices refer to original
    chain order, and combine with named filters by intersection.

    ``delta_t`` is the spacing BETWEEN INPUT SAMPLES (already includes any
    trajectory stride). mintime/maxtime (or time_start/time_stop) select stored,
    UNCORRECTED time, starting at time_origin for the first supplied sample;
    stop is exclusive. To select a previously cropped trajectory by original
    times, set time_origin to its first stored time. Frame limits are an
    alternative and do not combine with time limits. Numeric times are ns.

    Only the output lag axis is multiplied by kinetic_scale. max_lag and all
    later fitting limits use that physical lag axis; max_lag_frames is an
    alternative. Default maximum lag is 10% of the selected sample count.
    Never apply kinetic_scale again when fitting or plotting these results.

    NaNs split observations into continuous runs; no interpolation or pairing
    across gaps is done. Torsion source_file_indices and discontinuous
    source_frame_indices split runs as well. Supply segment_ids for other input
    containing concatenated independent runs. Distinct dataset keys are NEVER
    concatenated. Such boundaries reset pair accumulation but not the per-series
    window mean. Irregular sampling within an otherwise continuous run is not
    supported. Data must represent a stationary trajectory within each window.

    Computational cost is O(N log N) per chain/residue/angle. Output memory is
    bounded by max_output_gib (angles are processed one series at a time).
    """
    is_torsion = _field(data, "chains") is not None and _field(data, "unit") is not None
    field = "rg_per_chain_nm" if observable in {"auto", "rg", "rgyr"} else str(observable)
    is_scalar = _field(data, field) is not None or isinstance(data, np.ndarray)
    if isinstance(data, Mapping) and not is_torsion and not is_scalar:
        if not data:
            raise ValueError("data mapping is empty")
        for name, spec in (
            ("include_chains", include_chains),
            ("exclude_chains", exclude_chains),
            ("segment_ids", segment_ids),
        ):
            if isinstance(spec, Mapping) and any(key not in data for key in spec):
                raise KeyError(f"{name} contains unknown dataset keys")
        out = {}
        for key, item in data.items():
            out[key] = calculate_reconfiguration_correlations(
                item,
                delta_t,
                observable=observable,
                torsions=torsions,
                resnum=resnum,
                residue_range=residue_range,
                residues=residues,
                include_chains=(
                    include_chains.get(key)
                    if isinstance(include_chains, Mapping)
                    else include_chains
                ),
                exclude_chains=(
                    exclude_chains.get(key)
                    if isinstance(exclude_chains, Mapping)
                    else exclude_chains
                ),
                chain_indices=chain_indices,
                strict_chain_selection=strict_chain_selection,
                mintime=mintime,
                maxtime=maxtime,
                time_start=time_start,
                time_stop=time_stop,
                frame_start=frame_start,
                frame_stop=frame_stop,
                time_origin=time_origin,
                max_lag=max_lag,
                max_lag_frames=max_lag_frames,
                lag_stride=lag_stride,
                kinetic_scale=kinetic_scale,
                torsion_weighting=torsion_weighting,
                require_all_angles=require_all_angles,
                min_pairs=min_pairs,
                min_variance=min_variance,
                segment_ids=(
                    segment_ids.get(key) if isinstance(segment_ids, Mapping) else segment_ids
                ),
                correlation_dtype=correlation_dtype,
                max_output_gib=max_output_gib,
            )
            # replace keeps result caching simple: the fields remain dataclass fields.
            out[key] = replace(out[key], dataset_label=str(key))
        return out
    if isinstance(include_chains, Mapping) or isinstance(exclude_chains, Mapping):
        raise TypeError("for a single result, supply chain sequences rather than dataset mappings")
    dt = _time(delta_t, "delta_t")
    scale = float(kinetic_scale)
    if dt <= 0 or not np.isfinite(scale) or scale <= 0:
        raise ValueError("delta_t and kinetic_scale must be finite and positive")
    physical_dt = dt * scale
    if not np.isfinite(physical_dt):
        raise ValueError("delta_t * kinetic_scale must be finite")
    min_pairs = _integer(min_pairs, "min_pairs", 2)
    lag_stride = _integer(lag_stride, "lag_stride", 1)
    if not np.isfinite(min_variance) or min_variance < 0:
        raise ValueError("min_variance must be finite and nonnegative")
    if torsion_weighting not in {"equal", "variance"}:
        raise ValueError("torsion_weighting must be 'equal' or 'variance'")
    if not isinstance(require_all_angles, (bool, np.bool_)):
        raise TypeError("require_all_angles must be a boolean")
    dtype = np.dtype(correlation_dtype)
    if dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise ValueError("correlation_dtype must be float32 or float64")
    if not np.isfinite(max_output_gib) or max_output_gib <= 0:
        raise ValueError("max_output_gib must be finite and positive")

    if is_torsion:
        if observable not in {"auto", "torsions", "peptide-torsions"}:
            raise ValueError("for torsion results, select angle names with torsions=...")
        names = _torsion_names(torsions)
        chains = _field(data, "chains")
        if not chains:
            raise ValueError(
                "torsion result has no atomistic chains; select a nonempty peptide group"
            )
        labels = tuple(str(x) for x in chains)
        if any(not isinstance(x, str) for x in chains):
            raise TypeError("torsion chain keys must be strings")
        indices = _chain_indices(
            labels, chain_indices, include_chains, exclude_chains, strict_chain_selection, True
        )
        selected_labels = tuple(labels[i] for i in indices)
        n_frames = _integer(_field(data, "n_frames"), "n_frames", 2)
        per_chain_numbers = {}
        for label in selected_labels:
            raw = np.asarray(_field(chains[label], "residue_numbers"))
            if raw.ndim != 1:
                raise ValueError(f"{label}: residue_numbers must be 1D")
            numbers = np.asarray([_integer(x, "residue number") for x in raw], dtype=np.int64)
            if np.unique(numbers).size != numbers.size:
                raise ValueError(f"{label}: duplicate PDB residue numbers are ambiguous")
            per_chain_numbers[label] = numbers
        union = np.unique(np.concatenate(list(per_chain_numbers.values())))
        numbers = _residue_numbers(union, resnum, residue_range, residues)
        missing = sorted(set(numbers) - set(union))
        if missing:
            raise ValueError(f"residue numbers absent from every selected chain: {missing}")
        if not numbers.size:
            raise ValueError("residue selection is empty")
        unit = str(_field(data, "unit")).lower()
        if unit not in {"radians", "degrees", "radian", "degree", "rad", "deg"}:
            raise ValueError(f"unknown torsion unit {unit!r}")
        angle_scale = math.pi / 180.0 if unit in {"degrees", "degree", "deg"} else 1.0
        n_sites = len(numbers)
    else:
        if any(x is not None for x in (resnum, residue_range, residues)):
            raise ValueError("residue selection only applies to torsion results")
        values = np.asarray(
            data if isinstance(data, np.ndarray) else _field(data, field), dtype=float
        )
        if values.ndim == 1:
            values = values[None, :]
        if values.ndim != 2 or not values.size or values.shape[1] < 2:
            raise ValueError(f"{field} must be a nonempty (chains, frames) or 1D array")
        n_frames = values.shape[1]
        raw_labels = _field(data, "chain_labels")
        labeled = raw_labels is not None and len(raw_labels) > 0
        labels = (
            tuple(str(x) for x in raw_labels)
            if labeled
            else tuple(f"chain {i}" for i in range(values.shape[0]))
        )
        if len(labels) != values.shape[0]:
            raise ValueError("chain_labels does not match array rows")
        indices = _chain_indices(
            labels, chain_indices, include_chains, exclude_chains, strict_chain_selection, labeled
        )
        selected_labels = tuple(labels[i] for i in indices)
        n_sites, numbers, names = 1, None, (field,)

    if mintime is not None and time_start is not None:
        raise ValueError("use mintime OR time_start")
    if maxtime is not None and time_stop is not None:
        raise ValueError("use maxtime OR time_stop")
    t0 = mintime if mintime is not None else time_start
    t1 = maxtime if maxtime is not None else time_stop
    origin = _time(time_origin, "time_origin")
    if (t0 is not None or t1 is not None) and (frame_start is not None or frame_stop is not None):
        raise ValueError("time limits and frame limits are mutually exclusive")
    if t0 is not None or t1 is not None:
        start_time = origin if t0 is None else _time(t0, "mintime")
        stop_time = None if t1 is None else _time(t1, "maxtime")
        if start_time < origin or (stop_time is not None and stop_time <= start_time):
            raise ValueError("time limits must satisfy time_origin <= mintime < maxtime")
        start = max(0, _left_frame(start_time - origin, dt))
        stop = n_frames if stop_time is None else min(n_frames, _left_frame(stop_time - origin, dt))
    else:
        start = 0 if frame_start is None else _integer(frame_start, "frame_start", 0)
        stop = (
            n_frames if frame_stop is None else min(n_frames, _integer(frame_stop, "frame_stop", 0))
        )
    if stop - start < min_pairs:
        raise ValueError(f"analysis window contains fewer than min_pairs={min_pairs} frames")
    n = stop - start
    if max_lag is not None and max_lag_frames is not None:
        raise ValueError("max_lag and max_lag_frames are mutually exclusive")
    if max_lag is not None:
        lag = _time(max_lag, "max_lag") / physical_dt
        limit = int(math.floor(lag + 1e-12))
    elif max_lag_frames is not None:
        limit = _integer(max_lag_frames, "max_lag_frames", 1)
    else:
        limit = max(1, n // 10)
    if not 1 <= limit < n:
        raise ValueError("maximum lag must be at least one sample and shorter than the window")
    lag_frames = np.arange(0, limit + 1, lag_stride, dtype=np.int64)
    if lag_frames.size < 2:
        raise ValueError("lag_stride leaves no positive lag")
    if limit > n // 2:
        warnings.warn(
            "maximum lag exceeds half the window; long-lag correlations may be very noisy",
            RuntimeWarning,
            stacklevel=2,
        )
    shape = (len(indices), n_sites, len(lag_frames))
    estimated_gib = math.prod(shape) * (dtype.itemsize + 8) / 1024**3
    if estimated_gib > max_output_gib:
        raise MemoryError(
            f"correlations/counts need ~{estimated_gib:.2f} GiB; reduce max_lag, increase lag_stride, select fewer sites, or raise max_output_gib"
        )
    acfs = np.full(shape, np.nan, dtype=dtype)
    counts = np.zeros(shape, dtype=np.int64)
    variance = np.full(shape[:2] + (len(names),), np.nan)
    plateau = np.full_like(variance, np.nan)
    status = np.full(variance.shape, "missing", dtype="U24")

    breaks_full = np.zeros(n_frames, dtype=bool)
    breaks_full[0] = True
    for key, source in (
        ("segment_ids", segment_ids),
        ("source_file_indices", _field(data, "source_file_indices")),
    ):
        if source is not None:
            array = np.asarray(source)
            if array.shape != (n_frames,):
                raise ValueError(f"{key} must have one entry per input frame")
            breaks_full[1:] |= array[1:] != array[:-1]
    source_frames = _field(data, "source_frame_indices")
    if source_frames is not None:
        source_frames = np.asarray(source_frames)
        if source_frames.shape != (n_frames,):
            raise ValueError("source_frame_indices must match input frames")
        breaks_full[1:] |= np.diff(source_frames) != 1
    breaks = breaks_full[start:stop].copy()
    breaks[0] = True
    # This is an upper bound; individual NaN gaps may shorten a site's support.
    longest_global = max(right - left - 1 for left, right in _runs(np.ones(n, dtype=bool), breaks))
    support = np.zeros(shape[:2], dtype=float)
    for ci, label in enumerate(selected_labels):
        if is_torsion:
            chain = chains[label]
            local_numbers = per_chain_numbers[label]
            lookup = {int(number): row for row, number in enumerate(local_numbers)}
            arrays = []
            for name in names:
                array = np.asarray(_field(chain, name))
                if array.shape != (len(local_numbers), n_frames):
                    raise ValueError(f"{label}: {name} must have shape (residues, frames)")
                arrays.append(array)
        for ri in range(n_sites):
            if is_torsion:
                row = lookup.get(int(numbers[ri]))
                if row is None:
                    continue
                series = [
                    np.asarray(array[row, start:stop], dtype=float) * angle_scale
                    for array in arrays
                ]
                # For a required combination use a common finite frame mask.
                common_valid = (
                    np.logical_and.reduce([np.isfinite(x) for x in series])
                    if require_all_angles
                    else None
                )
            else:
                series = [values[indices[ci], start:stop]]
                common_valid = None
            component_acfs, component_counts, weights = [], [], []
            for ai, scalar in enumerate(series):
                if is_torsion:
                    finite = np.isfinite(scalar)
                    safe = np.where(finite, scalar, 0.0)
                    features = np.column_stack((np.cos(safe), np.sin(safe)))
                    features[~finite] = np.nan
                    if common_valid is not None:
                        features[~common_valid] = np.nan
                    good = np.all(np.isfinite(features), axis=1)
                    if np.any(good):
                        plateau[ci, ri, ai] = float(np.sum(features[good].mean(axis=0) ** 2))
                else:
                    features = scalar[:, None]
                corr, pairs, var, why, longest = _connected_acf(
                    features, breaks, limit, min_pairs, min_variance
                )
                variance[ci, ri, ai] = var
                status[ci, ri, ai] = why
                if why == "ok":
                    component_acfs.append(corr)
                    component_counts.append(pairs)
                    weights.append(1.0 if torsion_weighting == "equal" else var)
                    support[ci, ri] = (
                        longest if support[ci, ri] == 0 else min(support[ci, ri], longest)
                    )
            if not component_acfs or (require_all_angles and len(component_acfs) != len(names)):
                continue
            # Fixed component set: missing pairs at a lag make the combined ACF
            # unknown, rather than silently changing the observable with lag.
            combined = np.average(np.stack(component_acfs), axis=0, weights=weights)
            acfs[ci, ri] = combined[lag_frames]
            counts[ci, ri] = np.min(component_counts, axis=0)[lag_frames]
    result = ReconfigurationCorrelationResult(
        lag_time_ns=lag_frames * physical_dt,
        correlation=acfs,
        pair_counts=counts,
        chain_labels=selected_labels,
        residue_numbers=numbers,
        observables=names,
        kind="torsions" if is_torsion else "scalar",
        component_variance=variance,
        circular_plateau=plateau,
        component_status=status,
        dt_ns=dt,
        kinetic_scale=scale,
        n_frames=n,
        frame_start=start,
        frame_stop=stop,
        time_origin_ns=origin,
        max_contiguous_time_ns=longest_global * physical_dt,
        torsion_weighting=torsion_weighting,
        require_all_angles=bool(require_all_angles),
        min_pairs=min_pairs,
    )
    # Store per-site support as a declared field below, not as transient attributes.
    return replace(result, contiguous_time_per_site_ns=support * physical_dt)


def _empty_fit(model: str, reason: str) -> ReconfigurationDecayFit:
    return ReconfigurationDecayFit(
        model=model,
        tau_ns=float("nan"),
        tau_fast_ns=float("nan"),
        tau_slow_ns=float("nan"),
        fast_fraction=float("nan"),
        amplitude=float("nan"),
        r_squared=float("nan"),
        rmse=float("nan"),
        n_points=0,
        fit_tmin_ns=float("nan"),
        fit_tmax_ns=float("nan"),
        optimizer_success=False,
        accepted=False,
        flags=(reason,),
        rejection_flags=(reason,),
    )


def _fit_weights(time: np.ndarray, mode: str) -> np.ndarray:
    """Quadrature weights, not inverse statistical variances of lag estimates."""
    if len(time) < 2 or mode == "uniform":
        return np.ones(len(time), dtype=float)

    def widths(values: np.ndarray) -> np.ndarray:
        mid = 0.5 * (values[1:] + values[:-1])
        return np.diff(np.r_[values[0], mid, values[-1]])

    linear = widths(time)
    linear /= linear.sum()
    positive = time[time > 0]
    offset = float(positive[0]) if positive.size else 1.0
    logarithmic = widths(np.log1p(time / offset))
    logarithmic /= logarithmic.sum()
    weights = 0.5 * (linear + logarithmic)
    return weights * len(weights)


def _automatic_fit_end(
    time: np.ndarray,
    curve: np.ndarray,
    first: float,
    cutoff: float,
    min_fit_points: int,
) -> tuple[float, str]:
    """Use a sustained near-zero region, not an isolated crossing, as an endpoint.

    Smoothing is ONLY for window detection. Raw signed correlations are fitted.
    The cutoff is an operational threshold, not an estimated noise confidence
    interval. Significant later positive recovery prevents premature truncation.
    """
    keep = np.isfinite(curve) & (time >= first)
    x, y = time[keep], curve[keep]
    if len(x) < max(2, min_fit_points):
        return float(time[-1]), "available_lags"
    size = min(101, max(3, 2 * (len(x) // 200) + 1))
    smooth = median_filter(y, size=size, mode="nearest")
    hold = max(3, int(math.ceil(len(x) * 0.015)))
    low = smooth <= cutoff
    # A short excursion into zero followed by a clear recovery is not a tail.
    high = smooth > max(0.10, 3.0 * cutoff)
    high_cumulative = np.r_[0, np.cumsum(high, dtype=np.int64)]
    high_runs = high_cumulative[hold:] - high_cumulative[:-hold]
    last_recovery = np.flatnonzero(high_runs == hold)
    after = int(last_recovery[-1] + hold) if last_recovery.size else 0
    low_cumulative = np.r_[0, np.cumsum(low, dtype=np.int64)]
    runs = low_cumulative[hold:] - low_cumulative[:-hold]
    candidates = np.flatnonzero(runs == hold)
    candidates = candidates[(candidates >= after) & (candidates + hold >= min_fit_points)]
    if candidates.size:
        end = min(len(x) - 1, int(candidates[0] + hold - 1))
        return float(x[end]), "sustained_near_zero"
    return float(x[-1]), "available_lags"


def _model_tail_fraction(fit: ReconfigurationDecayFit, time: float) -> float:
    """Fraction of the normalized fitted correlation area beyond time."""
    if not fit.optimizer_success or not np.isfinite(fit.tau_ns) or fit.tau_ns <= 0:
        return float("nan")
    if fit.model == "stretched":
        z = (max(0.0, time) / fit.tau_scale_ns) ** fit.stretch_beta
        return float(gammaincc(1.0 / fit.stretch_beta, z))
    f = fit.fast_fraction
    return float(
        (
            f * fit.tau_fast_ns * np.exp(-time / fit.tau_fast_ns)
            + (1.0 - f) * fit.tau_slow_ns * np.exp(-time / fit.tau_slow_ns)
        )
        / fit.tau_ns
    )


def _fit_one(
    time: np.ndarray,
    correlation: np.ndarray,
    *,
    model: str,
    fit_amplitude: bool,
    fit_tmin: float,
    fit_tmax: float,
    physical_dt: float,
    duration: float,
    min_fit_points: int,
    min_r_squared: Optional[float],
    min_decay_times: float,
    min_observation_times: float,
    include_flagged: bool,
    loss: str = "soft_l1",
    f_scale: float = 0.05,
    lag_weighting: str = "balanced",
    max_fit_points: int = 512,
) -> ReconfigurationDecayFit:
    valid = np.isfinite(correlation) & (time >= fit_tmin) & (time <= fit_tmax)
    all_x, all_y = time[valid], correlation[valid]
    n_parameters = {"single": 1, "stretched": 2, "double": 3}[model] + int(fit_amplitude)
    if len(all_x) < max(min_fit_points, n_parameters + 2):
        return _empty_fit(model, "insufficient_fit_points")
    if not np.any((all_x > 0) & (all_y > 0)):
        return _empty_fit(model, "unresolved_at_sample_spacing")
    # Bounded deterministic sampling limits optimization cost on dense lag grids.
    # The union retains both early fast relaxation and the late slow tail.
    if len(all_x) > max_fit_points:
        half = max_fit_points // 2
        choose = np.unique(
            np.r_[
                np.linspace(0, len(all_x) - 1, half).astype(int),
                (np.geomspace(1, len(all_x), half) - 1).astype(int),
            ]
        )
        required = max(min_fit_points, n_parameters + 2)
        if choose.size < required:
            choose = np.linspace(0, len(all_x) - 1, max_fit_points).astype(int)
        x, y = all_x[choose], all_y[choose]
        # Preserve the weighting measure on the ORIGINAL lag grid even when
        # representative optimization samples are nonuniformly spaced.
        assignment = np.searchsorted(0.5 * (x[1:] + x[:-1]), all_x)
        weights = np.bincount(
            assignment,
            weights=_fit_weights(all_x, lag_weighting),
            minlength=len(x),
        )
        weights *= len(weights) / weights.sum()
    else:
        x, y = all_x, all_y
        weights = _fit_weights(x, lag_weighting)
    sqrt_weights = np.sqrt(weights)
    scale = max(float(x[-1]), physical_dt)
    u = x / scale
    low = physical_dt / scale / 100.0
    high = max(duration, float(x[-1]), physical_dt) / scale * 100.0
    below = np.flatnonzero((x > 0) & (y <= math.exp(-1)))
    guess = float(x[below[0]]) / scale if below.size else 1.0
    guess = float(np.clip(guess, low * 10, high / 10))

    def prediction(parameters: np.ndarray, times: np.ndarray = u) -> np.ndarray:
        if model == "single":
            decay = np.exp(-times / np.exp(parameters[0]))
        elif model == "stretched":
            decay = np.exp(-np.power(times / np.exp(parameters[0]), parameters[1]))
        else:
            tau1, tau2 = np.exp(parameters[:2])
            fraction = parameters[2]
            decay = fraction * np.exp(-times / tau1) + (1 - fraction) * np.exp(-times / tau2)
        amplitude = parameters[-1] if fit_amplitude else 1.0
        return amplitude * decay

    if model == "single":
        initial = [
            [math.log(np.clip(guess * factor, low * 2, high / 2))] for factor in (0.3, 1.0, 3.0)
        ]
        lower, upper = [math.log(low)], [math.log(high)]
    elif model == "stretched":
        initial = [[math.log(guess), beta] for beta in (0.45, 0.7, 0.95)]
        lower, upper = [math.log(low), 0.2], [math.log(high), 1.0]
    else:
        initial = []
        for ratio in (3.0, 10.0, 30.0):
            for fraction in (0.25, 0.75):
                initial.append(
                    [
                        math.log(max(low * 2, guess / ratio)),
                        math.log(min(high / 2, guess * ratio)),
                        fraction,
                    ]
                )
        lower = [math.log(low), math.log(low), 0.001]
        upper = [math.log(high), math.log(high), 0.999]
    if fit_amplitude:
        amplitude_guess = float(np.clip(np.max(y), 0.01, 1.9))
        initial = [item + [amplitude_guess] for item in initial]
        lower += [0.001]
        upper += [2.0]
    best = None

    # The loss acts on raw C residuals, then quadrature weights are applied.
    # Applying weights before a nonlinear loss would alter its outlier cutoff.
    def residual(parameters: np.ndarray) -> np.ndarray:
        difference = prediction(parameters) - y
        if loss == "soft_l1":
            z = difference / f_scale
            # Stable signed square root of the soft_l1 loss, incl. small z.
            difference = difference * np.sqrt(2.0 / (np.sqrt(1.0 + z * z) + 1.0))
        return sqrt_weights * difference

    for guess_values in initial:
        try:
            solution = least_squares(
                residual,
                guess_values,
                bounds=(lower, upper),
                max_nfev=2000,
                ftol=1e-10,
                xtol=1e-10,
                gtol=1e-10,
            )
        except (ValueError, FloatingPointError, RuntimeError):
            continue
        if solution.success and np.all(np.isfinite(solution.x)):
            if best is None or solution.cost < best.cost:
                best = solution
    if best is None:
        return _empty_fit(model, "optimizer_failed")
    p = best.x
    amplitude = float(p[-1]) if fit_amplitude else 1.0
    beta = float(p[1]) if model == "stretched" else 1.0
    tau_scale = float(np.exp(p[0]) * scale)
    if model == "stretched":
        tau = float(tau_scale * gamma(1.0 + 1.0 / beta))
        tau_fast = tau_slow = float("nan")
        fast_fraction = float("nan")
        slow_diagnostic = tau
        fast_diagnostic = tau_scale
    elif model == "single":
        tau_fast = tau_slow = tau = tau_scale
        fast_fraction = 1.0
        slow_diagnostic = fast_diagnostic = tau
    else:
        tau_fast, tau_slow = np.exp(p[:2]) * scale
        fast_fraction = float(p[2])
        if tau_fast > tau_slow:
            tau_fast, tau_slow = tau_slow, tau_fast
            fast_fraction = 1 - fast_fraction
        tau_fast, tau_slow = float(tau_fast), float(tau_slow)
        tau = fast_fraction * tau_fast + (1 - fast_fraction) * tau_slow
        tau_scale = float("nan")
        slow_diagnostic, fast_diagnostic = tau_slow, tau_fast
    differences = prediction(p, all_x / scale) - all_y
    ss = float(np.sum(differences**2))
    total = float(np.sum((all_y - np.mean(all_y)) ** 2))
    r2 = 1 - ss / total if total > 1e-20 else float("nan")
    # Identical evaluation grid for all candidate models; no log(C) or dropping
    # negative observations. These scores are descriptive, NOT likelihoods.
    eval_weights = _fit_weights(all_x, lag_weighting)
    weighted_rmse = float(np.sqrt(np.average(differences**2, weights=eval_weights)))
    robust_error = differences
    if loss == "soft_l1":
        z = differences / f_scale
        robust_error = differences * np.sqrt(2.0 / (np.sqrt(1.0 + z * z) + 1.0))
    score = float(np.sqrt(np.average(robust_error**2, weights=eval_weights)))
    flags = []
    if min_r_squared is not None and (not np.isfinite(r2) or r2 < min_r_squared):
        flags.append("poor_single_decay_fit" if model == "single" else "poor_decay_fit")
    if fast_diagnostic < physical_dt:
        flags.append("faster_than_sampling")
    if float(all_x[-1] - all_x[0]) < min_decay_times * slow_diagnostic:
        flags.append("insufficient_decay_window")
    if duration < min_observation_times * slow_diagnostic:
        flags.append("too_few_relaxation_times")
    active = best.active_mask.copy()
    if model == "stretched" and active[1] == 1:
        active[1] = 0  # beta=1 is a valid single exponential, not an invalid bound.
    if np.any(active != 0):
        flags.append("parameter_at_bound")
    try:
        condition = float(np.linalg.cond(best.jac))
    except np.linalg.LinAlgError:
        condition = float("inf")
    if not np.isfinite(condition) or condition > 1e8:
        flags.append("ill_conditioned_fit")
    if model == "double":
        if tau_slow < 1.5 * tau_fast:
            flags.append("components_not_separated")
        if min(fast_fraction, 1 - fast_fraction) < 0.05:
            flags.append("weak_component")
    return ReconfigurationDecayFit(
        model=model,
        tau_ns=float(tau),
        tau_fast_ns=tau_fast,
        tau_slow_ns=tau_slow,
        fast_fraction=fast_fraction,
        amplitude=amplitude,
        r_squared=float(r2),
        rmse=math.sqrt(ss / len(all_y)),
        n_points=len(x),
        fit_tmin_ns=float(all_x[0]),
        fit_tmax_ns=float(all_x[-1]),
        optimizer_success=True,
        accepted=not flags or bool(include_flagged),
        flags=tuple(flags),
        tau_scale_ns=tau_scale,
        stretch_beta=beta,
        weighted_rmse=weighted_rmse,
        selection_score=score,
    )


def _fit_one_automatic(
    time: np.ndarray,
    curve: np.ndarray,
    *,
    model: str,
    fit_tmax: float,
    automatic_window: bool,
    auto_cutoff: float,
    check_stability: bool,
    max_tail_fraction: float,
    max_window_sensitivity: float,
    quality_policy: str,
    model_improvement: float,
    min_model_improvement: float,
    max_fit_rmse: float,
    **kwargs: Any,
) -> ReconfigurationDecayFit:
    end = fit_tmax
    window_method = "manual"
    if automatic_window:
        end, window_method = _automatic_fit_end(
            time,
            curve,
            kwargs["fit_tmin"],
            auto_cutoff,
            kwargs["min_fit_points"],
        )
    names = ("single", "stretched", "double") if model == "auto" else (model,)
    candidates = [_fit_one(time, curve, model=name, fit_tmax=end, **kwargs) for name in names]
    usable = [item for item in candidates if item.optimizer_success]
    if not usable:
        return replace(candidates[0], window_method=window_method)
    chosen = usable[0]
    for item in usable[1:]:
        improvement = chosen.selection_score - item.selection_score
        if improvement > max(min_model_improvement, model_improvement * chosen.selection_score):
            chosen = item
    chosen = replace(
        chosen,
        window_method=window_method,
        candidate_models=names,
        candidate_scores=tuple(item.selection_score for item in candidates),
    )
    flags = list(chosen.flags)
    tail = _model_tail_fraction(chosen, chosen.fit_tmax_ns)
    if tail > max_tail_fraction:
        flags.append("excessive_tail_extrapolation")
    # A positive relaxation model cannot account for a large sustained negative
    # lobe. Small negative tails (as in noisy finite ACFs) are not clipped.
    valid = np.isfinite(curve)
    negative_end = (
        min(float(time[-1]), chosen.fit_tmax_ns * 1.5) if automatic_window else chosen.fit_tmax_ns
    )
    negative = curve[valid & (time >= kwargs["fit_tmin"]) & (time <= negative_end)]
    if len(negative) >= 5:
        smooth = median_filter(negative, size=5, mode="nearest")
        hold = max(3, min(21, int(math.ceil(len(negative) * 0.015))))
        runs = np.convolve((smooth < -0.1).astype(int), np.ones(hold, int), mode="valid")
        if np.any(runs == hold):
            flags.append("substantial_negative_lobe")
    window_ends = [chosen.fit_tmax_ns]
    window_taus = [chosen.tau_ns]
    if check_stability:
        start = kwargs["fit_tmin"]
        for factor in (0.8, 1.2):
            trial_end = min(float(time[-1]), start + factor * (chosen.fit_tmax_ns - start))
            if any(
                math.isclose(trial_end, value, rel_tol=1e-8, abs_tol=1e-12) for value in window_ends
            ):
                continue
            alternative = _fit_one(time, curve, model=chosen.model, fit_tmax=trial_end, **kwargs)
            if alternative.optimizer_success and alternative.tau_ns > 0:
                window_ends.append(alternative.fit_tmax_ns)
                window_taus.append(alternative.tau_ns)
    sensitivity = (
        max(abs(value / chosen.tau_ns - 1.0) for value in window_taus)
        if len(window_taus) > 1
        else float("nan")
    )
    if np.isfinite(sensitivity) and sensitivity > max_window_sensitivity:
        flags.append("window_sensitive")
    if chosen.selection_score > max_fit_rmse:
        flags.append("large_fit_residual")
    flags = tuple(dict.fromkeys(flags))
    # A successful fit is always returned. Policy controls only whether a
    # finite numerical estimate contributes to the reported summary, never
    # whether its diagnostics are available.
    advisory = {
        "insufficient_decay_window",
        "too_few_relaxation_times",
        "poor_single_decay_fit",
        "poor_decay_fit",
    }
    if quality_policy == "permissive":
        # Noise and an underdetermined slow tail are diagnostics, not a reason
        # to erase a usable numerical fit. A result may remain scientifically
        # unreliable; see flags, tail_fraction, and window_sensitivity.
        rejected = ()
    elif quality_policy == "standard":
        rejected = tuple(flag for flag in flags if flag not in advisory)
    else:
        rejected = flags
    return replace(
        chosen,
        flags=flags,
        rejection_flags=rejected,
        accepted=(not rejected or bool(kwargs["include_flagged"])),
        tail_fraction=tail,
        window_sensitivity=sensitivity,
        window_tmax_ns=tuple(window_ends),
        window_tau_ns=tuple(window_taus),
    )


def fit_reconfiguration_times(
    correlations: Any,
    *,
    fit_tmin: Any = 0.0,
    fit_tmax: Any = None,
    model: str = "auto",
    fit_amplitude: bool = False,
    min_fit_points: int = 8,
    min_r_squared: Optional[float] = 0.8,
    min_decay_times: float = 2.0,
    min_observation_times: float = 10.0,
    include_flagged: bool = False,
    residue_averaging: str = "common",
    warn: bool = True,
    auto_window: bool = True,
    auto_cutoff: float = 0.02,
    loss: str = "soft_l1",
    f_scale: float = 0.05,
    lag_weighting: str = "balanced",
    max_fit_points: int = 512,
    check_stability: bool = True,
    max_window_sensitivity: float = 0.35,
    max_tail_fraction: float = 0.20,
    model_improvement: float = 0.20,
    min_model_improvement: float = 0.005,
    max_fit_rmse: float = 0.08,
    quality_policy: str = "permissive",
    report: bool = False,
) -> Any:
    """Fit per-chain/site decays and summarize their amplitude-normalized areas.

    Default model='auto' compares single, stretched, and double exponentials.
    It keeps the simpler model unless the more complex model reduces the
    descriptive robust residual score by >20% AND >0.005 correlation units.
    This is an explicit heuristic, NOT an AIC/p-value test on independent lags.
    All candidates use the same window, observations, and weights.

    With fit_tmax=None and auto_window=True, stop after a sustained region at
    C<=auto_cutoff (0.02), retaining the slow decay before it. Isolated crossings
    and late substantial positive recovery are checked. If no tail is detected,
    use all saved lags. The cutoff is operational, not an estimated noise level.
    Explicit fit_tmax overrides automatic window selection. Set auto_window=False
    to use the entire saved lag range. Smoothing is ONLY for window detection;
    raw signed C values, including negatives, are used for all fits.

    soft_l1 (f_scale=0.05 correlation units) reduces outlier influence.
    lag_weighting='balanced' averages linear-time and log(1+t/t_first)-time
    quadrature weights, so early decay and slow tail both influence fitting.
    'uniform' gives equal weights to saved lag samples. These weights do NOT
    estimate independent observations or their statistical uncertainties.
    At most max_fit_points raw lag samples are used by each multistart optimizer;
    residual/model comparisons are evaluated on every finite lag in the window.

    single: C=A*exp(-t/tau). double: C=A*[f exp(-t/tau_fast)
    +(1-f) exp(-t/tau_slow)]. stretched: C=A*exp(-(t/tau_scale)**beta),
    0.2<=beta<=1. A is fixed at 1 unless fit_amplitude=True. tau_ns is always
    the INTEGRAL of C_fit/A over 0..infinity: tau, f*tau_fast+(1-f)*tau_slow,
    or tau_scale*Gamma(1+1/beta), respectively. A tail-only fit is not the full
    ACF integral. No additive offset is fitted.

    check_stability refits the chosen model with endpoints at 0.8 and 1.2 times
    the selected fitting span (limited by available lags). window_sensitivity
    is the largest relative change in tau_ns, NOT a statistical error bar.
    tail_fraction is the fraction of fitted area extrapolated beyond the window.
    Both refer to the chosen model; inspect alternative models as well.

    quality_policy='permissive' (default): admit all numerically successful
    finite fits to the summary, retaining every diagnostic flag. This avoids
    silently discarding multiscale fits merely because their tail is noisy,
    extrapolated, or sensitive to the chosen window. It DOES NOT imply physical
    convergence or identifiability of components. 'standard' retains the prior
    acceptance gates (large residual, poor identifiability, large extrapolated
    area, or high window sensitivity), treating R-squared and trajectory-span
    checks as cautions. 'strict' rejects any flagged fit. include_flagged=True
    overrides those rejections only for successful numerical fits.
    report=True prints per-chain/site fit parameters and diagnostics; the same
    formatted report is available afterward via result.report(). With many
    residues, use result.report(max_rows=20) to limit output.

    Numeric fit limits are physical lag times in ns, inclusive, AFTER kinetic
    scaling. ACF construction and original chain/residue selection are unchanged.
    Mapping inputs preserve their keys. Each chain/site is fitted separately,
    then site times are averaged within chains and chain means averaged equally.
    residue_averaging='common' requires accepted sites in EVERY selected chain;
    'available' can mix residue coverage and must be used with caution. All
    across-chain SD/SEM values remain NaN for fewer than two contributing chains.
    """
    if isinstance(correlations, Mapping):
        if not correlations:
            raise ValueError("correlations mapping is empty")
        return {
            key: fit_reconfiguration_times(
                value,
                fit_tmin=fit_tmin,
                fit_tmax=fit_tmax,
                model=model,
                fit_amplitude=fit_amplitude,
                min_fit_points=min_fit_points,
                min_r_squared=min_r_squared,
                min_decay_times=min_decay_times,
                min_observation_times=min_observation_times,
                include_flagged=include_flagged,
                residue_averaging=residue_averaging,
                warn=warn,
                auto_window=auto_window,
                auto_cutoff=auto_cutoff,
                loss=loss,
                f_scale=f_scale,
                lag_weighting=lag_weighting,
                max_fit_points=max_fit_points,
                check_stability=check_stability,
                max_window_sensitivity=max_window_sensitivity,
                max_tail_fraction=max_tail_fraction,
                model_improvement=model_improvement,
                min_model_improvement=min_model_improvement,
                max_fit_rmse=max_fit_rmse,
                quality_policy=quality_policy,
                report=report,
            )
            for key, value in correlations.items()
        }
    if not isinstance(correlations, ReconfigurationCorrelationResult):
        raise TypeError("correlations must come from calculate_reconfiguration_correlations")
    aliases = {
        "single": "single",
        "exponential": "single",
        "mono": "single",
        "double": "double",
        "biexponential": "double",
        "bi": "double",
        "auto": "auto",
        "stretched": "stretched",
        "kww": "stretched",
    }
    if model not in aliases:
        raise ValueError("model must be 'auto', 'single', 'double', or 'stretched'")
    model = aliases[model]
    if residue_averaging not in {"common", "available"}:
        raise ValueError("residue_averaging must be 'common' or 'available'")
    for name, value in (
        ("fit_amplitude", fit_amplitude),
        ("include_flagged", include_flagged),
        ("warn", warn),
        ("report", report),
        ("auto_window", auto_window),
        ("check_stability", check_stability),
    ):
        if not isinstance(value, (bool, np.bool_)):
            raise TypeError(f"{name} must be a boolean")
    min_fit_points = _integer(min_fit_points, "min_fit_points", 3)
    if min_r_squared is not None and (not np.isfinite(min_r_squared) or min_r_squared > 1):
        raise ValueError("min_r_squared must be finite and <=1, or None")
    for name, value in (
        ("min_decay_times", min_decay_times),
        ("min_observation_times", min_observation_times),
    ):
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if loss not in {"linear", "soft_l1"}:
        raise ValueError("loss must be 'linear' or 'soft_l1'")
    if lag_weighting not in {"uniform", "balanced"}:
        raise ValueError("lag_weighting must be 'uniform' or 'balanced'")
    if quality_policy not in {"permissive", "standard", "strict"}:
        raise ValueError("quality_policy must be 'permissive', 'standard', or 'strict'")
    for name, value in (("f_scale", f_scale), ("max_fit_rmse", max_fit_rmse)):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    for name, value in (
        ("auto_cutoff", auto_cutoff),
        ("max_tail_fraction", max_tail_fraction),
        ("model_improvement", model_improvement),
    ):
        if not np.isfinite(value) or not 0 < value < 1:
            raise ValueError(f"{name} must be between 0 and 1")
    for name, value in (
        ("max_window_sensitivity", max_window_sensitivity),
        ("min_model_improvement", min_model_improvement),
    ):
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    max_fit_points = _integer(max_fit_points, "max_fit_points", max(8, min_fit_points))
    time = np.asarray(correlations.lag_time_ns, dtype=float)
    data = np.asarray(correlations.correlation)
    if (
        time.ndim != 1
        or len(time) < 2
        or not np.all(np.isfinite(time))
        or np.any(np.diff(time) <= 0)
    ):
        raise ValueError("invalid lag_time_ns")
    if data.ndim != 3 or data.shape[0] != correlations.n_chains or data.shape[2] != len(time):
        raise ValueError("invalid correlation shape")
    tmin = _time(fit_tmin, "fit_tmin")
    tmax = float(time[-1]) if fit_tmax is None else _time(fit_tmax, "fit_tmax")
    if not 0 <= tmin < tmax or tmax > time[-1] + 1e-10 * max(1, time[-1]):
        raise ValueError("fit limits must satisfy 0 <= fit_tmin < fit_tmax <= last saved lag")
    fits = {}
    tau_grid = np.full(data.shape[:2], np.nan)
    accepted = np.zeros(data.shape[:2], dtype=bool)
    n_fitted = 0
    for ci, label in enumerate(correlations.chain_labels):
        chain_fits = []
        for ri in range(data.shape[1]):
            curve = data[ci, ri].astype(float)
            if not np.isfinite(curve[0]):
                states = correlations.component_status[ci, ri]
                reason = "constant" if "constant" in states else "missing_or_insufficient_data"
                fitted = _empty_fit(model, reason)
            else:
                n_fitted += 1
                duration = correlations.max_contiguous_time_ns
                if correlations.contiguous_time_per_site_ns is not None:
                    duration = float(correlations.contiguous_time_per_site_ns[ci, ri])
                fitted = _fit_one_automatic(
                    time,
                    curve,
                    model=model,
                    fit_amplitude=fit_amplitude,
                    fit_tmin=tmin,
                    fit_tmax=tmax,
                    physical_dt=correlations.dt_ns * correlations.kinetic_scale,
                    duration=duration,
                    min_fit_points=min_fit_points,
                    min_r_squared=min_r_squared,
                    min_decay_times=min_decay_times,
                    min_observation_times=min_observation_times,
                    include_flagged=include_flagged,
                    automatic_window=bool(auto_window and fit_tmax is None),
                    auto_cutoff=auto_cutoff,
                    loss=loss,
                    f_scale=f_scale,
                    lag_weighting=lag_weighting,
                    max_fit_points=max_fit_points,
                    check_stability=check_stability,
                    max_window_sensitivity=max_window_sensitivity,
                    max_tail_fraction=max_tail_fraction,
                    model_improvement=model_improvement,
                    min_model_improvement=min_model_improvement,
                    max_fit_rmse=max_fit_rmse,
                    quality_policy=quality_policy,
                )
            chain_fits.append(fitted)
            if fitted.accepted:
                tau_grid[ci, ri] = fitted.tau_ns
                accepted[ci, ri] = True
        fits[label] = tuple(chain_fits)
    profile_mean, profile_std, profile_sem, profile_count = _stats(tau_grid, axis=0)
    if correlations.kind == "torsions" and residue_averaging == "common":
        common = np.all(accepted, axis=0)
        for_summary = np.where(common[None, :], tau_grid, np.nan)
        summary_numbers = correlations.residue_numbers[common].copy()
    else:
        for_summary = tau_grid
        summary_numbers = (
            None
            if correlations.residue_numbers is None
            else correlations.residue_numbers[np.any(accepted, axis=0)].copy()
        )
    chain_mean, _, _, chain_n_residues = _stats(for_summary, axis=1)
    mean, sd, sem, n_chains = _stats(chain_mean, axis=0)
    n_accepted = int(accepted.sum())
    label = correlations.dataset_label
    if warn and n_accepted < n_fitted:
        reasons = sorted(
            {
                reason
                for entries in fits.values()
                for item in entries
                if not item.accepted
                for reason in (item.rejection_flags or item.flags)
            }
        )
        warnings.warn(
            f"{label}: accepted {n_accepted}/{n_fitted} fluctuating chain/site fits. "
            f"Reasons: {', '.join(reasons)}. Inspect .fits[chain][site].flags.",
            RuntimeWarning,
            stacklevel=2,
        )
    if warn and quality_policy != "permissive":
        cautions = sorted(
            {
                reason
                for entries in fits.values()
                for item in entries
                if item.accepted
                for reason in item.flags
            }
        )
        if cautions:
            warnings.warn(
                f"{label}: accepted fits carry cautions: {', '.join(cautions)}. "
                "Accepted is not a convergence guarantee; inspect fit diagnostics.",
                RuntimeWarning,
                stacklevel=2,
            )
    if warn and not int(n_chains):
        warnings.warn(
            f"{label}: no comparable accepted sites remain for the summary; mean_tau_ns is NaN. "
            "Inspect fit diagnostics and residue coverage rather than treating this as zero relaxation time.",
            RuntimeWarning,
            stacklevel=2,
        )
    result = ReconfigurationTimeResult(
        fits=fits,
        chain_labels=correlations.chain_labels,
        residue_numbers=correlations.residue_numbers,
        observables=correlations.observables,
        kind=correlations.kind,
        model=model,
        tau_per_chain_residue_ns=tau_grid,
        accepted=accepted,
        per_chain_tau_ns=chain_mean,
        mean_tau_ns=float(mean),
        std_tau_ns=float(sd),
        sem_tau_ns=float(sem),
        n_chains=int(n_chains),
        n_total_chains=correlations.n_chains,
        residue_mean_tau_ns=profile_mean,
        residue_std_tau_ns=profile_std,
        residue_sem_tau_ns=profile_sem,
        residue_n_chains=profile_count,
        summary_residue_numbers=summary_numbers,
        summary_n_residues_per_chain=chain_n_residues,
        residue_averaging=residue_averaging,
        n_fitted=n_fitted,
        n_accepted=n_accepted,
        kinetic_scale=correlations.kinetic_scale,
        source_frame_start=correlations.frame_start,
        source_frame_stop=correlations.frame_stop,
        source_dt_ns=correlations.dt_ns,
        dataset_label=label,
    )
    if report:
        print(result.report())
    return result


def _plot_scale(time_unit: str) -> tuple[float, str]:
    name = time_unit.lower().replace("µ", "u").replace("μ", "u")
    options = {
        "ps": (1e3, "ps"),
        "ns": (1.0, "ns"),
        "us": (1e-3, "µs"),
        "ms": (1e-6, "ms"),
        "s": (1e-9, "s"),
    }
    if name not in options:
        raise ValueError("time_unit must be ps, ns, us, ms, or s")
    return options[name]


def _plot_items(data: Any, cls: type) -> list[tuple[str, Any]]:
    if isinstance(data, cls):
        return [(data.dataset_label, data)]
    if isinstance(data, Mapping) and data and all(isinstance(x, cls) for x in data.values()):
        return [(str(k), v) for k, v in data.items()]
    raise TypeError(f"expected {cls.__name__} or a nonempty flat mapping of these results")


def _save_plot(fig: Any, path: Any, dpi: float) -> None:
    if path is not None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination, dpi=dpi, bbox_inches="tight")


def plot_reconfiguration_correlations(
    correlations: Any,
    *,
    fits: Any = None,
    include_flagged_fits: bool = False,
    show_fit_parameters: bool = False,
    resnum: Optional[int] = None,
    show_chains: bool = False,
    uncertainty: Optional[str] = "sem",
    time_unit: str = "ns",
    xscale: str = "linear",
    yscale: str = "linear",
    xlim: Optional[tuple[float, float]] = None,
    ylim: Optional[tuple[float, float]] = None,
    labels: Optional[Mapping[str, str]] = None,
    title: Optional[str] = None,
    figsize: tuple[float, float] = (6.0, 4.0),
    ax: Any = None,
    legend: bool = True,
    save_path: Any = None,
    save_dpi: float = 300,
) -> tuple[Any, Any]:
    """Plot ACFs and, optionally, the average of individual fitted decays.

    Without resnum, residues are averaged equally within each chain, then chains
    equally. With resnum, show that PDB residue. Shading is descriptive SD/SEM
    across chains, not an independent-lag fit uncertainty. The dashed curve is
    the mean of ACCEPTED individual fits, evaluated only within their fitting
    windows; it is NOT exp(-t/mean_tau). It uses the same accepted sites as the
    solid curve whenever fits are supplied, avoiding a coverage mismatch.
    Residue-profile selection is made at correlation calculation time.
    If no selected fit is accepted, show the raw curve rather than a blank plot.
    include_flagged_fits=True overlays all numerically successful fits for
    diagnosis without changing the accepted time-scale summary.
    show_fit_parameters=True includes component parameters in the legend when
    exactly one fitted chain/site contributes to a plotted curve.
    """
    if not isinstance(include_flagged_fits, (bool, np.bool_)):
        raise TypeError("include_flagged_fits must be a boolean")
    if not isinstance(show_fit_parameters, (bool, np.bool_)):
        raise TypeError("show_fit_parameters must be a boolean")
    if uncertainty not in {None, "sem", "std"}:
        raise ValueError("uncertainty must be sem, std, or None")
    factor, unit = _plot_scale(time_unit)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    items = _plot_items(correlations, ReconfigurationCorrelationResult)
    fit_map = {} if fits is None else dict(_plot_items(fits, ReconfigurationTimeResult))
    if fits is not None and len(items) == 1 and isinstance(fits, ReconfigurationTimeResult):
        fit_map = {items[0][0]: fits}
    if fits is not None and any(key not in fit_map for key, _ in items):
        raise KeyError("fits must have the same dataset keys as correlations")
    plotted = False
    for key, raw in items:
        site = None
        if resnum is not None:
            if raw.residue_numbers is None:
                raise ValueError("resnum applies only to torsion correlations")
            matched = np.flatnonzero(raw.residue_numbers == _integer(resnum, "resnum"))
            if len(matched) != 1:
                raise ValueError(f"{key}: residue {resnum} not present")
            site = int(matched[0])
        grid = np.asarray(raw.correlation, dtype=float)
        fit = fit_map.get(key)
        predictions = None
        if fit is not None:
            if fit.chain_labels != raw.chain_labels or not np.array_equal(
                fit.residue_numbers, raw.residue_numbers
            ):
                raise ValueError(
                    f"{key}: fits and correlations have different chain/residue selections"
                )
            if fit.observables != raw.observables or fit.kinetic_scale != raw.kinetic_scale:
                raise ValueError(
                    f"{key}: fits and correlations have different observables or time scaling"
                )
            if (
                fit.source_frame_start != raw.frame_start
                or fit.source_frame_stop != raw.frame_stop
                or fit.source_dt_ns != raw.dt_ns
            ):
                raise ValueError(f"{key}: fits and correlations have different source time windows")
            mask = fit.accepted.copy()
            if include_flagged_fits:
                mask = np.asarray(
                    [
                        [item.optimizer_success for item in fit.fits[chain]]
                        for chain in raw.chain_labels
                    ],
                    dtype=bool,
                )
            if (
                not include_flagged_fits
                and site is None
                and fit.kind == "torsions"
                and fit.residue_averaging == "common"
            ):
                mask &= np.isin(raw.residue_numbers, fit.summary_residue_numbers)[None, :]
            has_selected_fit = np.any(mask) if site is None else np.any(mask[:, site])
            if has_selected_fit:
                grid = np.where(mask[:, :, None], grid, np.nan)
            else:
                warnings.warn(
                    f"{key}: no accepted fits; showing the original correlation without "
                    "a fitted overlay. Set include_flagged_fits=True for diagnostics.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            predictions = np.full_like(grid, np.nan)
            for ci, chain in enumerate(raw.chain_labels):
                for ri, item in enumerate(fit.fits[chain]):
                    if mask[ci, ri]:
                        in_window = (raw.lag_time_ns >= item.fit_tmin_ns) & (
                            raw.lag_time_ns <= item.fit_tmax_ns
                        )
                        predictions[ci, ri, in_window] = item.predict(raw.lag_time_ns[in_window])
        rows = _stats(grid, axis=1)[0] if site is None else grid[:, site, :]
        mean, sd, sem, _ = _stats(rows, axis=0)
        x = raw.lag_time_ns * factor
        good = np.isfinite(mean)
        if xscale == "log":
            good &= x > 0
        if yscale == "log":
            good &= mean > 0
        if not np.any(good):
            continue
        label = labels.get(key, key) if labels is not None else key
        line = ax.plot(x[good], mean[good], label=label, linewidth=1.8)[0]
        plotted = True
        color = line.get_color()
        if uncertainty is not None:
            error = sem if uncertainty == "sem" else sd
            fill = good & np.isfinite(error)
            if yscale == "log":
                fill &= mean - error > 0
            ax.fill_between(x, mean - error, mean + error, where=fill, alpha=0.2, color=color)
        if show_chains:
            for row in rows:
                keep = np.isfinite(row) & ((x > 0) if xscale == "log" else True)
                if yscale == "log":
                    keep &= row > 0
                ax.plot(x[keep], row[keep], alpha=0.25, linewidth=0.7, color=color)
        if predictions is not None:
            prediction_rows = (
                _stats(predictions, axis=1)[0] if site is None else predictions[:, site, :]
            )
            prediction_mean = _stats(prediction_rows, axis=0)[0]
            keep = np.isfinite(prediction_mean) & ((x > 0) if xscale == "log" else True)
            if yscale == "log":
                keep &= prediction_mean > 0
            if np.any(keep):
                used = [
                    item
                    for chain in raw.chain_labels
                    for ri, item in enumerate(fit.fits[chain])
                    if (site is None or ri == site)
                    and (item.accepted or (include_flagged_fits and item.optimizer_success))
                ]
                models = "/".join(sorted({item.model for item in used}))
                suffix = " (includes flagged)" if any(not item.accepted for item in used) else ""
                fit_label = f"{label}: {models} fit{suffix}"
                if show_fit_parameters and len(used) == 1:
                    item = used[0]
                    if item.model == "double":
                        fit_label += (
                            f" (tau_fast={item.tau_fast_ns:.3g}, "
                            f"tau_slow={item.tau_slow_ns:.3g} ns; "
                            f"f={item.fast_fraction:.2g}; "
                            f"tau_eff={item.tau_ns:.3g} ns)"
                        )
                    elif item.model == "stretched":
                        fit_label += (
                            f" (tau_scale={item.tau_scale_ns:.3g} ns; "
                            f"beta={item.stretch_beta:.2g}; "
                            f"tau_eff={item.tau_ns:.3g} ns)"
                        )
                    else:
                        fit_label += f" (tau={item.tau_ns:.3g} ns)"
                    if item.amplitude != 1.0:
                        fit_label += f" A={item.amplitude:.2g}"
                ax.plot(
                    x[keep], prediction_mean[keep], linestyle="--", color=color, label=fit_label
                )
    if not plotted:
        raise ValueError(
            "no finite curves to plot (or no accepted fits); omit fits to inspect all ACFs"
        )
    ax.set(
        xlabel=f"lag time ({unit})",
        ylabel="connected autocorrelation",
        xscale=xscale,
        yscale=yscale,
    )
    if title is not None:
        ax.set_title(title)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if legend:
        ax.legend()
    _save_plot(fig, save_path, save_dpi)
    return fig, ax


def plot_reconfiguration_times(
    fits: Any,
    *,
    uncertainty: Optional[str] = "sem",
    show_chains: bool = False,
    min_chains: int = 1,
    time_unit: str = "ns",
    yscale: str = "log",
    xlim: Optional[tuple[float, float]] = None,
    ylim: Optional[tuple[float, float]] = None,
    labels: Optional[Mapping[str, str]] = None,
    title: Optional[str] = None,
    figsize: tuple[float, float] = (8.0, 4.0),
    ax: Any = None,
    legend: bool = True,
    save_path: Any = None,
    save_dpi: float = 300,
) -> tuple[Any, Any]:
    """Plot per-residue mean fitted times, with chain SD or SEM error bars.

    Uses PDB residue numbers, not 0-based row indices. Missing/unresolved fits
    remain gaps. min_chains can require a fixed minimum coverage per residue;
    inspect residue_n_chains when comparing profiles. For one chain the point
    is plotted with no error bar, because chain variation is not estimable.
    """
    if uncertainty not in {None, "sem", "std"}:
        raise ValueError("uncertainty must be sem, std, or None")
    min_chains = _integer(min_chains, "min_chains", 1)
    factor, unit = _plot_scale(time_unit)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    plotted = False
    for key, fit in _plot_items(fits, ReconfigurationTimeResult):
        if fit.residue_numbers is None:
            raise ValueError("residue profiles require torsion fits, not scalar/Rg fits")
        # Sort residue numbers for line plots; explicit analysis ordering may differ.
        order = np.argsort(fit.residue_numbers)
        x = fit.residue_numbers[order]
        y = fit.residue_mean_tau_ns[order] * factor
        y = np.where(fit.residue_n_chains[order] >= min_chains, y, np.nan)
        good = np.isfinite(y) & ((y > 0) if yscale == "log" else True)
        if not np.any(good):
            continue
        plotted = True
        label = labels.get(key, key) if labels is not None else key
        line = ax.plot(
            x, np.where(good, y, np.nan), marker="o", markersize=3, linewidth=1.0, label=label
        )[0]
        color = line.get_color()
        if uncertainty is not None:
            error = (fit.residue_sem_tau_ns if uncertainty == "sem" else fit.residue_std_tau_ns)[
                order
            ] * factor
            keep = good & np.isfinite(error)
            if yscale == "log":
                if np.any(keep & (y - error <= 0)):
                    warnings.warn(
                        "Some error bars extend to nonpositive times and cannot be shown on a log axis; "
                        "use yscale='linear' to display those uncertainties.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                keep &= y - error > 0
            ax.errorbar(x[keep], y[keep], yerr=error[keep], fmt="none", capsize=2, color=color)
        if show_chains:
            for row in fit.tau_per_chain_residue_ns[:, order]:
                ax.plot(x, row * factor, alpha=0.25, linewidth=0.6, color=color)
    if not plotted:
        raise ValueError("no accepted residue fits meet min_chains; inspect fit flags")
    ax.set(xlabel="PDB residue number", ylabel=f"reconfiguration time ({unit})", yscale=yscale)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if title is not None:
        ax.set_title(title)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if legend:
        ax.legend()
    _save_plot(fig, save_path, save_dpi)
    return fig, ax
