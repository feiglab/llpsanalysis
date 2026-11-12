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
from scipy.optimize import least_squares

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
    """One fit; flagged numerical estimates are retained but not accepted by default.

    tau_ns is tau for a single exponential, or the amplitude-weighted mean
    fast_fraction*tau_fast_ns + (1-fast_fraction)*tau_slow_ns for a biexponential.
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

    def predict(self, time_ns: Any) -> np.ndarray:
        time = np.asarray(time_ns, dtype=float)
        if not self.optimizer_success:
            return np.full_like(time, np.nan)
        return self.amplitude * (
            self.fast_fraction * np.exp(-time / self.tau_fast_ns)
            + (1.0 - self.fast_fraction) * np.exp(-time / self.tau_slow_ns)
        )


@dataclass(frozen=True)
class ReconfigurationTimeResult:
    """Per-chain/residue fits plus descriptive chain-level uncertainty.

    tau_per_chain_residue_ns contains only accepted estimates. Consult fits for
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
) -> ReconfigurationDecayFit:
    valid = np.isfinite(correlation) & (time >= fit_tmin) & (time <= fit_tmax)
    x, y = time[valid], correlation[valid]
    n_parameters = (1 if model == "single" else 3) + int(fit_amplitude)
    if x.size < max(min_fit_points, n_parameters + 2):
        return _empty_fit(model, "insufficient_fit_points")
    if not np.any((x > 0) & (y > 0)):
        return _empty_fit(model, "unresolved_at_sample_spacing")
    # Normalize times for the optimizer, but report ns. Fit in correlation space,
    # never log(C): negative/noisy values are retained, not silently clipped.
    scale = max(float(x[-1]), physical_dt)
    u = x / scale
    low = physical_dt / scale / 100.0
    high = max(duration, float(x[-1]), physical_dt) / scale * 100.0
    below = np.flatnonzero((x > 0) & (y <= math.exp(-1)))
    guess = float(x[below[0]]) / scale if below.size else max(1.0, float(x[-1]) / scale)
    guess = float(np.clip(guess, low * 10, high / 10))

    def prediction(parameters: np.ndarray) -> np.ndarray:
        if model == "single":
            decay = np.exp(-u / np.exp(parameters[0]))
        else:
            tau1, tau2 = np.exp(parameters[:2])
            fraction = parameters[2]
            decay = fraction * np.exp(-u / tau1) + (1 - fraction) * np.exp(-u / tau2)
        amplitude = parameters[-1] if fit_amplitude else 1.0
        return amplitude * decay

    if model == "single":
        initial = [[math.log(guess)]]
        lower, upper = [math.log(low)], [math.log(high)]
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
        lower, upper = [math.log(low), math.log(low), 0.001], [
            math.log(high),
            math.log(high),
            0.999,
        ]
    if fit_amplitude:
        initial = [item + [1.0] for item in initial]
        lower += [0.001]
        upper += [2.0]
    best = None
    for guess_values in initial:
        try:
            solution = least_squares(
                lambda p: prediction(p) - y,
                guess_values,
                bounds=(lower, upper),
                max_nfev=4000,
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
    if model == "single":
        tau_fast = tau_slow = float(np.exp(p[0]) * scale)
        fast_fraction = 1.0
    else:
        tau_fast, tau_slow = np.exp(p[:2]) * scale
        fast_fraction = float(p[2])
        if tau_fast > tau_slow:
            tau_fast, tau_slow = tau_slow, tau_fast
            fast_fraction = 1 - fast_fraction
        tau_fast, tau_slow = float(tau_fast), float(tau_slow)
    tau = fast_fraction * tau_fast + (1 - fast_fraction) * tau_slow
    residual = prediction(p) - y
    ss = float(np.sum(residual**2))
    total = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1 - ss / total if total > 1e-20 else float("nan")
    flags = []
    if min_r_squared is not None and (not np.isfinite(r2) or r2 < min_r_squared):
        flags.append("poor_single_decay_fit" if model == "single" else "poor_decay_fit")
    if tau_fast < physical_dt:
        flags.append("faster_than_sampling")
    if float(x[-1] - x[0]) < min_decay_times * tau_slow:
        flags.append("insufficient_decay_window")
    if duration < min_observation_times * tau_slow:
        flags.append("too_few_relaxation_times")
    if np.any(best.active_mask != 0):
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
        rmse=math.sqrt(ss / len(y)),
        n_points=len(y),
        fit_tmin_ns=float(x[0]),
        fit_tmax_ns=float(x[-1]),
        optimizer_success=True,
        accepted=not flags or bool(include_flagged),
        flags=tuple(flags),
    )


def fit_reconfiguration_times(
    correlations: Any,
    *,
    fit_tmin: Any = 0.0,
    fit_tmax: Any = None,
    model: str = "single",
    fit_amplitude: bool = False,
    min_fit_points: int = 8,
    min_r_squared: Optional[float] = 0.8,
    min_decay_times: float = 2.0,
    min_observation_times: float = 10.0,
    include_flagged: bool = False,
    residue_averaging: str = "common",
    warn: bool = True,
) -> Any:
    """Fit each chain/site independently, then summarize times across chains.

    A single ReconfigurationCorrelationResult returns a ReconfigurationTimeResult;
    a mapping returns the same mapping of fitted results. No correlations are
    pooled between chains before fitting. ACF construction must be repeated to
    change chain/residue/time selection; fitting alone can be rerun for new
    lag windows or models without rereading trajectories.

    Default: C_fit(t)=exp(-t/tau). With fit_amplitude=True use A exp(-t/tau).
    model='double'/'biexponential' fits A*[f exp(-t/tau_fast)+(1-f) exp(-t/tau_slow)].
    A is fixed at 1 unless requested. tau_ns then means f*tau_fast+(1-f)*tau_slow,
    i.e. the area under the fitted decay divided by its fitted initial amplitude.
    For a fit starting after a fast transient, this is a TAIL timescale, not the
    area under the complete measured ACF. All models decay to zero; no adjustable
    baseline is used to hide an unresolved slow component.

    Numeric fit limits are PHYSICAL lag times in ns, inclusive, after any
    kinetic_scale used during calculation. Default tmax is the last saved lag.
    Least squares uses equal weights in C-space, retaining negative observations.
    Lag points are correlated, so fit covariance is not used for error bars.

    Quality gates are heuristics, NOT a proof of kinetic convergence:
    - >=8 fitted points and R-squared >=0.8;
    - fastest fitted time >= one input-sample interval;
    - fitting span >=2 times the slowest fitted time;
    - longest continuous observation >=10 times the slowest fitted time;
    - no optimizer-bound, ill-conditioning, or unresolved biexponential components.
    Numerical estimates and flags are always retained in .fits. Flagged estimates
    do not enter means unless include_flagged=True. Change thresholds explicitly
    to relax a gate; inspect curves, support, and stationarity before doing so.

    For torsions, residue_averaging='common' uses residues with accepted fits in
    EVERY selected chain for the one-number summary. 'available' uses each
    chain's accepted residues, which can compare different residue sets and bias
    the summary if missing/unresolved sites are systematically slower. The
    residue profile always reports available-chain means, SD, SEM and counts.
    Do not treat excluded slow/constant sites as fast: inspect coverage and flags.
    Rg simply averages accepted per-chain fits. All SEMs are NaN for <2 chains.
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
    }
    if model not in aliases:
        raise ValueError("model must be 'single' or 'double'/'biexponential'")
    model = aliases[model]
    if residue_averaging not in {"common", "available"}:
        raise ValueError("residue_averaging must be 'common' or 'available'")
    for name, value in (
        ("fit_amplitude", fit_amplitude),
        ("include_flagged", include_flagged),
        ("warn", warn),
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
                fitted = _fit_one(
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
        warnings.warn(
            f"{label}: accepted {n_accepted}/{n_fitted} fluctuating chain/site fits. "
            "Summary is conditional on accepted fits; inspect .fits[chain][site].flags.",
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
    return ReconfigurationTimeResult(
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
    """
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
            if site is None and fit.kind == "torsions" and fit.residue_averaging == "common":
                mask &= np.isin(raw.residue_numbers, fit.summary_residue_numbers)[None, :]
            grid = np.where(mask[:, :, None], grid, np.nan)
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
            ax.plot(
                x[keep],
                prediction_mean[keep],
                linestyle="--",
                color=color,
                label=f"{label}: mean individual fits",
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
