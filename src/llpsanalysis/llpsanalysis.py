from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np


def _resolve_aux_file(directory: str | Path, filename: str) -> Path:
    base = Path(directory)
    if not base.is_dir():
        raise FileNotFoundError(f"directory does not exist: {base}")

    direct = base / filename
    if direct.is_file():
        return direct

    results = base / "results" / filename
    if results.is_file():
        return results

    raise FileNotFoundError(f"file '{filename}' not found in {base} or {base / 'results'}")


_FILE_RE_TEMPLATE = r"^r(?P<replica>\d+)(?:\.p(?P<peptide>\d+))?\.(?P<tag>{tag})$"


def read_ascii_timeseries(
    directory: str | Path,
    tag: str,
    *,
    stride: int = 1,
    time_range_ns: tuple[float | None, float | None] | None = None,
    angstrom_to_nm: bool = False,
) -> dict[int, dict[str, Any]]:
    """
    Read ASCII time series data.

    Files are searched first in `directory`, then in `directory / "results"`.

    Supported file name patterns are:
      - per-peptide: r?.p?.tag
      - per-replica: r?.tag

    Input format:
      - at least 4 whitespace-separated columns
      - column 3: time in ps
      - columns 4+: data values (float)

    Conversions:
      - time is converted from ps to ns
      - data are optionally converted from Angstrom to nm

    Filtering:
      - `stride` keeps every nth frame after time filtering
      - `time_range_ns=(tmin, tmax)` keeps frames with
        `tmin <= time <= tmax`; either bound may be None

    Returns
    -------
    dict[int, dict[str, Any]]
        Keys are replica indices. Each value is a dict with:
          - "values": ndarray, shape (n_peptides, n_values, n_frames)
          - "time_ns": ndarray, shape (n_frames,)
          - "peptides": ndarray, shape (n_peptides,)

        For replica-only input files (`r?.tag`), `n_peptides` is 1 and
        `peptides` is `[1]`.

    Raises
    ------
    FileNotFoundError
        If no matching files are found.
    ValueError
        If files have inconsistent time grids or inconsistent numbers of
        value columns within a replica.
    """
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")

    pattern = re.compile(_FILE_RE_TEMPLATE.format(tag=re.escape(tag)))
    base_dir = _resolve_data_dir(directory, pattern)

    files_by_replica: dict[int, dict[int, Path]] = {}
    has_peptide_files = False
    has_replica_only_files = False

    for path in sorted(base_dir.iterdir()):
        if not path.is_file():
            continue

        match = pattern.match(path.name)
        if match is None:
            continue

        replica = int(match.group("replica"))
        peptide_group = match.group("peptide")

        if peptide_group is None:
            peptide = 1
            has_replica_only_files = True
        else:
            peptide = int(peptide_group)
            has_peptide_files = True

        files_by_replica.setdefault(replica, {})
        if peptide in files_by_replica[replica]:
            prev = files_by_replica[replica][peptide]
            raise ValueError(
                f"duplicate file for replica {replica}, peptide {peptide}: " f"{prev} and {path}"
            )

        files_by_replica[replica][peptide] = path

    if not files_by_replica:
        raise FileNotFoundError(
            f"no files matching pattern 'r?.p?.{tag}' or 'r?.{tag}' " f"found in {base_dir}"
        )

    if has_peptide_files and has_replica_only_files:
        raise ValueError(
            f"mixed per-peptide and replica-only files found for tag '{tag}' " f"in {base_dir}"
        )

    out: dict[int, dict[str, Any]] = {}
    for replica, peptide_files in sorted(files_by_replica.items()):
        peptide_ids = np.array(sorted(peptide_files), dtype=int)

        time_ref: np.ndarray | None = None
        values_ref: list[np.ndarray] = []
        n_value_cols_ref: int | None = None

        for peptide in peptide_ids:
            path = peptide_files[int(peptide)]
            time_ns, values = _read_one_file(
                path,
                stride=stride,
                time_range_ns=time_range_ns,
                angstrom_to_nm=angstrom_to_nm,
            )

            if time_ref is None:
                time_ref = time_ns
                n_value_cols_ref = values.shape[1]
            else:
                if n_value_cols_ref != values.shape[1]:
                    raise ValueError(
                        "inconsistent number of value columns within replica "
                        f"{replica}: expected {n_value_cols_ref}, got "
                        f"{values.shape[1]} for peptide {peptide} in {path}"
                    )
                if time_ref.shape != time_ns.shape or not np.array_equal(
                    time_ref,
                    time_ns,
                ):
                    raise ValueError(
                        "inconsistent time grid within replica "
                        f"{replica}: peptide {peptide} in {path} has "
                        f"{time_ns.shape[0]} frames, expected "
                        f"{time_ref.shape[0]}"
                    )

            values_ref.append(values)

        assert time_ref is not None
        values_arr = np.stack(values_ref, axis=0)
        values_arr = np.transpose(values_arr, (0, 2, 1))

        out[replica] = {
            "values": values_arr,
            "time_ns": time_ref,
            "peptides": peptide_ids,
        }

    return out


def _resolve_data_dir(directory: str | Path, pattern: re.Pattern[str]) -> Path:
    base = Path(directory)
    if not base.is_dir():
        raise FileNotFoundError(f"directory does not exist: {base}")

    if _has_matching_files(base, pattern):
        return base

    results_dir = base / "results"
    if results_dir.is_dir() and _has_matching_files(results_dir, pattern):
        return results_dir

    raise FileNotFoundError(
        f"no files matching pattern '{pattern.pattern}' found in " f"{base} or {results_dir}"
    )


def _has_matching_files(directory: Path, pattern: re.Pattern[str]) -> bool:
    for path in directory.iterdir():
        if path.is_file() and pattern.match(path.name):
            return True
    return False


def _read_one_file(
    path: Path,
    *,
    stride: int,
    time_range_ns: tuple[float | None, float | None] | None,
    angstrom_to_nm: bool,
) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, dtype=float, ndmin=2)

    if data.shape[1] < 4:
        raise ValueError(f"{path} has {data.shape[1]} columns, but at least 4 are required")

    time_ns = data[:, 2] * 1.0e-3
    values = data[:, 3:]

    if angstrom_to_nm:
        values = values * 0.1

    if time_range_ns is not None:
        tmin, tmax = time_range_ns
        mask = np.ones(time_ns.shape[0], dtype=bool)
        if tmin is not None:
            mask &= time_ns >= tmin
        if tmax is not None:
            mask &= time_ns <= tmax
        time_ns = time_ns[mask]
        values = values[mask]

    if stride != 1:
        time_ns = time_ns[::stride]
        values = values[::stride]

    return time_ns, values


def subset_peptides(
    data: dict[int, dict[str, Any]],
    peptide_indices: list[int] | range | np.ndarray,
) -> dict[int, dict[str, Any]]:
    """
    Extract a peptide subset from the output of `read_ascii_timeseries`.

    Parameters
    ----------
    data
        Output from `read_ascii_timeseries`.
    peptide_indices
        Peptide indices to keep, e.g. `range(1, 14)` or `[1, 2, 3]`.

    Returns
    -------
    dict[int, dict[str, Any]]
        Same structure as input, but with only the selected peptides.

    Raises
    ------
    ValueError
        If any requested peptide index is missing for a replica.
    """
    requested = np.array(list(peptide_indices), dtype=int)
    out: dict[int, dict[str, Any]] = {}

    for replica, replica_data in sorted(data.items()):
        peptides = np.asarray(replica_data["peptides"], dtype=int)
        values = np.asarray(replica_data["values"])
        time_ns = np.asarray(replica_data["time_ns"])

        index_map = {pep: i for i, pep in enumerate(peptides)}
        missing = [pep for pep in requested if pep not in index_map]
        if missing:
            raise ValueError(f"replica {replica} is missing requested peptides: {missing}")

        sel = np.array([index_map[pep] for pep in requested], dtype=int)

        out[replica] = {
            "values": values[sel, :, :],
            "time_ns": time_ns.copy(),
            "peptides": peptides[sel],
        }

    return out


def read_peptide_indices_from_segments(
    directory: str | Path,
    filename: str,
) -> list[int]:
    """
    Read peptide indices from a segment file.

    The file is searched first in `directory`, then in `directory / "results"`.

    Segment names are expected to look like:
      - P000
      - P001
      - P002

    They are converted to 1-based peptide indices:
      - P000 -> 1
      - P001 -> 2
      - P002 -> 3

    Parameters
    ----------
    directory
        Base directory for the dataset.
    filename
        Segment file name, e.g. "rlp.seg".

    Returns
    -------
    list[int]
        Peptide indices in the order found in the file.

    Raises
    ------
    FileNotFoundError
        If the file is not found.
    ValueError
        If no segment names are found or if a token has invalid format.
    """
    path = _resolve_aux_file(directory, filename)

    text = path.read_text(encoding="utf-8")
    tokens = text.split()

    peptide_indices: list[int] = []
    for token in tokens:
        match = re.fullmatch(r"P(\d+)", token)
        if match is None:
            continue
        peptide_indices.append(int(match.group(1)) + 1)

    if not peptide_indices:
        raise ValueError(f"no segment names like 'P000' found in {path}")

    return peptide_indices


def plot_histogram_with_sem(
    data: dict[int, dict[str, Any]],
    *,
    value_index: int = 0,
    peptide_indices: list[int] | range | np.ndarray | None = None,
    bins: int | np.ndarray = 50,
    hist_range: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    density: bool = True,
    xlabel: str | None = None,
    ylabel: str | None = None,
    save_path: str | Path | None = None,
    ax: Any = None,
):
    """
    Plot a histogram with SEM estimated from replica-to-replica variation.

    For each replica, values are pooled over the selected peptides and all
    time steps. A histogram is computed for each replica on a common bin grid.
    The plotted curve is the mean over replicas, with shaded SEM.
    """
    import matplotlib.pyplot as plt

    replica_ids = sorted(data)
    if not replica_ids:
        raise ValueError("data is empty")

    per_replica: list[np.ndarray] = []

    for replica in replica_ids:
        replica_data = data[replica]
        peptides = np.asarray(replica_data["peptides"], dtype=int)
        values = np.asarray(replica_data["values"], dtype=float)

        if values.ndim != 3:
            raise ValueError(
                f"replica {replica} values must have shape " "(n_peptides, n_values, n_frames)"
            )

        if value_index < 0 or value_index >= values.shape[1]:
            raise ValueError(
                f"value_index {value_index} out of range for replica {replica}; "
                f"available indices: 0..{values.shape[1] - 1}"
            )

        if peptide_indices is None:
            sel = np.arange(values.shape[0], dtype=int)
        else:
            requested = np.array(list(peptide_indices), dtype=int)
            index_map = {pep: i for i, pep in enumerate(peptides)}
            missing = [pep for pep in requested if pep not in index_map]
            if missing:
                raise ValueError(f"replica {replica} is missing requested peptides: {missing}")
            sel = np.array([index_map[pep] for pep in requested], dtype=int)

        replica_values = values[sel, value_index, :].reshape(-1)

        hist, bin_edges = np.histogram(
            replica_values,
            bins=bins,
            range=hist_range,
            density=density,
        )
        per_replica.append(hist.astype(float))

    per_replica_arr = np.stack(per_replica, axis=0)
    mean_hist = per_replica_arr.mean(axis=0)

    if per_replica_arr.shape[0] > 1:
        sem_hist = per_replica_arr.std(axis=0, ddof=1)
        sem_hist /= np.sqrt(per_replica_arr.shape[0])
    else:
        sem_hist = np.zeros_like(mean_hist)

    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    ax.step(bin_centers, mean_hist, where="mid")
    ax.fill_between(
        bin_centers,
        mean_hist - sem_hist,
        mean_hist + sem_hist,
        alpha=0.3,
        step="mid",
    )

    if xlabel is None:
        xlabel = f"value[{value_index}]"
    if ylabel is None:
        ylabel = "probability density" if density else "count"

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if hist_range is not None:
        ax.set_xlim(hist_range)
    if ylim is not None:
        ax.set_ylim(ylim)

    if save_path is not None:
        save_path = Path(save_path)
        if save_path.parent and not save_path.parent.exists():
            save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")

    stats = {
        "bin_edges": bin_edges,
        "bin_centers": bin_centers,
        "mean": mean_hist,
        "sem": sem_hist,
        "per_replica": per_replica_arr,
        "replicas": np.array(replica_ids, dtype=int),
    }

    return fig, ax, stats


def calculate_mean_sem(
    data: dict[int, dict[str, Any]],
    *,
    value_index: int = 0,
    peptide_indices: list[int] | range | np.ndarray | None = None,
) -> dict[str, Any]:
    """
    Calculate a single mean and SEM over the selected peptides, using
    replica-to-replica variation for the SEM.
    """
    replica_ids = sorted(data)
    if not replica_ids:
        raise ValueError("data is empty")

    per_replica_means: list[float] = []

    for replica in replica_ids:
        replica_data = data[replica]
        peptides = np.asarray(replica_data["peptides"], dtype=int)
        values = np.asarray(replica_data["values"], dtype=float)

        if value_index < 0 or value_index >= values.shape[1]:
            raise ValueError(
                f"value_index {value_index} out of range for replica {replica}; "
                f"available indices: 0..{values.shape[1] - 1}"
            )

        if peptide_indices is None:
            sel = np.arange(values.shape[0], dtype=int)
        else:
            requested = np.array(list(peptide_indices), dtype=int)
            index_map = {pep: i for i, pep in enumerate(peptides)}
            missing = [pep for pep in requested if pep not in index_map]
            if missing:
                raise ValueError(f"replica {replica} is missing requested peptides: {missing}")
            sel = np.array([index_map[pep] for pep in requested], dtype=int)

        replica_mean = values[sel, value_index, :].mean()
        per_replica_means.append(float(replica_mean))

    per_replica_arr = np.array(per_replica_means, dtype=float)
    mean = float(per_replica_arr.mean())

    if per_replica_arr.shape[0] > 1:
        sem = float(per_replica_arr.std(ddof=1) / np.sqrt(per_replica_arr.shape[0]))
    else:
        sem = 0.0

    return {
        "mean": mean,
        "sem": sem,
        "per_replica": per_replica_arr,
        "replicas": np.array(replica_ids, dtype=int),
    }


def calculate_msd_curves(
    data: dict[int, dict[str, Any]],
    *,
    coord_indices: tuple[int, int, int] = (0, 1, 2),
    peptide_indices: list[int] | range | np.ndarray | None = None,
    max_lag: int | None = None,
) -> dict[int, dict[str, np.ndarray]]:
    """
    Calculate individual MSD curves for each peptide and replica.

    Parameters
    ----------
    data
        Output from `read_ascii_timeseries`. The values array is expected to
        have shape (n_peptides, n_values, n_frames).
    coord_indices
        Indices of the x, y, z coordinates along the value axis.
    peptide_indices
        Optional peptide indices to include. These are peptide IDs from
        `data[replica]["peptides"]`. If None, all peptides are used.
    max_lag
        Maximum lag in frames. If None, use all possible lags.

    Returns
    -------
    dict[int, dict[str, np.ndarray]]
        Keys are replica indices. Each value contains:
          - "peptides": selected peptide IDs, shape (n_peptides,)
          - "lag_time_ns": lag times, shape (n_lags,)
          - "msd": MSD curves, shape (n_peptides, n_lags)

    Notes
    -----
    MSD is computed as a time-origin average:

        MSD(tau) = <|r(t + tau) - r(t)|^2>_t

    for each peptide separately.
    """
    replica_ids = sorted(data)
    if not replica_ids:
        raise ValueError("data is empty")

    if len(coord_indices) != 3:
        raise ValueError("coord_indices must contain exactly three indices")

    out: dict[int, dict[str, np.ndarray]] = {}

    for replica in replica_ids:
        replica_data = data[replica]
        peptides = np.asarray(replica_data["peptides"], dtype=int)
        values = np.asarray(replica_data["values"], dtype=float)
        time_ns = np.asarray(replica_data["time_ns"], dtype=float)

        if values.ndim != 3:
            raise ValueError(
                f"replica {replica} values must have shape " "(n_peptides, n_values, n_frames)"
            )

        n_peptides, n_values, n_frames = values.shape
        if n_frames < 2:
            raise ValueError(f"replica {replica} must contain at least two frames")

        for idx in coord_indices:
            if idx < 0 or idx >= n_values:
                raise ValueError(
                    f"coord index {idx} out of range for replica {replica}; "
                    f"available indices: 0..{n_values - 1}"
                )

        if peptide_indices is None:
            sel = np.arange(n_peptides, dtype=int)
            selected_peptides = peptides
        else:
            requested = np.array(list(peptide_indices), dtype=int)
            index_map = {pep: i for i, pep in enumerate(peptides)}
            missing = [pep for pep in requested if pep not in index_map]
            if missing:
                raise ValueError(f"replica {replica} is missing requested peptides: {missing}")
            sel = np.array([index_map[pep] for pep in requested], dtype=int)
            selected_peptides = peptides[sel]

        coords = values[sel][:, coord_indices, :]
        coords = np.transpose(coords, (0, 2, 1))

        if max_lag is None:
            n_lags = n_frames
        else:
            if max_lag < 0:
                raise ValueError(f"max_lag must be >= 0, got {max_lag}")
            n_lags = min(max_lag + 1, n_frames)

        lag_time_ns = time_ns[:n_lags] - time_ns[0]
        msd = np.empty((coords.shape[0], n_lags), dtype=float)

        msd[:, 0] = 0.0
        for lag in range(1, n_lags):
            disp = coords[:, lag:, :] - coords[:, :-lag, :]
            sq_disp = np.sum(disp * disp, axis=2)
            msd[:, lag] = np.mean(sq_disp, axis=1)

        out[replica] = {
            "peptides": selected_peptides,
            "lag_time_ns": lag_time_ns,
            "msd": msd,
        }

    return out


def calculate_msd_curves_fft(
    data: dict[int, dict[str, Any]],
    *,
    coord_indices: tuple[int, int, int] = (0, 1, 2),
    peptide_indices: list[int] | range | np.ndarray | None = None,
    max_lag: int | None = None,
) -> dict[int, dict[str, np.ndarray]]:
    replica_ids = sorted(data)
    if not replica_ids:
        raise ValueError("data is empty")

    if len(coord_indices) != 3:
        raise ValueError("coord_indices must contain exactly three indices")

    out: dict[int, dict[str, np.ndarray]] = {}

    for replica in replica_ids:
        replica_data = data[replica]
        peptides = np.asarray(replica_data["peptides"], dtype=int)
        values = np.asarray(replica_data["values"], dtype=float)
        time_ns = np.asarray(replica_data["time_ns"], dtype=float)

        if values.ndim != 3:
            raise ValueError(
                f"replica {replica} values must have shape " "(n_peptides, n_values, n_frames)"
            )

        n_peptides, n_values, n_frames = values.shape
        if n_frames < 2:
            raise ValueError(f"replica {replica} must contain at least two frames")

        for idx in coord_indices:
            if idx < 0 or idx >= n_values:
                raise ValueError(
                    f"coord index {idx} out of range for replica {replica}; "
                    f"available indices: 0..{n_values - 1}"
                )

        if peptide_indices is None:
            sel = np.arange(n_peptides, dtype=int)
            selected_peptides = peptides
        else:
            requested = np.array(list(peptide_indices), dtype=int)
            index_map = {pep: i for i, pep in enumerate(peptides)}
            missing = [pep for pep in requested if pep not in index_map]
            if missing:
                raise ValueError(f"replica {replica} is missing requested peptides: {missing}")
            sel = np.array([index_map[pep] for pep in requested], dtype=int)
            selected_peptides = peptides[sel]

        coords = values[sel][:, coord_indices, :]
        coords = np.transpose(coords, (0, 2, 1))

        if max_lag is None:
            n_lags = n_frames
        else:
            if max_lag < 0:
                raise ValueError(f"max_lag must be >= 0, got {max_lag}")
            n_lags = min(max_lag + 1, n_frames)

        lag_time_ns = time_ns[:n_lags] - time_ns[0]
        msd = _msd_fft_batch(coords, n_lags)

        out[replica] = {
            "peptides": selected_peptides,
            "lag_time_ns": lag_time_ns,
            "msd": msd,
        }

    return out


def _msd_fft_batch(coords: np.ndarray, n_lags: int) -> np.ndarray:
    n_peptides, n_frames, n_dim = coords.shape
    if n_dim != 3:
        raise ValueError(f"coords must have shape (n, t, 3), got {coords.shape}")

    r2 = np.sum(coords * coords, axis=2)

    ac = np.zeros((n_peptides, n_frames), dtype=float)
    for dim in range(3):
        ac += _autocorr_fft_1d_batch(coords[:, :, dim])

    csum = np.concatenate(
        [np.zeros((n_peptides, 1), dtype=float), np.cumsum(r2, axis=1)],
        axis=1,
    )

    lags = np.arange(n_lags, dtype=int)
    n_terms = (n_frames - lags).astype(float)

    left = csum[:, [n_frames]] - csum[:, lags]
    right = csum[:, n_frames - lags]

    msd = (left + right - 2.0 * ac[:, :n_lags]) / n_terms[np.newaxis, :]
    msd[:, 0] = 0.0
    return msd


def _autocorr_fft_1d_batch(x: np.ndarray) -> np.ndarray:
    n_signals, n_frames = x.shape
    n_fft = 1 << (2 * n_frames - 1).bit_length()

    fx = np.fft.rfft(x, n=n_fft, axis=1)
    ac = np.fft.irfft(fx * np.conjugate(fx), n=n_fft, axis=1)
    return ac[:, :n_frames]


def calculate_diffusion_constants(
    msd_data: dict[int, dict[str, np.ndarray]],
    *,
    fit_range_ns: tuple[float, float],
    ndim: int = 3,
) -> dict[int, dict[str, np.ndarray]]:
    """
    Calculate diffusion constants from linear fits to MSD curves.

    Parameters
    ----------
    msd_data
        Output from `calculate_msd_curves()` or `calculate_msd_curves_fft()`.
        For each replica, it must contain:
          - "peptides": shape (n_peptides,)
          - "lag_time_ns": shape (n_lags,)
          - "msd": shape (n_peptides, n_lags)
    fit_range_ns
        Lag-time interval `(tmin, tmax)` in ns used for the linear fit.
    ndim
        Dimensionality for the diffusion relation:
            MSD(t) = 2 * ndim * D * t
        Supported values: 1, 2, 3. Default is 3.

    Returns
    -------
    dict[int, dict[str, np.ndarray]]
        Keys are replica indices. Each value contains:
          - "peptides": peptide IDs, shape (n_peptides,)
          - "D": diffusion constants, shape (n_peptides,)
          - "slope": fitted slopes, shape (n_peptides,)
          - "intercept": fitted intercepts, shape (n_peptides,)
          - "fit_range_ns": shape (2,)

    Raises
    ------
    ValueError
        If the fit range is invalid or too few lag points are available.
    """
    if ndim not in (1, 2, 3):
        raise ValueError(f"ndim must be 1, 2, or 3, got {ndim}")

    tmin, tmax = fit_range_ns
    if tmin >= tmax:
        raise ValueError(f"fit_range_ns must satisfy tmin < tmax, got {fit_range_ns}")

    out: dict[int, dict[str, np.ndarray]] = {}

    for replica, replica_data in sorted(msd_data.items()):
        peptides = np.asarray(replica_data["peptides"], dtype=int)
        lag_time_ns = np.asarray(replica_data["lag_time_ns"], dtype=float)
        msd = np.asarray(replica_data["msd"], dtype=float)

        if lag_time_ns.ndim != 1:
            raise ValueError(
                f"replica {replica} lag_time_ns must be 1D, got " f"{lag_time_ns.ndim}D"
            )
        if msd.ndim != 2:
            raise ValueError(
                f"replica {replica} msd must have shape " f"(n_peptides, n_lags), got {msd.shape}"
            )
        if msd.shape[1] != lag_time_ns.shape[0]:
            raise ValueError(
                f"replica {replica} has inconsistent MSD/time shapes: "
                f"{msd.shape} vs {lag_time_ns.shape}"
            )

        mask = (lag_time_ns >= tmin) & (lag_time_ns <= tmax)
        n_fit = int(np.count_nonzero(mask))
        if n_fit < 2:
            raise ValueError(
                f"replica {replica} has only {n_fit} lag points in fit range "
                f"{fit_range_ns}; need at least 2"
            )

        x = lag_time_ns[mask]
        y = msd[:, mask]

        slopes = np.empty(msd.shape[0], dtype=float)
        intercepts = np.empty(msd.shape[0], dtype=float)

        for i in range(msd.shape[0]):
            coeffs = np.polyfit(x, y[i], 1)
            slopes[i] = coeffs[0]
            intercepts[i] = coeffs[1]

        dvals = slopes / (2.0 * float(ndim))

        out[replica] = {
            "peptides": peptides.copy(),
            "D": dvals,
            "slope": slopes,
            "intercept": intercepts,
            "fit_range_ns": np.array([tmin, tmax], dtype=float),
        }

    return out


def average_over_peptides(
    data: dict[int, dict[str, np.ndarray]],
    *,
    key: str = "D",
    peptide_indices: list[int] | range | np.ndarray | None = None,
) -> dict[str, Any]:
    """
    Average a per-peptide quantity over peptides and estimate SEM from
    replica-to-replica variation.

    Parameters
    ----------
    data
        Dictionary keyed by replica. Each replica entry must contain:
          - "peptides": shape (n_peptides,)
          - key: shape (n_peptides,)
    key
        Name of the per-peptide quantity to average, e.g. "D".
    peptide_indices
        Optional peptide IDs to include. If None, all peptides are used.

    Returns
    -------
    dict[str, Any]
        Dictionary with:
          - "mean": float
          - "sem": float
          - "per_replica": ndarray, shape (n_replicas,)
          - "replicas": ndarray, shape (n_replicas,)
          - "peptides": ndarray, selected peptide IDs
          - "key": str

    Raises
    ------
    ValueError
        If input is empty, the key is missing, or requested peptides are
        missing.
    """
    replica_ids = sorted(data)
    if not replica_ids:
        raise ValueError("data is empty")

    peptide_ref: np.ndarray | None = None
    per_replica: list[float] = []

    for replica in replica_ids:
        replica_data = data[replica]

        if key not in replica_data:
            raise ValueError(f"replica {replica} does not contain key '{key}'")

        peptides = np.asarray(replica_data["peptides"], dtype=int)
        values = np.asarray(replica_data[key], dtype=float)

        if peptides.ndim != 1:
            raise ValueError(f"replica {replica} peptides must be 1D, got {peptides.ndim}D")
        if values.ndim != 1:
            raise ValueError(f"replica {replica} '{key}' must be 1D, got {values.ndim}D")
        if peptides.shape[0] != values.shape[0]:
            raise ValueError(
                f"replica {replica} has inconsistent peptide/{key} shapes: "
                f"{peptides.shape} vs {values.shape}"
            )

        if peptide_indices is None:
            sel = np.arange(peptides.shape[0], dtype=int)
            selected_peptides = peptides
        else:
            requested = np.array(list(peptide_indices), dtype=int)
            index_map = {pep: i for i, pep in enumerate(peptides)}
            missing = [pep for pep in requested if pep not in index_map]
            if missing:
                raise ValueError(f"replica {replica} is missing requested peptides: {missing}")
            sel = np.array([index_map[pep] for pep in requested], dtype=int)
            selected_peptides = peptides[sel]

        if peptide_ref is None:
            peptide_ref = selected_peptides.copy()
        elif not np.array_equal(peptide_ref, selected_peptides):
            raise ValueError(f"inconsistent peptide selection/order in replica {replica}")

        per_replica.append(float(np.mean(values[sel])))

    assert peptide_ref is not None
    per_replica_arr = np.array(per_replica, dtype=float)
    mean = float(np.mean(per_replica_arr))

    if per_replica_arr.shape[0] > 1:
        sem = float(np.std(per_replica_arr, ddof=1) / np.sqrt(per_replica_arr.shape[0]))
    else:
        sem = 0.0

    return {
        "key": key,
        "mean": mean,
        "sem": sem,
        "per_replica": per_replica_arr,
        "replicas": np.array(replica_ids, dtype=int),
        "peptides": peptide_ref,
    }


def correct_diffusion_pbc(
    d_data: dict[int, dict[str, np.ndarray]],
    box_data: dict[int, dict[str, Any]],
    *,
    concentration_gl: float | dict[int, float],
    rp_nm: float = 4.0,
    temperature_k: float = 298.0,
    etawater_mpa_s: float = 0.890,
    b: float = 63.0,
    box_mode: str = "xyz",
    box_indices: tuple[int, int, int] = (0, 1, 2),
    d_key: str = "D",
    scale_to_etawater: bool = True,
) -> dict[int, dict[str, np.ndarray]]:
    """
    Correct diffusion constants for periodic-boundary finite-size artifacts.

    The correction added to each diffusion constant is

        delta_D = (k_B T xi) / (6 pi eta L)

    with

        xi = 2.837297 - (4 pi rp^2) / (3 L^2)

    and viscosity estimated from concentration as

        f = 1.0 + 2.5 phi + b phi^2
        phi = conc / 1430
        eta = etawater * f

    Optionally, the final corrected diffusion constants are scaled by

        etawater_mpa_s / 0.890

    to map them to a different water-viscosity reference. This scaling is
    enabled by default and can be disabled with `scale_to_etawater=False`.

    Parameters
    ----------
    d_data
        Output from `calculate_diffusion_constants()`. For each replica, must
        contain at least:
          - "peptides": shape (n_peptides,)
          - d_key: shape (n_peptides,)
    box_data
        Time-series box-size data from `read_ascii_timeseries()`. For each
        replica, expected keys are:
          - "values": shape (n_peptides, n_values, n_frames)
          - "time_ns": shape (n_frames,)
    concentration_gl
        Concentration in g/L. May be a single float applied to all replicas or
        a dict keyed by replica.
    rp_nm
        Diffuser size in nm. Default is 4.0.
    temperature_k
        Temperature in K. Default is 298.0.
    etawater_mpa_s
        Water viscosity in mPa*s. Default is 0.890.
    b
        Quadratic coefficient in the viscosity model. Default is 63.0.
    box_mode
        How to reduce box dimensions to a scalar L:
          - "xyz": average over time of (Lx * Ly * Lz) ** (1/3)
          - "xy": average over time of (Lx * Ly) ** (1/2)
          - "x": average over time of Lx
          - "y": average over time of Ly
          - "z": average over time of Lz
    box_indices
        Indices of x, y, z box lengths in the box_data value axis.
    d_key
        Name of the diffusion field to correct. Default is "D".
    scale_to_etawater
        If True, scale the final corrected diffusion constants by
        etawater_mpa_s / 0.890. Default is True.

    Returns
    -------
    dict[int, dict[str, np.ndarray]]
        Same replica structure as input `d_data`, with the following added:
          - d_key: corrected diffusion constants, shape (n_peptides,)
          - "uncorrected_<d_key>": original diffusion constants
          - "delta_<d_key>": applied correction before optional scaling
          - "scaled_<d_key>": corrected and optionally scaled values
          - "box_length_nm": scalar effective box length
          - "viscosity_mpa_s": scalar viscosity
          - "xi": scalar xi factor
          - "concentration_gl": scalar concentration
          - "scale_factor": scalar applied scaling factor

    Raises
    ------
    ValueError
        If inputs are inconsistent or required keys are missing.
    """
    if d_key == "":
        raise ValueError("d_key must not be empty")

    if rp_nm <= 0.0:
        raise ValueError(f"rp_nm must be > 0, got {rp_nm}")
    if temperature_k <= 0.0:
        raise ValueError(f"temperature_k must be > 0, got {temperature_k}")
    if etawater_mpa_s <= 0.0:
        raise ValueError(f"etawater_mpa_s must be > 0, got {etawater_mpa_s}")
    if len(box_indices) != 3:
        raise ValueError("box_indices must contain exactly three indices")
    if box_mode not in {"xyz", "xy", "x", "y", "z"}:
        raise ValueError(f"box_mode must be one of 'xyz', 'xy', 'x', 'y', 'z', got {box_mode}")

    out: dict[int, dict[str, np.ndarray]] = {}
    k_b = 1.380649e-23
    eta_ref_mpa_s = 0.890

    for replica, replica_data in sorted(d_data.items()):
        if replica not in box_data:
            raise ValueError(f"box_data is missing replica {replica}")
        if d_key not in replica_data:
            raise ValueError(f"replica {replica} does not contain key '{d_key}'")

        peptides = np.asarray(replica_data["peptides"], dtype=int)
        dvals = np.asarray(replica_data[d_key], dtype=float)

        if peptides.ndim != 1:
            raise ValueError(f"replica {replica} peptides must be 1D, got {peptides.ndim}D")
        if dvals.ndim != 1:
            raise ValueError(f"replica {replica} '{d_key}' must be 1D, got {dvals.ndim}D")
        if peptides.shape[0] != dvals.shape[0]:
            raise ValueError(
                f"replica {replica} has inconsistent peptide/{d_key} shapes: "
                f"{peptides.shape} vs {dvals.shape}"
            )

        l_nm = _box_length_from_timeseries(
            box_data[replica],
            box_mode=box_mode,
            box_indices=box_indices,
        )

        conc_gl = _replica_scalar_value(concentration_gl, replica)
        if conc_gl < 0.0:
            raise ValueError(f"concentration must be >= 0, got {conc_gl} for replica {replica}")

        phi = conc_gl / 1430.0
        fvisc = 1.0 + 2.5 * phi + b * phi * phi
        eta_mpa_s = etawater_mpa_s * fvisc

        xi = 2.837297 - (4.0 * np.pi * rp_nm * rp_nm) / (3.0 * l_nm * l_nm)

        eta_pa_s = eta_mpa_s * 1.0e-3
        l_m = l_nm * 1.0e-9

        delta_d_m2_s = (k_b * temperature_k * xi) / (6.0 * np.pi * eta_pa_s * l_m)
        delta_d_nm2_ns = delta_d_m2_s * 1.0e9

        corrected = dvals + delta_d_nm2_ns

        if scale_to_etawater:
            scale_factor = etawater_mpa_s / eta_ref_mpa_s
            corrected = corrected * scale_factor
        else:
            scale_factor = 1.0

        out_replica = dict(replica_data)
        out_replica[d_key] = corrected
        out_replica[f"uncorrected_{d_key}"] = dvals.copy()
        out_replica[f"delta_{d_key}"] = np.full_like(dvals, delta_d_nm2_ns)
        out_replica["box_length_nm"] = np.array(l_nm, dtype=float)
        out_replica["viscosity_mpa_s"] = np.array(eta_mpa_s, dtype=float)
        out_replica["xi"] = np.array(xi, dtype=float)
        out_replica["concentration_gl"] = np.array(conc_gl, dtype=float)
        out_replica["scale_factor"] = np.array(scale_factor, dtype=float)

        out[replica] = out_replica

    return out


def _box_length_from_timeseries(
    replica_box_data: dict[int, Any],
    *,
    box_mode: str,
    box_indices: tuple[int, int, int],
) -> float:
    values = np.asarray(replica_box_data["values"], dtype=float)

    if values.ndim != 3:
        raise ValueError(
            "box values must have shape (n_peptides, n_values, n_frames), got " f"{values.shape}"
        )

    if values.shape[0] != 1:
        raise ValueError(
            "box data must contain exactly one peptide-like entry per replica; "
            f"got {values.shape[0]}"
        )

    n_values = values.shape[1]
    ix, iy, iz = box_indices
    for idx in (ix, iy, iz):
        if idx < 0 or idx >= n_values:
            raise ValueError(
                f"box index {idx} out of range; available indices: " f"0..{n_values - 1}"
            )

    lx = values[0, ix, :]
    ly = values[0, iy, :]
    lz = values[0, iz, :]

    if np.any(lx <= 0.0) or np.any(ly <= 0.0) or np.any(lz <= 0.0):
        raise ValueError("box lengths must be positive")

    if box_mode == "xyz":
        l_eff = np.mean((lx * ly * lz) ** (1.0 / 3.0))
    elif box_mode == "xy":
        l_eff = np.mean((lx * ly) ** 0.5)
    elif box_mode == "x":
        l_eff = np.mean(lx)
    elif box_mode == "y":
        l_eff = np.mean(ly)
    else:
        l_eff = np.mean(lz)

    return float(l_eff)


def _replica_scalar_value(
    value: float | dict[int, float],
    replica: int,
) -> float:
    if isinstance(value, dict):
        if replica not in value:
            raise ValueError(f"missing value for replica {replica}")
        return float(value[replica])

    return float(value)


def plot_msd_by_replica(
    msd_data: dict[int, dict[str, np.ndarray]],
    *,
    peptide_indices: list[int] | range | np.ndarray | None = None,
    time_range_ns: tuple[float | None, float | None] | None = None,
    xlabel: str = "Lag time (ns)",
    ylabel: str = r"MSD (nm$^2$)",
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    save_path: str | Path | None = None,
    figsize: tuple[float, float] | None = None,
    legend_on_right: bool = True,
):
    """
    Plot scalar MSD curves with one panel per replica and a shared y-axis.
    """
    import matplotlib.pyplot as plt

    replica_ids = sorted(msd_data)
    if not replica_ids:
        raise ValueError("msd_data is empty")

    n_replicas = len(replica_ids)
    if figsize is None:
        figsize = (4.0 * n_replicas, 4.0)

    fig, axes = plt.subplots(
        1,
        n_replicas,
        sharey=True,
        figsize=figsize,
        squeeze=False,
    )
    axes = axes[0]

    legend_handles = None
    legend_labels = None

    for ax, replica in zip(axes, replica_ids):
        replica_data = msd_data[replica]
        peptides = np.asarray(replica_data["peptides"], dtype=int)
        lag_time_ns = np.asarray(replica_data["lag_time_ns"], dtype=float)
        msd = np.asarray(replica_data["msd"], dtype=float)

        if peptides.ndim != 1:
            raise ValueError(f"replica {replica} peptides must be 1D, got {peptides.ndim}D")
        if lag_time_ns.ndim != 1:
            raise ValueError(
                f"replica {replica} lag_time_ns must be 1D, got " f"{lag_time_ns.ndim}D"
            )
        if msd.ndim != 2:
            raise ValueError(f"replica {replica} msd must be 2D, got {msd.ndim}D")
        if msd.shape != (peptides.shape[0], lag_time_ns.shape[0]):
            raise ValueError(
                f"replica {replica} has inconsistent shapes: "
                f"peptides={peptides.shape}, time={lag_time_ns.shape}, "
                f"msd={msd.shape}"
            )

        if peptide_indices is None:
            sel = np.arange(peptides.shape[0], dtype=int)
            selected_peptides = peptides
        else:
            requested = np.array(list(peptide_indices), dtype=int)
            index_map = {pep: i for i, pep in enumerate(peptides)}
            missing = [pep for pep in requested if pep not in index_map]
            if missing:
                raise ValueError(f"replica {replica} is missing requested peptides: {missing}")
            sel = np.array([index_map[pep] for pep in requested], dtype=int)
            selected_peptides = peptides[sel]

        if time_range_ns is None:
            mask = np.ones(lag_time_ns.shape[0], dtype=bool)
        else:
            tmin, tmax = time_range_ns
            mask = np.ones(lag_time_ns.shape[0], dtype=bool)
            if tmin is not None:
                mask &= lag_time_ns >= tmin
            if tmax is not None:
                mask &= lag_time_ns <= tmax

        lag_time_sel = lag_time_ns[mask]
        msd_sel = msd[sel][:, mask]

        if lag_time_sel.size == 0:
            raise ValueError(f"replica {replica} has no lag times in range {time_range_ns}")

        for i, peptide in enumerate(selected_peptides):
            ax.plot(lag_time_sel, msd_sel[i], label=f"P{peptide:03d}")

        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

        ax.set_title(f"Replica {replica}")
        ax.set_xlabel(xlabel)

        if xlim is not None:
            ax.set_xlim(xlim)
        elif time_range_ns is not None:
            tmin, tmax = time_range_ns
            if tmin is not None and tmax is not None:
                ax.set_xlim((tmin, tmax))

        if ylim is not None:
            ax.set_ylim(ylim)

    axes[0].set_ylabel(ylabel)

    for ax in axes[1:]:
        ax.tick_params(labelleft=False)

    if legend_handles is not None and legend_labels is not None:
        if legend_on_right:
            fig.legend(
                legend_handles,
                legend_labels,
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                borderaxespad=0.0,
            )
            fig.tight_layout(rect=(0.0, 0.0, 0.88, 1.0))
        else:
            axes[-1].legend()
            fig.tight_layout()
    else:
        fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        if save_path.parent and not save_path.parent.exists():
            save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")

    return fig, axes


def read_density_profile_timeseries(
    directory: str | Path,
    profile_suffix: str,
    *,
    angstrom_to_nm: bool = True,
) -> dict[str, Any]:
    """
    Read all density-profile records from files matching `r?.<profile_suffix>`.

    Files are searched first in `directory`, then in `directory / "results"`.

    Expected line format:
      col 0: time
      col 1: xmin
      col 2: dx
      col 3: nx
      col 4: literal separator, e.g. ::
      col 5..: profile values

    Parameters
    ----------
    directory
        Base directory of the data set.
    profile_suffix
        File suffix after `r<replica>.`, for example
        `rlp.heavygperl.zprofile`.
    angstrom_to_nm
        If True, convert xmin and dx from Angstrom to nm. Default is True.

    Returns
    -------
    dict[str, Any]
        Dictionary with:
          - "directory": input directory
          - "profile_suffix": input suffix
          - "replicas": ndarray, shape (n_records,)
          - "time_ns": ndarray, shape (n_records,)
          - "xmin": ndarray, shape (n_records,)
          - "dx": ndarray, shape (n_records,)
          - "nx": ndarray, shape (n_records,)
          - "values": ndarray, shape (n_records, n_bins)
          - "records": list of raw record dicts
    """
    pattern = re.compile(rf"^r(?P<replica>\d+)\.(?P<suffix>{re.escape(profile_suffix)})$")
    base_dir = _resolve_data_dir(directory, pattern)

    length_scale = 0.1 if angstrom_to_nm else 1.0

    records: list[dict[str, Any]] = []
    found_profile_files = False

    for path in sorted(base_dir.iterdir()):
        if not path.is_file():
            continue

        match = pattern.match(path.name)
        if match is None:
            continue

        found_profile_files = True
        replica = int(match.group("replica"))

        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                fields = line.split()
                if not fields:
                    continue

                if len(fields) < 6:
                    raise ValueError(
                        f"{path}:{line_no} has too few columns for " "density-profile data"
                    )

                time_ns = float(fields[0])
                xmin = float(fields[1]) * length_scale
                dx = float(fields[2]) * length_scale
                nx = int(float(fields[3]))

                if len(fields) < 5 + nx:
                    raise ValueError(
                        f"{path}:{line_no} expected at least {5 + nx} columns "
                        f"for nx={nx}, got {len(fields)}"
                    )

                values = np.array(
                    [float(v) for v in fields[5 : 5 + nx]],
                    dtype=float,
                )

                records.append(
                    {
                        "replica": replica,
                        "time_ns": time_ns,
                        "xmin": xmin,
                        "dx": dx,
                        "nx": nx,
                        "values": values,
                        "source": path,
                        "line_no": line_no,
                    }
                )

    if not found_profile_files:
        raise FileNotFoundError(
            f"no files matching pattern 'r?.{profile_suffix}' found in {base_dir}"
        )

    if not records:
        raise ValueError(
            f"no density-profile records found in files matching "
            f"'r?.{profile_suffix}' in {base_dir}"
        )

    nx_ref = int(records[0]["nx"])
    for rec in records[1:]:
        if rec["nx"] != nx_ref:
            raise ValueError(
                f"inconsistent nx in record from {rec['source']}: "
                f"expected {nx_ref}, got {rec['nx']}"
            )

    return {
        "directory": str(directory),
        "profile_suffix": profile_suffix,
        "replicas": np.array([rec["replica"] for rec in records], dtype=int),
        "time_ns": np.array([rec["time_ns"] for rec in records], dtype=float),
        "xmin": np.array([rec["xmin"] for rec in records], dtype=float),
        "dx": np.array([rec["dx"] for rec in records], dtype=float),
        "nx": np.array([rec["nx"] for rec in records], dtype=int),
        "values": np.stack([rec["values"] for rec in records], axis=0),
        "records": records,
        "angstrom_to_nm": angstrom_to_nm,
    }


def select_density_profile_time(
    profile_data: dict[str, Any],
    time_ns: float,
) -> dict[str, Any]:
    """
    Select the profile records at the closest available time point and compute
    mean, std, and SEM.

    Parameters
    ----------
    profile_data
        Output from `read_density_profile_timeseries()`.
    time_ns
        Target time in ns.

    Returns
    -------
    dict[str, Any]
        Dictionary with:
          - "directory"
          - "profile_suffix"
          - "time_ns_requested"
          - "time_ns_selected"
          - "replicas"
          - "x"
          - "mean"
          - "std"
          - "sem"
          - "per_replica"

    Raises
    ------
    ValueError
        If the selected profile records have inconsistent grids.
    """
    times = np.asarray(profile_data["time_ns"], dtype=float)
    if times.size == 0:
        raise ValueError("profile_data does not contain any time points")

    unique_times = np.unique(times)
    itime = int(np.argmin(np.abs(unique_times - time_ns)))
    selected_time = float(unique_times[itime])

    mask = times == selected_time

    replicas = np.asarray(profile_data["replicas"], dtype=int)[mask]
    xmin = np.asarray(profile_data["xmin"], dtype=float)[mask]
    dx = np.asarray(profile_data["dx"], dtype=float)[mask]
    nx = np.asarray(profile_data["nx"], dtype=int)[mask]
    values = np.asarray(profile_data["values"], dtype=float)[mask]

    xmin_ref = float(xmin[0])
    dx_ref = float(dx[0])
    nx_ref = int(nx[0])

    if not np.all(nx == nx_ref):
        raise ValueError("selected profile records have inconsistent nx")
    if not np.allclose(xmin, xmin_ref):
        raise ValueError("selected profile records have inconsistent xmin")
    if not np.allclose(dx, dx_ref):
        raise ValueError("selected profile records have inconsistent dx")

    x = xmin_ref + dx_ref * np.arange(nx_ref, dtype=float)
    mean = np.mean(values, axis=0)

    if values.shape[0] > 1:
        std = np.std(values, axis=0, ddof=1)
        sem = std / np.sqrt(values.shape[0])
    else:
        std = np.zeros_like(mean)
        sem = np.zeros_like(mean)

    return {
        "directory": profile_data["directory"],
        "profile_suffix": profile_data["profile_suffix"],
        "time_ns_requested": float(time_ns),
        "time_ns_selected": selected_time,
        "replicas": replicas,
        "x": x,
        "mean": mean,
        "std": std,
        "sem": sem,
        "per_replica": values,
    }


def plot_density_profiles(
    datasets: list[dict[str, Any]],
    *,
    labels: list[str] | None = None,
    colors: list[str] | None = None,
    xlabel: str = "z",
    ylabel: str = "Density",
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    show_sem: bool = False,
    linewidth: float = 2.0,
    save_path: str | Path | None = None,
    ax: Any = None,
):
    """
    Plot one or more selected density profiles.

    Parameters
    ----------
    datasets
        List of outputs from `select_density_profile_time()`.
    labels
        Optional legend labels.
    colors
        Optional line colors.
    xlabel
        X-axis label.
    ylabel
        Y-axis label.
    xlim
        Optional x-axis limits.
    ylim
        Optional y-axis limits.
    show_sem
        If True, show SEM shading.
    linewidth
        Line width.
    save_path
        Optional output file path.
    ax
        Optional matplotlib axes.

    Returns
    -------
    tuple[Any, Any]
        `(fig, ax)`
    """
    import matplotlib.pyplot as plt

    if not datasets:
        raise ValueError("datasets is empty")

    n_sets = len(datasets)

    if labels is not None and len(labels) != n_sets:
        raise ValueError(f"labels must have length {n_sets}, got {len(labels)}")
    if colors is not None and len(colors) != n_sets:
        raise ValueError(f"colors must have length {n_sets}, got {len(colors)}")

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    for i, ds in enumerate(datasets):
        x = np.asarray(ds["x"], dtype=float)
        mean = np.asarray(ds["mean"], dtype=float)
        sem = np.asarray(ds["sem"], dtype=float)

        label = labels[i] if labels is not None else ds["directory"]
        color = colors[i] if colors is not None else None

        ax.plot(x, mean, label=label, color=color, linewidth=linewidth)

        if show_sem:
            ax.fill_between(
                x,
                mean - sem,
                mean + sem,
                color=color,
                alpha=0.3,
            )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    ax.legend()

    if save_path is not None:
        save_path = Path(save_path)
        if save_path.parent and not save_path.parent.exists():
            save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")

    return fig, ax


def average_density_over_range(
    profile_data: dict[str, Any],
    z_range: tuple[float, float],
) -> dict[str, Any]:
    """
    Average density over a selected z range.

    Parameters
    ----------
    profile_data
        Output from `select_density_profile_time()`.
    z_range
        Interval `(zmin, zmax)` over which to average the density.

    Returns
    -------
    dict[str, Any]
        Dictionary with:
          - "z_range": ndarray, shape (2,)
          - "mean": float
          - "sem": float
          - "std": float
          - "per_replica": ndarray, shape (n_replicas,)
          - "replicas": ndarray, shape (n_replicas,)

    Raises
    ------
    ValueError
        If the z range is invalid or contains no bins.
    """
    zmin, zmax = z_range
    if zmin >= zmax:
        raise ValueError(f"z_range must satisfy zmin < zmax, got {z_range}")

    x = np.asarray(profile_data["x"], dtype=float)
    per_replica = np.asarray(profile_data["per_replica"], dtype=float)
    replicas = np.asarray(profile_data["replicas"], dtype=int)

    if x.ndim != 1:
        raise ValueError(f"profile_data['x'] must be 1D, got {x.ndim}D")
    if per_replica.ndim != 2:
        raise ValueError(f"profile_data['per_replica'] must be 2D, got {per_replica.ndim}D")
    if per_replica.shape[1] != x.shape[0]:
        raise ValueError(f"inconsistent shapes: x={x.shape}, per_replica={per_replica.shape}")

    mask = (x >= zmin) & (x <= zmax)
    if not np.any(mask):
        raise ValueError(f"no profile bins found in z range {z_range}")

    per_replica_avg = np.mean(per_replica[:, mask], axis=1)
    mean = float(np.mean(per_replica_avg))

    if per_replica_avg.shape[0] > 1:
        std = float(np.std(per_replica_avg, ddof=1))
        sem = float(std / np.sqrt(per_replica_avg.shape[0]))
    else:
        std = 0.0
        sem = 0.0

    return {
        "z_range": np.array([zmin, zmax], dtype=float),
        "mean": mean,
        "sem": sem,
        "std": std,
        "per_replica": per_replica_avg,
        "replicas": replicas,
    }
