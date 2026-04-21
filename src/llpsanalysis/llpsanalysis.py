from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np

_FILE_RE_TEMPLATE = r"^r(?P<replica>\d+).p(?P<peptide>\d+)\.{tag}$"


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


def read_ascii_timeseries(
    directory: str | Path,
    tag: str,
    *,
    stride: int = 1,
    time_range_ns: tuple[float | None, float | None] | None = None,
    angstrom_to_nm: bool = False,
) -> dict[int, dict[str, Any]]:
    """
    Read per-peptide time series from ASCII files.

    Files are searched first in `directory`, then in `directory / "results"`.
    Expected file names follow the pattern `r?,p?.tag`, with integer replica
    and peptide indices.

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
    """
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")

    pattern = re.compile(_FILE_RE_TEMPLATE.format(tag=re.escape(tag)))
    base_dir = _resolve_data_dir(directory, pattern)

    files_by_replica: dict[int, dict[int, Path]] = {}
    for path in sorted(base_dir.iterdir()):
        if not path.is_file():
            continue
        match = pattern.match(path.name)
        if match is None:
            continue
        replica = int(match.group("replica"))
        peptide = int(match.group("peptide"))
        files_by_replica.setdefault(replica, {})
        if peptide in files_by_replica[replica]:
            prev = files_by_replica[replica][peptide]
            raise ValueError(
                f"duplicate file for replica {replica}, peptide {peptide}: " f"{prev} and {path}"
            )
        files_by_replica[replica][peptide] = path

    if not files_by_replica:
        raise FileNotFoundError(f"no files matching pattern 'r?,p?.{tag}' found in {base_dir}")

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
