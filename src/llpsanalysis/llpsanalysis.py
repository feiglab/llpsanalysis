from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np

_FILE_RE_TEMPLATE = r"^r(?P<replica>\d+),p(?P<peptide>\d+)\.{tag}$"


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

    Raises
    ------
    FileNotFoundError
        If no matching files are found.
    ValueError
        If files have too few columns, inconsistent time arrays within a
        replica, or inconsistent numbers of value columns within a replica.
    """
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")

    base_dir = _resolve_data_dir(directory)
    pattern = re.compile(_FILE_RE_TEMPLATE.format(tag=re.escape(tag)))

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
                if time_ref.shape != time_ns.shape or not np.allclose(
                    time_ref,
                    time_ns,
                    rtol=0.0,
                    atol=0.0,
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


def _resolve_data_dir(directory: str | Path) -> Path:
    base = Path(directory)
    if not base.exists():
        raise FileNotFoundError(f"directory does not exist: {base}")

    direct_files = [p for p in base.iterdir() if p.is_file()]
    if direct_files:
        return base

    results_dir = base / "results"
    if results_dir.is_dir():
        return results_dir

    return base


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
