"""Notebook-facing convenience functions for condensate trajectory analysis.

This module intentionally re-exports a selected set of :mod:`mdsim` analysis
functions and OpenMM units used by the accompanying notebooks.  The local
helpers provide result caching, histogram calculation, plotting, and a compact
dispatch layer for common trajectory analyses.
"""

from __future__ import annotations

import importlib
import json
import math
import os
import pickle
import re
import uuid
import warnings
import zipfile
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, fields, is_dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from mdsim import MDSim, Model, PDBReader, StructureSelector
from mdsim.analysis import (
    ChainCOMTrajectoryResult,
    InterchainDistanceResult,
    MassDensityProfileResult,
    MinimumContactDistanceResult,
    MSDFitResult,
    MSDResult,
    PeptideTorsionChainResult,
    PeptideTorsionResult,
    RadialTranslationalDiffusionFitResult,
    RadialTranslationalMSDResult,
    ResidueDensityProfileResult,
    SiteRDFResult,
    TranslationalDiffusionFitResult,
    TranslationalMSDResult,
    chain_com_trajectories_from_dcd,
    fit_msd_linear,
    fit_radial_translational_msd,
    fit_translational_msd,
    interchain_distances_all_references_from_dcd,
    interchain_distances_from_dcd,
    intrachain_distances_from_dcd,
    mass_density_profile_from_dcd,
    minimum_contact_distances_from_dcd,
    msd_from_dcd,
    peptide_torsions_from_dcd,
    radial_translational_msd_from_trajectories,
    reference_center_distances_from_dcd,
    residue_density_profile_from_dcd,
    rg_from_dcd,
    site_rdf_from_dcd,
    translational_msd_from_trajectories,
    yeh_hummer_diffusion_correction,
)
from mdsim.clustering import CondensateResult, condensates_from_dcd, contacts_from_dcd
from mdsim.contact_map import (
    CAContactTrajectoryResult,
    ResidueContactMapResult,
    average_residue_contact_maps,
    ca_contact_trajectory_from_dcd,
    residue_contact_map_from_trajectory,
)
from mdsim.distance_dependent_diffusion import (
    DistanceDiffusion1DFitResult,
    DistanceMSD1DResult,
    DistDiffusionMapFitResult,
    DistMSDMapResult,
    dist_msd_binned_from_dcd,
    dist_msd_binned_from_series,
    dist_msd_map_from_dcd,
    distance_msd_1d_from_dcd,
    distance_msd_1d_from_series,
    fit_dist_msd_map_linear,
    fit_distance_msd_1d,
)
from mdsim.distance_dependent_diffusion import (
    fit_dist_msd_linear_bins as _fit_dist_msd_linear_bins,
)
from mdsim.quenching import (
    BrightnessAutocorrelationResult,
    BrightnessSignalResult,
    FixedHorizonSegmentCurveResult,
    FluorescenceAutocorrelationResult,
    FluorescenceSignalResult,
    QuenchingCurveResult,
    QuenchingDecayResult,
    ReactiveQuenchingCorrelationResult,
    ReactiveQuenchingHazardResult,
    apply_brightness_signal,
    apply_decay,
    apply_fluorescence_signal,
    apply_reactive_quenching_hazard,
    brightness_autocorrelation,
    brightness_autocorrelation_all,
    fluorescence_autocorrelation,
    fluorescence_autocorrelation_all,
    quench,
    quench_all,
    quench_fixed_horizon_segments,
    reactive_quenching_correlation,
    reactive_quenching_correlation_all,
)
from numpy.polynomial.legendre import leggauss
from openmm.unit import (
    angstroms,
    degrees,
    kelvin,
    kilojoule,
    mole,
    nanoseconds,
    radian,
)
from scipy.integrate import cumulative_trapezoid, simpson, trapezoid
from scipy.interpolate import PchipInterpolator
from scipy.linalg import eigh_tridiagonal
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit

PathLike = Union[str, Path]

# Custom colormap: turbo compressed into the first 10% of the range,
# followed by a slow transition to vivid magenta.
_rainbow = plt.get_cmap("turbo")
_compressed_turbo_positions = np.concatenate((np.linspace(0.0, 0.10, 8), [1.0]))
_compressed_turbo_endpoint = np.array([255, 0, 200], dtype=float) / 255.0
_compressed_turbo_colors = [_rainbow(x) for x in np.linspace(0.0, 1.0, 8)] + [
    _compressed_turbo_endpoint
]
compressed_turbo_custom = LinearSegmentedColormap.from_list(
    "compressed_turbo_custom",
    list(zip(_compressed_turbo_positions, _compressed_turbo_colors)),
)


def _normalize_diffusion_fit_mode(mode: Optional[str]) -> Optional[str]:
    """Normalize intra/inter names used by the notebook diffusion-fit wrapper."""
    if mode is None:
        return None
    if not isinstance(mode, str):
        raise TypeError("mode must be 'intra', 'inter', or None")
    value = mode.strip().lower().replace("-", "_")
    aliases = {
        "intra": "intra",
        "intrachain": "intra",
        "inter": "inter",
        "interchain": "inter",
    }
    if value not in aliases:
        raise ValueError("mode must be 'intra' or 'inter'")
    return aliases[value]


def _diffusion_component_for_fit(value: Any, *, mode: Optional[str], label: str) -> Any:
    """Return one binned intra/inter MSD result from a fit input value."""
    if hasattr(value, "msd_rep_nm2") and hasattr(value, "chain_labels"):
        if mode is not None and str(getattr(value, "mode", "")) != mode:
            raise ValueError(
                f"{label!r}: requested mode={mode!r}, but result has "
                f"mode={getattr(value, 'mode', None)!r}"
            )
        return value

    if hasattr(value, "intra") and hasattr(value, "inter"):
        selected_mode = mode
        intra = getattr(value, "intra", None)
        inter = getattr(value, "inter", None)
        if selected_mode is None:
            available = [
                name for name, result in (("intra", intra), ("inter", inter)) if result is not None
            ]
            if len(available) != 1:
                raise ValueError(f"{label!r}: mode='intra' or mode='inter' is required")
            selected_mode = available[0]
        result = intra if selected_mode == "intra" else inter
        if result is None:
            raise ValueError(f"{label!r}: requested {selected_mode!r} result is absent")
        return result

    raise TypeError(
        f"{label!r}: diffusion fit input must contain a binned MSD result or "
        "an intra/inter pair result"
    )


def _diffusion_chain_selection(
    specification: Any,
    *,
    key: str,
    name: str,
) -> Optional[tuple[str, ...]]:
    """Resolve one global or per-dataset exact chain-label selection."""
    if specification is None:
        return None
    value = specification.get(key) if isinstance(specification, Mapping) else specification
    if value is None:
        return None
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} for {key!r} must be a sequence of chain labels")
    labels = tuple(str(item) for item in value)
    if len(set(labels)) != len(labels):
        raise ValueError(f"{name} for {key!r} contains duplicate chain labels")
    return labels


def _subset_binned_diffusion_replicates(
    result: Any,
    *,
    include_chains: Optional[tuple[str, ...]],
    exclude_chains: Optional[tuple[str, ...]],
    strict: bool,
    label: str,
) -> Any:
    """Subset chain/reference-chain MSD replicates and recompute their mean/SEM.

    Intrachain replicates are physical chains. Interchain replicates are reference
    chains; target chains have already been pooled inside each reference replicate
    by the trajectory calculation and therefore cannot be changed at fit time.
    """
    if include_chains is not None and exclude_chains is not None:
        raise ValueError("include_chains and exclude_chains are mutually exclusive")
    if include_chains is None and exclude_chains is None:
        return result

    rep_msd = np.asarray(result.msd_rep_nm2, dtype=np.float64)
    if rep_msd.ndim != 3:
        raise ValueError(f"{label!r}: msd_rep_nm2 must have shape (n_replicates, n_bins, n_lags)")
    labels = tuple(str(x) for x in getattr(result, "chain_labels", ()))
    if len(labels) != rep_msd.shape[0]:
        raise ValueError(
            f"{label!r}: chain_labels contains {len(labels)} entries but "
            f"msd_rep_nm2 contains {rep_msd.shape[0]} replicates"
        )
    if len(set(labels)) != len(labels):
        raise ValueError(f"{label!r}: chain_labels contains duplicates")

    label_to_index = {chain_label: index for index, chain_label in enumerate(labels)}
    if include_chains is not None:
        missing = [name for name in include_chains if name not in label_to_index]
        if missing and strict:
            raise ValueError(
                f"{label!r}: include_chains contains labels not present in diffusion "
                f"result: {missing}. Available labels: {list(labels)}"
            )
        wanted = set(include_chains)
        indices = [i for i, chain_label in enumerate(labels) if chain_label in wanted]
    else:
        assert exclude_chains is not None
        missing = [name for name in exclude_chains if name not in label_to_index]
        if missing and strict:
            raise ValueError(
                f"{label!r}: exclude_chains contains labels not present in diffusion "
                f"result: {missing}. Available labels: {list(labels)}"
            )
        unwanted = set(exclude_chains)
        indices = [i for i, chain_label in enumerate(labels) if chain_label not in unwanted]

    if not indices:
        raise ValueError(f"{label!r}: chain selection leaves no diffusion replicates")

    selected = rep_msd[np.asarray(indices, dtype=np.int64), :, :]
    selected_labels = tuple(labels[i] for i in indices)
    finite = np.isfinite(selected)
    n_eff_i = np.sum(finite, axis=0, dtype=np.int64)
    n_eff = n_eff_i.astype(np.float64)

    mean = np.full(selected.shape[1:], np.nan, dtype=np.float64)
    sums = np.sum(np.where(finite, selected, 0.0), axis=0)
    np.divide(sums, n_eff, out=mean, where=n_eff_i > 0)

    stderr = np.full_like(mean, np.nan, dtype=np.float64)
    stderr[n_eff_i == 1] = 0.0
    enough = n_eff_i > 1
    if np.any(enough):
        deviations = np.where(finite, selected - mean[None, :, :], 0.0)
        ss = np.sum(deviations * deviations, axis=0)
        variance = np.zeros_like(mean, dtype=np.float64)
        np.divide(ss, n_eff - 1.0, out=variance, where=enough)
        stderr[enough] = np.sqrt(variance[enough] / n_eff[enough])

    updates = {
        "msd_nm2": mean,
        "msd_stderr_nm2": stderr,
        "msd_rep_nm2": selected,
        "n_replicates": int(len(indices)),
        "chain_labels": selected_labels,
    }
    # For intrachain data the replicate count is also the selected chain count.
    # For interchain data n_chains still describes the original target-chain pool.
    if str(getattr(result, "mode", "")) == "intra":
        updates["n_chains"] = int(len(indices))

    try:
        return replace(result, **updates)
    except TypeError:
        # Duck-typed fallback for compatible non-dataclass results.
        fields = dict(vars(result))
        fields.update(updates)
        return SimpleNamespace(**fields)


def fit_dist_msd_linear_bins(
    data: Any,
    *args: Any,
    include_chains: Any = None,
    exclude_chains: Any = None,
    strict_chain_selection: bool = True,
    **kwargs: Any,
) -> Any:
    """Fit distance-binned MSDs with optional exact chain-label filtering.

    ``include_chains`` and ``exclude_chains`` follow the same interface used by
    :func:`calculate_quenching_statistics`: either one sequence applied to every
    dataset or a mapping ``dataset_key -> sequence``. They are mutually exclusive.

    Intrachain filtering selects physical-chain replicates. Interchain filtering
    selects reference-chain replicates only; target chains are already pooled in
    the cached interchain MSD result and cannot be removed at fit time.

    A one-entry mapping is unwrapped before calling the mdsim fitter so that
    ``d_stderr_nm2_per_ns`` remains the within-simulation SEM across the selected
    chain/reference-chain replicates. Mappings with two or more simulation sets
    retain equal-set averaging and across-set SEM.
    """
    if include_chains is not None and exclude_chains is not None:
        raise ValueError("include_chains and exclude_chains are mutually exclusive")
    if not isinstance(strict_chain_selection, (bool, np.bool_)):
        raise TypeError("strict_chain_selection must be a boolean")

    mode = _normalize_diffusion_fit_mode(kwargs.get("mode"))

    if isinstance(data, Mapping):
        if not data:
            raise ValueError("data mapping is empty")
        transformed: dict[Any, Any] = {}
        for raw_key, value in data.items():
            key = str(raw_key)
            component = _diffusion_component_for_fit(value, mode=mode, label=key)
            include_for_key = _diffusion_chain_selection(
                include_chains, key=key, name="include_chains"
            )
            exclude_for_key = _diffusion_chain_selection(
                exclude_chains, key=key, name="exclude_chains"
            )
            transformed[raw_key] = _subset_binned_diffusion_replicates(
                component,
                include_chains=include_for_key,
                exclude_chains=exclude_for_key,
                strict=bool(strict_chain_selection),
                label=key,
            )
        data_for_fit: Any = transformed
        if len(transformed) == 1:
            data_for_fit = next(iter(transformed.values()))
    else:
        component = _diffusion_component_for_fit(data, mode=mode, label="data")
        include_one = _diffusion_chain_selection(include_chains, key="data", name="include_chains")
        exclude_one = _diffusion_chain_selection(exclude_chains, key="data", name="exclude_chains")
        data_for_fit = _subset_binned_diffusion_replicates(
            component,
            include_chains=include_one,
            exclude_chains=exclude_one,
            strict=bool(strict_chain_selection),
            label="data",
        )

    return _fit_dist_msd_linear_bins(data_for_fit, *args, **kwargs)


__all__ = [
    # Local helpers
    "AverageResult",
    "OnOffRatesResult",
    "QuenchingAverageResult",
    "BrightnessAutocorrelationResult",
    "BrightnessSignalResult",
    "FluorescenceAutocorrelationResult",
    "FluorescenceSignalResult",
    "FixedHorizonSegmentCurveResult",
    "ReactiveQuenchingCorrelationResult",
    "ReactiveQuenchingHazardResult",
    "QuenchingStatisticsResult",
    "QuenchingFitResult",
    "apply_decay",
    "HistogramResult",
    "RadialNumberDensityResult",
    "ResidueDensityProfileResult",
    "ResidueDensityMapResult",
    "residue_density_profile_from_dcd",
    "calculate_residue_density_maps",
    "plot_residue_density_map",
    "calculate",
    "calculate_averages",
    "calculate_on_off_rates",
    "calculate_translational_msd",
    "calculate_radial_translational_msd",
    "jackknife_rate_sem",
    "summarize_translational_diffusion",
    "TranslationalDiffusionSummary",
    "calculate_distance_msd_1d",
    "calculate_distance_dependent_msd",
    "calculate_distance_msd_binned_1d",
    "calculate_contact_maps",
    "calculate_histograms",
    "calculate_radial_number_density",
    "calculate_quenching_statistics",
    "compressed_turbo_custom",
    "count_chains_above_threshold",
    "get_group_averages",
    "get_interchain_values",
    "get_quenching_curves",
    "get_quenching_fcs_curves",
    "get_quenching_fcs_values",
    "get_quenching_values",
    "get_reactive_quenching_values",
    "get_reactive_quenching_correlation",
    "get_brightness_values",
    "get_brightness_autocorrelation",
    "get_values",
    "load_data",
    "plot_errorbar_datasets",
    "plot_histograms",
    "plot_contact_map",
    "plot_contact_diffusion_triangles",
    "plot_quenching_statistics",
    "plot_timeseries",
    "save_data",
    "fit_quenching_decay",
    "load_quenching_time_series_csv",
    "load_multiple_quenching_time_series_csv",
    # Re-exported mdsim objects and functions used by notebooks
    "CAContactTrajectoryResult",
    "CondensateResult",
    "ResidueContactMapResult",
    "MDSim",
    "MassDensityProfileResult",
    "MinimumContactDistanceResult",
    "PeptideTorsionChainResult",
    "PeptideTorsionResult",
    "peptide_torsions_from_dcd",
    "MSDFitResult",
    "MSDResult",
    "ChainCOMTrajectoryResult",
    "TranslationalMSDResult",
    "TranslationalDiffusionFitResult",
    "RadialTranslationalMSDResult",
    "RadialTranslationalDiffusionFitResult",
    "Model",
    "PDBReader",
    "QuenchingCurveResult",
    "QuenchingDecayResult",
    "apply_fluorescence_signal",
    "brightness_autocorrelation",
    "brightness_autocorrelation_all",
    "fluorescence_autocorrelation",
    "fluorescence_autocorrelation_all",
    "reactive_quenching_correlation",
    "reactive_quenching_correlation_all",
    "apply_brightness_signal",
    "apply_reactive_quenching_hazard",
    "SiteRDFResult",
    "StructureSelector",
    "condensates_from_dcd",
    "ca_contact_trajectory_from_dcd",
    "residue_contact_map_from_trajectory",
    "average_residue_contact_maps",
    "contacts_from_dcd",
    "quench",
    "quench_all",
    "quench_fixed_horizon_segments",
    "DistDiffusionMapFitResult",
    "DistMSDMapResult",
    "DistanceMSD1DResult",
    "DistanceDiffusion1DFitResult",
    "distance_msd_1d_from_series",
    "distance_msd_1d_from_dcd",
    "fit_distance_msd_1d",
    "dist_msd_binned_from_dcd",
    "dist_msd_binned_from_series",
    "dist_msd_map_from_dcd",
    "fit_dist_msd_linear_bins",
    "fit_dist_msd_map_linear",
    "fit_dist_diffusion_maps",
    "plot_dist_diffusion_map",
    "fit_msd_linear",
    "interchain_distances_from_dcd",
    "intrachain_distances_from_dcd",
    "mass_density_profile_from_dcd",
    "minimum_contact_distances_from_dcd",
    "msd_from_dcd",
    "chain_com_trajectories_from_dcd",
    "translational_msd_from_trajectories",
    "radial_translational_msd_from_trajectories",
    "fit_translational_msd",
    "fit_radial_translational_msd",
    "yeh_hummer_diffusion_correction",
    "reference_center_distances_from_dcd",
    "rg_from_dcd",
    "site_rdf_from_dcd",
    # Common OpenMM units retained for notebook compatibility
    "angstroms",
    "degrees",
    "kelvin",
    "kilojoule",
    "mole",
    "nanoseconds",
    "radian",
    "PMFResult",
    "calculate_pmf",
    "calculate_pmfs",
    "plot_pmfs",
    "plot_pmf",
]


_PORTABLE_CACHE_FORMAT = "condensate_analysis_portable_cache"
_PORTABLE_CACHE_VERSION = 1


_LEGACY_CONDENSATE_MODULES = {
    "condensate_analysis": "condensate.condensate_analysis",
    "llpsanalysis.condensate_analysis": "condensate.condensate_analysis",
}


def _portable_cache_class(module_name: str, qualname: str) -> Any:
    """Resolve a cached class name, returning None if it is unavailable."""
    module_name = _LEGACY_CONDENSATE_MODULES.get(module_name, module_name)
    try:
        module = importlib.import_module(module_name)
        value: Any = module
        for component in qualname.split("."):
            value = getattr(value, component)
        return value
    except (ImportError, AttributeError):
        return None


def _portable_cache_encode(
    value: Any,
    archive: zipfile.ZipFile,
    array_counter: list[int],
) -> Any:
    """Encode one cache value as JSON-compatible metadata plus .npy members."""
    if value is None or isinstance(value, (bool, int, str)):
        return value

    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return {"__kind__": "float", "value": repr(value)}

    if isinstance(value, complex):
        return {
            "__kind__": "complex",
            "real": _portable_cache_encode(float(value.real), archive, array_counter),
            "imag": _portable_cache_encode(float(value.imag), archive, array_counter),
        }

    if isinstance(value, np.generic):
        return _portable_cache_encode(value.item(), archive, array_counter)

    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            return {
                "__kind__": "object_array",
                "shape": list(value.shape),
                "items": [
                    _portable_cache_encode(item, archive, array_counter)
                    for item in value.reshape(-1).tolist()
                ],
            }

        index = array_counter[0]
        array_counter[0] += 1
        member = f"arrays/{index:06d}.npy"
        # ``np.save`` streams the array into the ZIP member, so zipfile cannot
        # know the final member size when the member is opened.  Force ZIP64
        # up front; otherwise arrays that grow beyond the classic ZIP 2-GiB
        # streaming limit fail on close even though the archive itself was
        # created with allowZip64=True.
        with archive.open(member, "w", force_zip64=True) as handle:
            np.save(handle, value, allow_pickle=False)
        return {"__kind__": "ndarray", "member": member}

    if isinstance(value, Path):
        return {"__kind__": "path", "value": str(value)}

    if isinstance(value, Mapping):
        return {
            "__kind__": "mapping",
            "items": [
                [
                    _portable_cache_encode(key, archive, array_counter),
                    _portable_cache_encode(item, archive, array_counter),
                ]
                for key, item in value.items()
            ],
        }

    if isinstance(value, tuple):
        return {
            "__kind__": "tuple",
            "items": [_portable_cache_encode(item, archive, array_counter) for item in value],
        }

    if isinstance(value, list):
        return {
            "__kind__": "list",
            "items": [_portable_cache_encode(item, archive, array_counter) for item in value],
        }

    if isinstance(value, set):
        return {
            "__kind__": "set",
            "items": [_portable_cache_encode(item, archive, array_counter) for item in value],
        }

    if isinstance(value, SimpleNamespace):
        return {
            "__kind__": "namespace",
            "state": _portable_cache_encode(vars(value), archive, array_counter),
        }

    if is_dataclass(value) and not isinstance(value, type):
        state = {field.name: getattr(value, field.name) for field in fields(value)}
        cls = type(value)
        return {
            "__kind__": "dataclass",
            "module": cls.__module__,
            "qualname": cls.__qualname__,
            "state": _portable_cache_encode(state, archive, array_counter),
        }

    if hasattr(value, "__dict__"):
        cls = type(value)
        return {
            "__kind__": "object",
            "module": cls.__module__,
            "qualname": cls.__qualname__,
            "state": _portable_cache_encode(vars(value), archive, array_counter),
        }

    raise TypeError(
        "portable cache cannot serialize object of type "
        f"{type(value).__module__}.{type(value).__qualname__}"
    )


def _portable_cache_decode(value: Any, archive: zipfile.ZipFile) -> Any:
    """Reconstruct one value from portable-cache metadata."""
    if not isinstance(value, dict) or "__kind__" not in value:
        return value

    kind = value["__kind__"]

    if kind == "float":
        return float(value["value"])

    if kind == "complex":
        return complex(
            _portable_cache_decode(value["real"], archive),
            _portable_cache_decode(value["imag"], archive),
        )

    if kind == "ndarray":
        with archive.open(value["member"], "r") as handle:
            return np.load(handle, allow_pickle=False)

    if kind == "object_array":
        items = [_portable_cache_decode(item, archive) for item in value["items"]]
        return np.asarray(items, dtype=object).reshape(tuple(value["shape"]))

    if kind == "path":
        return Path(value["value"])

    if kind == "mapping":
        return {
            _portable_cache_decode(key, archive): _portable_cache_decode(item, archive)
            for key, item in value["items"]
        }

    if kind in {"tuple", "list", "set"}:
        items = [_portable_cache_decode(item, archive) for item in value["items"]]
        if kind == "tuple":
            return tuple(items)
        if kind == "set":
            return set(items)
        return items

    if kind == "namespace":
        state = _portable_cache_decode(value["state"], archive)
        return SimpleNamespace(**state)

    if kind in {"dataclass", "object"}:
        state = _portable_cache_decode(value["state"], archive)
        cls = _portable_cache_class(value["module"], value["qualname"])

        # Prefer reconstructing the original result class.  If that class has
        # moved or disappeared, retain all fields in a SimpleNamespace instead
        # of making the entire numerical cache unreadable.
        if cls is None:
            fallback = SimpleNamespace(**state)
            setattr(
                fallback,
                "_cache_original_type",
                f"{value['module']}.{value['qualname']}",
            )
            return fallback

        if kind == "dataclass":
            try:
                return cls(**state)
            except TypeError:
                # Allow compatible reading after dataclass signatures evolve.
                try:
                    current_names = {field.name for field in fields(cls)}
                    filtered = {name: item for name, item in state.items() if name in current_names}
                    return cls(**filtered)
                except Exception:
                    fallback = SimpleNamespace(**state)
                    setattr(
                        fallback,
                        "_cache_original_type",
                        f"{value['module']}.{value['qualname']}",
                    )
                    return fallback

        try:
            instance = cls.__new__(cls)
            for name, item in state.items():
                try:
                    setattr(instance, name, item)
                except Exception:
                    object.__setattr__(instance, name, item)
            return instance
        except Exception:
            fallback = SimpleNamespace(**state)
            setattr(
                fallback,
                "_cache_original_type",
                f"{value['module']}.{value['qualname']}",
            )
            return fallback

    raise ValueError(f"unknown portable-cache object kind {kind!r}")


def _save_portable_cache(path: Path, obj: Any) -> None:
    """Write a versioned ZIP/JSON/.npy cache atomically."""
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with zipfile.ZipFile(
            temporary,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=1,
            allowZip64=True,
        ) as archive:
            array_counter = [0]
            payload = _portable_cache_encode(obj, archive, array_counter)
            manifest = {
                "format": _PORTABLE_CACHE_FORMAT,
                "version": _PORTABLE_CACHE_VERSION,
                "numpy_version": np.__version__,
                "payload": payload,
            }
            archive.writestr(
                "manifest.json",
                json.dumps(manifest, separators=(",", ":"), ensure_ascii=False),
            )

        # Ensure the completed archive reaches disk before the atomic replace.
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_portable_cache(path: Path) -> Any:
    """Read a versioned ZIP/JSON/.npy cache."""
    with zipfile.ZipFile(path, mode="r") as archive:
        try:
            manifest = json.loads(archive.read("manifest.json").decode("utf-8"))
        except KeyError as exc:
            raise ValueError(f"portable cache {path} has no manifest.json") from exc

        if manifest.get("format") != _PORTABLE_CACHE_FORMAT:
            raise ValueError(f"unsupported cache format {manifest.get('format')!r} in {path}")
        version = int(manifest.get("version", -1))
        if version != _PORTABLE_CACHE_VERSION:
            raise ValueError(
                f"unsupported cache version {version} in {path}; "
                f"reader supports version {_PORTABLE_CACHE_VERSION}"
            )
        return _portable_cache_decode(manifest["payload"], archive)


def save_data(path: PathLike, obj: Any) -> None:
    """Save *obj* using the portable versioned cache format.

    New caches are ZIP archives containing JSON metadata and ``.npy`` numeric
    arrays.  The on-disk filename is left unchanged for backward compatibility
    with existing ``*.pkl`` cache naming, but the contents no longer depend on
    Python pickle or NumPy's private module names.  The write is atomic.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _save_portable_cache(destination, obj)


class _NumpyCompatibleUnpickler(pickle.Unpickler):
    """Read legacy NumPy-2.x pickle caches under NumPy 1.x when possible."""

    def find_class(self, module: str, name: str) -> Any:
        module = _LEGACY_CONDENSATE_MODULES.get(module, module)
        if module == "numpy._core" or module.startswith("numpy._core."):
            try:
                __import__("numpy._core")
            except ModuleNotFoundError:
                module = "numpy.core" + module[len("numpy._core") :]
        return super().find_class(module, name)


def load_data(path: PathLike) -> Any:
    """Load either a new portable cache or a legacy pickle cache.

    Portable caches are independent of NumPy's private ``numpy.core`` versus
    ``numpy._core`` module naming and are therefore suitable for moving between
    NumPy 1.x and 2.x machines.  Existing trusted pickle caches remain readable
    as a fallback, including the common NumPy-2.x -> NumPy-1.x module rename.
    """
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"data file not found: {source}")

    if zipfile.is_zipfile(source):
        return _load_portable_cache(source)

    # Legacy cache path. Pickle files must only be loaded from trusted sources.
    with source.open("rb") as handle:
        return _NumpyCompatibleUnpickler(handle).load()


def get_quenching_values(
    data: Any,
    q0: float,
    beta: float,
    a0: float,
    delta_t: Any,
    *,
    eps: Optional[float] = None,
    kinetic_scale: float = 1.0,
    distance_field: str = "distance_per_chain_nm",
    time_start: Any = None,
    time_stop: Any = None,
    cap_rate_at_q0: bool = False,
    cache_name: str = "quenching",
    overwrite: bool = False,
    base: Optional[PathLike] = None,
    tag: str = "",
    results_subdir: str = "results",
) -> Any:
    """Apply and cache quenching decay for minimum-contact results.

    ``data`` may be one minimum-contact result, the flat mapping returned by
    :func:`get_values`, or an arbitrarily nested mapping of compatible results.
    Mapping keys and nesting are preserved by :func:`mdsim.quenching.apply_decay`.

    Numeric ``delta_t``, ``time_start``, and ``time_stop`` values are
    interpreted as nanoseconds, matching the time conventions used by the
    other notebook helpers. Frames at ``t >= time_start`` and
    ``t < time_stop`` are retained. Selection occurs before ``kinetic_scale``
    is applied. OpenMM time quantities are also accepted.

    Cache files are written as::

        <base>/<results_subdir>/<cache_name>.quenching<tag>.pkl

    The input data and numerical parameters are intentionally not encoded in
    the filename. Use a distinct ``tag`` or ``overwrite=True`` whenever they
    change. ``cap_rate_at_q0=True`` caps ``q(r)`` at ``q0`` for distances
    below ``a0`` when ``eps`` is not used.
    """
    if not isinstance(cache_name, str) or not cache_name.strip():
        raise ValueError("cache_name must be a non-empty string")
    if not isinstance(tag, str):
        raise TypeError("tag must be a string")
    if not isinstance(results_subdir, str) or not results_subdir.strip():
        raise ValueError("results_subdir must be a non-empty string")

    delta_t_ns = _time_value_to_ns(delta_t, name="delta_t")
    if delta_t_ns <= 0.0:
        raise ValueError("delta_t must be > 0")
    time_start_ns = 0.0 if time_start is None else _time_value_to_ns(time_start, name="time_start")
    time_stop_ns = None if time_stop is None else _time_value_to_ns(time_stop, name="time_stop")
    if time_start_ns < 0.0:
        raise ValueError("time_start must be >= 0")
    if time_stop_ns is not None and time_stop_ns <= time_start_ns:
        raise ValueError("time_stop must be greater than time_start")

    root = Path(".") if base is None else Path(base)
    results_dir = root / results_subdir
    results_dir.mkdir(parents=True, exist_ok=True)

    cache_label = f"{cache_name}.quenching{tag}"
    cache_file = results_dir / f"{_filename_component(cache_label)}.pkl"

    if overwrite or not cache_file.exists():
        result = apply_decay(
            data,
            q0=q0,
            beta=beta,
            a0=a0,
            delta_t_ps=delta_t_ns * 1000.0,
            eps=eps,
            kinetic_scale=kinetic_scale,
            distance_field=distance_field,
            time_start_ps=time_start_ns * 1000.0,
            time_stop_ps=(None if time_stop_ns is None else time_stop_ns * 1000.0),
            cap_rate_at_q0=cap_rate_at_q0,
        )
        save_data(cache_file, result)
        return result

    return load_data(cache_file)


def _time_value_to_ps(value: Any, *, name: str) -> float:
    """Convert a numeric picosecond value or OpenMM time Quantity to ps."""
    if isinstance(value, (float, int, np.floating, np.integer)):
        converted = float(value)
    else:
        try:
            from openmm import unit

            converted = float(value.value_in_unit(unit.picoseconds))
        except Exception as exc:
            raise TypeError(f"{name} must be a number in ps or an OpenMM time Quantity") from exc
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


def _normalize_quenching_condition_mode(mode: str) -> str:
    """Normalize notebook-facing quenching conditioning mode names."""
    if not isinstance(mode, str):
        raise TypeError("condition_mode must be a string")
    value = mode.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "origin": "origin",
        "t0": "origin",
        "initial": "origin",
        "start": "origin",
        "continuous": "continuous",
        "path": "continuous",
        "all_times": "continuous",
        "until_violation": "continuous",
        "endpoint": "endpoint",
        "final": "endpoint",
        "current": "endpoint",
        "origin_endpoint": "origin_endpoint",
        "start_end": "origin_endpoint",
        "t0_endpoint": "origin_endpoint",
        "both_ends": "origin_endpoint",
        "fixed_horizon_segment": "fixed_horizon_segment",
        "fixed_horizon_segments": "fixed_horizon_segment",
        "fixed_horizon": "fixed_horizon_segment",
        "segment_fixed_horizon": "fixed_horizon_segment",
        "segments": "fixed_horizon_segment",
    }
    if value not in aliases:
        raise ValueError(
            "condition_mode must be 'origin', 'continuous', 'endpoint', "
            "'origin_endpoint', or 'fixed_horizon_segment'"
        )
    return aliases[value]


def _quenching_condition_array(
    decay_result: Any,
    filter_result: Any,
    *,
    filter_prop: str,
    filter_min: Optional[float],
    filter_max: Optional[float],
    path: str,
) -> np.ndarray:
    """Return a thresholded chain/frame mask aligned to one decay result."""
    hazards = np.asarray(_result_field(decay_result, "decay_exponent_per_chain"), dtype=float)
    if hazards.ndim != 2:
        raise ValueError(f"{path}: decay_exponent_per_chain must be two-dimensional")
    n_chains, n_frames = hazards.shape

    values = np.asarray(_result_field(filter_result, filter_prop), dtype=float)
    if values.ndim != 2:
        raise ValueError(
            f"{path}: filter property {filter_prop!r} must have shape "
            f"(n_chains, n_frames); got {values.shape}"
        )

    # Align physical chains by label rather than assuming the two calculations
    # happened to return rows in the same order.
    decay_labels = (
        tuple(str(x) for x in _result_field(decay_result, "chain_labels"))
        if _result_has_field(decay_result, "chain_labels")
        else None
    )
    filter_labels = (
        tuple(str(x) for x in _result_field(filter_result, "chain_labels"))
        if _result_has_field(filter_result, "chain_labels")
        else None
    )

    if decay_labels is not None:
        if len(decay_labels) != n_chains or len(set(decay_labels)) != len(decay_labels):
            raise ValueError(f"{path}: invalid or duplicate decay chain_labels")
    if filter_labels is not None:
        if len(filter_labels) != values.shape[0] or len(set(filter_labels)) != len(filter_labels):
            raise ValueError(f"{path}: invalid or duplicate filter chain_labels")

    if decay_labels is not None and filter_labels is not None:
        lookup = {label: index for index, label in enumerate(filter_labels)}
        missing = [label for label in decay_labels if label not in lookup]
        if missing:
            raise ValueError(f"{path}: filter result is missing decay chains {missing}")
        rows = np.asarray([lookup[label] for label in decay_labels], dtype=np.int64)
        values = values[rows, :]
    elif values.shape[0] != n_chains:
        raise ValueError(
            f"{path}: filter has {values.shape[0]} chain rows but decay has {n_chains}; "
            "chain_labels are required to align differing row sets"
        )

    # get_quenching_values may have selected a trajectory sub-window.  Its
    # source_frame_* metadata are indices on the original minimum-contact time
    # series, so use the same slice on the separately calculated contact series.
    source_start = int(getattr(decay_result, "source_frame_start", 0) or 0)
    source_stop_attr = getattr(decay_result, "source_frame_stop", None)
    source_stop = source_start + n_frames if source_stop_attr is None else int(source_stop_attr)

    if source_stop - source_start != n_frames:
        raise ValueError(
            f"{path}: decay source-frame metadata {source_start}:{source_stop} "
            f"does not match its {n_frames} frames"
        )

    if values.shape[1] >= source_stop:
        values = values[:, source_start:source_stop]
    elif values.shape[1] == n_frames:
        # Permit a filter trajectory that was already sliced identically.
        values = values.copy()
    else:
        raise ValueError(
            f"{path}: filter property has {values.shape[1]} frames and cannot be "
            f"aligned to decay source slice {source_start}:{source_stop}"
        )

    finite = np.isfinite(values)
    keep = finite.copy()
    if filter_min is not None:
        keep &= values >= float(filter_min)
    if filter_max is not None:
        keep &= values <= float(filter_max)
    return keep


def _fixed_horizon_segment_quenching_curve_result(
    decay_result: Any,
    condition: np.ndarray,
    *,
    tres_ps: float,
    fixed_horizon_ps: float,
    path: str,
) -> QuenchingCurveResult:
    """Return one fixed-horizon curve row per qualifying contiguous segment."""
    hazards = np.asarray(_result_field(decay_result, "decay_exponent_per_chain"), dtype=float)
    time_ps = np.asarray(_result_field(decay_result, "time_ps"), dtype=float)
    if hazards.ndim != 2:
        raise ValueError(f"{path}: decay_exponent_per_chain must be two-dimensional")
    n_source_chains, n_frames = hazards.shape
    if time_ps.shape != (n_frames,):
        raise ValueError(f"{path}: time_ps has shape {time_ps.shape}, expected ({n_frames},)")
    if condition.shape != hazards.shape:
        raise ValueError(
            f"{path}: condition mask has shape {condition.shape}, expected {hazards.shape}"
        )

    labels = (
        tuple(str(x) for x in _result_field(decay_result, "chain_labels"))
        if _result_has_field(decay_result, "chain_labels")
        else tuple(f"chain {i}" for i in range(n_source_chains))
    )
    if len(labels) != n_source_chains or len(set(labels)) != len(labels):
        raise ValueError(f"{path}: invalid or duplicate decay chain_labels")

    source_frame_start = int(getattr(decay_result, "source_frame_start", 0) or 0)
    lag_time: Optional[np.ndarray] = None
    survival_rows: list[np.ndarray] = []
    count_rows: list[np.ndarray] = []
    row_labels: list[str] = []
    source_labels: list[str] = []
    segment_starts: list[int] = []
    segment_stops: list[int] = []
    segment_origin_counts: list[int] = []

    for chain_index, source_label in enumerate(labels):
        segment_result = quench_fixed_horizon_segments(
            np.column_stack((time_ps, hazards[chain_index])),
            condition=condition[chain_index],
            fixed_horizon_ps=fixed_horizon_ps,
            tres=tres_ps,
        )

        if lag_time is None:
            lag_time = np.asarray(segment_result.lag_time_ps, dtype=float)
        elif not np.array_equal(lag_time, segment_result.lag_time_ps):
            raise RuntimeError(f"{path}: fixed-horizon segment curves do not share a lag grid")

        for segment_index in range(int(segment_result.n_segments)):
            local_start = int(segment_result.segment_start_frames[segment_index])
            local_stop = int(segment_result.segment_stop_frames[segment_index])
            global_start = source_frame_start + local_start
            global_stop = source_frame_start + local_stop

            survival_rows.append(
                np.asarray(segment_result.survival_per_segment[segment_index], dtype=float)
            )
            count_rows.append(
                np.asarray(segment_result.counts_per_segment[segment_index], dtype=float)
            )
            row_labels.append(f"{source_label}#segment{segment_index:04d}")
            source_labels.append(source_label)
            segment_starts.append(global_start)
            segment_stops.append(global_stop)
            segment_origin_counts.append(int(segment_result.segment_origin_counts[segment_index]))

    if lag_time is None:
        lag_time = np.zeros(0, dtype=float)

    if not survival_rows:
        raise ValueError(
            f"{path}: no contiguous condition-satisfying segment is long enough "
            f"for fixed_horizon={fixed_horizon_ps:g} ps"
        )

    survival = np.stack(survival_rows, axis=0)
    counts = np.stack(count_rows, axis=0)
    return QuenchingCurveResult(
        lag_time_ps=lag_time.copy(),
        survival_per_chain=survival,
        counts_per_chain=counts,
        chain_labels=tuple(row_labels),
        n_chains=int(len(row_labels)),
        n_frames=n_frames,
        n_lag_bins=int(lag_time.size),
        tres_ps=float(tres_ps),
        row_kind="segment",
        source_chain_labels=tuple(source_labels),
        segment_start_frames=tuple(segment_starts),
        segment_stop_frames=tuple(segment_stops),
        segment_origin_counts=tuple(segment_origin_counts),
        fixed_horizon_ps=float(fixed_horizon_ps),
        condition_mode="fixed_horizon_segment",
    )


def _conditioned_quenching_curve_result(
    decay_result: Any,
    condition: np.ndarray,
    *,
    tres_ps: float,
    condition_mode: str,
    fixed_horizon_ps: Optional[float],
    path: str,
) -> QuenchingCurveResult:
    mode = _normalize_quenching_condition_mode(condition_mode)
    if mode == "fixed_horizon_segment":
        if fixed_horizon_ps is None:
            raise ValueError("fixed_horizon is required for condition_mode='fixed_horizon_segment'")
        return _fixed_horizon_segment_quenching_curve_result(
            decay_result,
            condition,
            tres_ps=tres_ps,
            fixed_horizon_ps=float(fixed_horizon_ps),
            path=path,
        )

    hazards = np.asarray(_result_field(decay_result, "decay_exponent_per_chain"), dtype=float)
    time_ps = np.asarray(_result_field(decay_result, "time_ps"), dtype=float)
    if hazards.ndim != 2:
        raise ValueError(f"{path}: decay_exponent_per_chain must be two-dimensional")
    n_chains, n_frames = hazards.shape
    if time_ps.shape != (n_frames,):
        raise ValueError(f"{path}: time_ps has shape {time_ps.shape}, expected ({n_frames},)")
    if condition.shape != hazards.shape:
        raise ValueError(
            f"{path}: condition mask has shape {condition.shape}, expected {hazards.shape}"
        )

    labels = (
        tuple(str(x) for x in _result_field(decay_result, "chain_labels"))
        if _result_has_field(decay_result, "chain_labels")
        else tuple(f"chain {i}" for i in range(n_chains))
    )
    if len(labels) != n_chains:
        raise ValueError(f"{path}: chain_labels length does not match decay rows")

    curves = [
        quench(
            np.column_stack((time_ps, hazards[i])),
            tres=tres_ps,
            condition=condition[i],
            condition_mode=mode,
        )
        for i in range(n_chains)
    ]

    # Conditioning can legitimately leave different chains populated to different
    # maximum lags, or leave a chain with no observations at all.  Assemble the
    # union of lag bins and represent missing observations as NaN/count=0.
    populated_times = [curve[:, 0] for curve in curves if curve.shape[0] > 0]
    if not populated_times:
        return QuenchingCurveResult(
            lag_time_ps=np.zeros(0, dtype=float),
            survival_per_chain=np.zeros((n_chains, 0), dtype=float),
            counts_per_chain=np.zeros((n_chains, 0), dtype=float),
            chain_labels=labels,
            n_chains=n_chains,
            n_frames=n_frames,
            n_lag_bins=0,
            tres_ps=float(tres_ps),
            row_kind="chain",
            source_chain_labels=labels,
            condition_mode=mode,
        )

    lag_time = np.unique(np.concatenate(populated_times))
    survival = np.full((n_chains, lag_time.size), np.nan, dtype=float)
    counts = np.zeros((n_chains, lag_time.size), dtype=float)
    lookup = {float(t): index for index, t in enumerate(lag_time.tolist())}
    for chain_index, curve in enumerate(curves):
        for t, value, count in curve:
            out_index = lookup[float(t)]
            survival[chain_index, out_index] = float(value)
            counts[chain_index, out_index] = float(count)

    return QuenchingCurveResult(
        lag_time_ps=lag_time,
        survival_per_chain=survival,
        counts_per_chain=counts,
        chain_labels=labels,
        n_chains=n_chains,
        n_frames=n_frames,
        n_lag_bins=int(lag_time.size),
        tres_ps=float(tres_ps),
        row_kind="chain",
        source_chain_labels=labels,
        condition_mode=mode,
    )


def _conditioned_quenching_curves_recursive(
    data: Any,
    filter_data: Any,
    *,
    tres_ps: float,
    filter_prop: str,
    filter_min: Optional[float],
    filter_max: Optional[float],
    condition_mode: str,
    fixed_horizon_ps: Optional[float],
    path: str = "data",
) -> Any:
    if _result_has_field(data, "decay_exponent_per_chain") and _result_has_field(data, "time_ps"):
        mask = _quenching_condition_array(
            data,
            filter_data,
            filter_prop=filter_prop,
            filter_min=filter_min,
            filter_max=filter_max,
            path=path,
        )
        return _conditioned_quenching_curve_result(
            data,
            mask,
            tres_ps=tres_ps,
            condition_mode=condition_mode,
            fixed_horizon_ps=fixed_horizon_ps,
            path=path,
        )

    if isinstance(data, Mapping):
        if not data:
            raise ValueError(f"{path} is an empty mapping")
        if not isinstance(filter_data, Mapping):
            raise TypeError(f"{path}: filter_by must have matching mapping structure")
        missing = [key for key in data if key not in filter_data]
        if missing:
            raise KeyError(f"{path}: filter_by is missing keys {missing}")
        return {
            key: _conditioned_quenching_curves_recursive(
                child,
                filter_data[key],
                tres_ps=tres_ps,
                filter_prop=filter_prop,
                filter_min=filter_min,
                filter_max=filter_max,
                condition_mode=condition_mode,
                fixed_horizon_ps=fixed_horizon_ps,
                path=f"{path}[{key!r}]",
            )
            for key, child in data.items()
        }

    raise TypeError(f"{path} must be a quenching decay result or a mapping of such results")


def get_quenching_curves(
    data: Any,
    tres: Any = 100.0,
    *,
    filter_by: Optional[Mapping[str, Any]] = None,
    filter_min: Optional[float] = None,
    filter_max: Optional[float] = None,
    filter_prop: str = "contacts_per_chain",
    condition_mode: str = "origin",
    fixed_horizon: Any = None,
    cache_name: str = "quenching_curve",
    overwrite: bool = False,
    base: Optional[PathLike] = None,
    tag: str = "",
    results_subdir: str = "results",
) -> Any:
    """Calculate and cache time-origin-averaged quenching curves.

    With ``filter_by=None`` this is the original unconditioned calculation.
    Otherwise ``filter_by`` supplies a matching result mapping, normally the
    ``contacts`` output from :func:`get_values`, and ``filter_min`` and/or
    ``filter_max`` define the inclusive per-chain condition.  ``filter_prop``
    defaults to ``contacts_per_chain``.

    ``condition_mode`` controls the conditioning estimator:

    - ``"origin"``: require the condition only at the time origin.
    - ``"continuous"``: require it from origin through endpoint, inclusive.
    - ``"endpoint"``: require it only at the endpoint.
    - ``"origin_endpoint"``: require it at origin and endpoint only.
    - ``"fixed_horizon_segment"``: split every chain into maximal contiguous
      condition-satisfying segments and return one curve row per segment.  Within
      each retained segment, the same origins are used at every lag through
      ``fixed_horizon``.  Only segments long enough to support at least one such
      origin are retained.

    ``fixed_horizon`` is required only for ``fixed_horizon_segment``. Numeric
    values are interpreted as physical picoseconds, like ``tres``; OpenMM time
    quantities are accepted. The horizon should normally equal or exceed the
    upper end of the intended fit range.

    Fixed-horizon output has ``row_kind='segment'``. ``chain_labels`` then labels
    segment rows, while ``source_chain_labels``, ``segment_start_frames``,
    ``segment_stop_frames`` (exclusive), and ``segment_origin_counts`` retain the
    source metadata. To emulate an equal average over independent dilute-like
    trajectory stretches, use
    ``calculate_quenching_statistics(..., within_set_weighting='equal')``.
    Count weighting instead pools raw time-origin observations and gives longer
    segments greater weight.

    Chain rows and filter rows are aligned by ``chain_labels`` when available.
    If :func:`get_quenching_values` selected a time window, the contact trajectory
    is sliced using ``source_frame_start/source_frame_stop``.

    Cache filenames do not encode thresholds, mode, or horizon; use a distinct
    ``tag`` or ``overwrite=True`` whenever any of them changes.
    """
    if not isinstance(cache_name, str) or not cache_name.strip():
        raise ValueError("cache_name must be a non-empty string")
    if not isinstance(tag, str):
        raise TypeError("tag must be a string")
    if not isinstance(results_subdir, str) or not results_subdir.strip():
        raise ValueError("results_subdir must be a non-empty string")
    if not isinstance(filter_prop, str) or not filter_prop.strip():
        raise ValueError("filter_prop must be a non-empty string")

    tres_ps = _time_value_to_ps(tres, name="tres")
    if tres_ps <= 0.0:
        raise ValueError("tres must be > 0")

    condition_mode_name = _normalize_quenching_condition_mode(condition_mode)
    if condition_mode_name == "fixed_horizon_segment":
        if filter_by is None:
            raise ValueError("filter_by is required for condition_mode='fixed_horizon_segment'")
        if fixed_horizon is None:
            raise ValueError("fixed_horizon is required for condition_mode='fixed_horizon_segment'")
        fixed_horizon_ps: Optional[float] = _time_value_to_ps(fixed_horizon, name="fixed_horizon")
        if fixed_horizon_ps <= 0.0:
            raise ValueError("fixed_horizon must be > 0")
    else:
        if fixed_horizon is not None:
            raise ValueError(
                "fixed_horizon is only valid with " "condition_mode='fixed_horizon_segment'"
            )
        fixed_horizon_ps = None

    # Reuse the same inclusive threshold validation as the other analysis helpers.
    filter_min, filter_max = _validate_filter_options(
        filter_by,
        filter_min=filter_min,
        filter_max=filter_max,
        filter_prop=filter_prop if filter_by is not None else None,
    )

    root = Path(".") if base is None else Path(base)
    results_dir = root / results_subdir
    results_dir.mkdir(parents=True, exist_ok=True)

    cache_label = f"{cache_name}.quenchcurve{tag}"
    cache_file = results_dir / f"{_filename_component(cache_label)}.pkl"

    if overwrite or not cache_file.exists():
        if filter_by is None:
            result = quench_all(data, tres=tres_ps)
        else:
            result = _conditioned_quenching_curves_recursive(
                data,
                filter_by,
                tres_ps=tres_ps,
                filter_prop=filter_prop,
                filter_min=filter_min,
                filter_max=filter_max,
                condition_mode=condition_mode_name,
                fixed_horizon_ps=fixed_horizon_ps,
            )
        save_data(cache_file, result)
        return result

    return load_data(cache_file)


def get_quenching_fcs_values(
    data: Any,
    q0: float,
    beta: float,
    a0: float,
    delta_t: Any,
    *,
    intrinsic_lifetime: Any = 2.0,
    eps: Optional[float] = None,
    kinetic_scale: float = 1.0,
    distance_field: str = "distance_per_chain_nm",
    time_start: Any = None,
    time_stop: Any = None,
    cap_rate_at_q0: bool = True,
    cache_name: str = "quenching_fcs",
    overwrite: bool = False,
    base: Optional[PathLike] = None,
    tag: str = "",
    results_subdir: str = "results",
) -> Any:
    """Map minimum-contact trajectories to relative Atto brightness and cache them.

    Legacy name retained for compatibility.  Prefer
    :func:`get_brightness_values`.  This function
    evaluates the distance-dependent Trp rate and converts it to relative
    fluorescence brightness as

    ``B(r) = k_intrinsic / (k_intrinsic + q(r))``.

    Numeric ``delta_t``, ``intrinsic_lifetime``, ``time_start``, and
    ``time_stop`` values are interpreted as nanoseconds.  OpenMM time
    quantities are also accepted.  ``kinetic_scale`` scales the physical time
    axis but not the instantaneous brightness mapping.

    Cache files are written as::

        <base>/<results_subdir>/<cache_name>.fcssignal<tag>.pkl
    """
    if not isinstance(cache_name, str) or not cache_name.strip():
        raise ValueError("cache_name must be a non-empty string")
    if not isinstance(tag, str):
        raise TypeError("tag must be a string")
    if not isinstance(results_subdir, str) or not results_subdir.strip():
        raise ValueError("results_subdir must be a non-empty string")

    delta_t_ns = _time_value_to_ns(delta_t, name="delta_t")
    lifetime_ns = _time_value_to_ns(intrinsic_lifetime, name="intrinsic_lifetime")
    time_start_ns = 0.0 if time_start is None else _time_value_to_ns(time_start, name="time_start")
    time_stop_ns = None if time_stop is None else _time_value_to_ns(time_stop, name="time_stop")
    if delta_t_ns <= 0.0:
        raise ValueError("delta_t must be > 0")
    if lifetime_ns <= 0.0:
        raise ValueError("intrinsic_lifetime must be > 0")
    if time_start_ns < 0.0:
        raise ValueError("time_start must be >= 0")
    if time_stop_ns is not None and time_stop_ns <= time_start_ns:
        raise ValueError("time_stop must be greater than time_start")

    root = Path(".") if base is None else Path(base)
    results_dir = root / results_subdir
    results_dir.mkdir(parents=True, exist_ok=True)

    cache_label = f"{cache_name}.fcssignal{tag}"
    cache_file = results_dir / f"{_filename_component(cache_label)}.pkl"

    if overwrite or not cache_file.exists():
        result = apply_fluorescence_signal(
            data,
            q0=q0,
            beta=beta,
            a0=a0,
            intrinsic_lifetime_ps=lifetime_ns * 1000.0,
            delta_t_ps=delta_t_ns * 1000.0,
            eps=eps,
            kinetic_scale=kinetic_scale,
            distance_field=distance_field,
            time_start_ps=time_start_ns * 1000.0,
            time_stop_ps=None if time_stop_ns is None else time_stop_ns * 1000.0,
            cap_rate_at_q0=cap_rate_at_q0,
        )
        save_data(cache_file, result)
        return result

    return load_data(cache_file)


def get_quenching_fcs_curves(
    data: Any,
    tres: Any = None,
    *,
    signal_field: str = "brightness_per_chain",
    normalization: str = "fcs",
    max_lag: Any = None,
    max_lag_frames: Optional[int] = None,
    lag_stride: int = 1,
    include_zero_lag: bool = False,
    unbiased: bool = True,
    min_pairs: int = 2,
    cache_name: str = "quenching_fcs",
    overwrite: bool = False,
    base: Optional[PathLike] = None,
    tag: str = "",
    results_subdir: str = "results",
) -> Any:
    """Calculate and cache instantaneous-brightness autocorrelation curves.

    Legacy name retained for compatibility.  Prefer
    :func:`get_brightness_autocorrelation`.  ``data`` is normally returned by
    :func:`get_brightness_values`.  The
    default normalization is

    ``<delta B(t) delta B(t+lag)> / <B>**2``.

    Numeric ``tres`` is interpreted as picoseconds, matching
    :func:`get_quenching_curves`; it requests an output lag spacing rounded to
    the nearest trajectory-frame multiple. Numeric ``max_lag`` is interpreted
    as nanoseconds, matching other trajectory-facing notebook time limits.
    ``max_lag`` and ``max_lag_frames`` are mutually exclusive.

    Cache files are written as::

        <base>/<results_subdir>/<cache_name>.fcsacf<tag>.pkl
    """
    if not isinstance(cache_name, str) or not cache_name.strip():
        raise ValueError("cache_name must be a non-empty string")
    if not isinstance(tag, str):
        raise TypeError("tag must be a string")
    if not isinstance(results_subdir, str) or not results_subdir.strip():
        raise ValueError("results_subdir must be a non-empty string")

    lag_spacing_ps = None if tres is None else _time_value_to_ps(tres, name="tres")
    if lag_spacing_ps is not None and lag_spacing_ps <= 0.0:
        raise ValueError("tres must be > 0")
    max_lag_ps = None if max_lag is None else _time_value_to_ns(max_lag, name="max_lag") * 1000.0
    if max_lag_ps is not None and max_lag_ps <= 0.0:
        raise ValueError("max_lag must be > 0")

    root = Path(".") if base is None else Path(base)
    results_dir = root / results_subdir
    results_dir.mkdir(parents=True, exist_ok=True)

    cache_label = f"{cache_name}.fcsacf{tag}"
    cache_file = results_dir / f"{_filename_component(cache_label)}.pkl"

    if overwrite or not cache_file.exists():
        result = fluorescence_autocorrelation_all(
            data,
            signal_field=signal_field,
            normalization=normalization,
            max_lag_ps=max_lag_ps,
            max_lag_frames=max_lag_frames,
            lag_spacing_ps=lag_spacing_ps,
            lag_stride=lag_stride,
            include_zero_lag=include_zero_lag,
            unbiased=unbiased,
            min_pairs=min_pairs,
        )
        save_data(cache_file, result)
        return result

    return load_data(cache_file)


# --- preferred quenching terminology -------------------------------------------


def get_reactive_quenching_values(
    data: Any,
    q0: float,
    beta: float,
    a0: float,
    delta_t: Any,
    **kwargs: Any,
) -> Any:
    """Prepare finite-q(r) reactive-quenching hazards for the k_obs analysis.

    This is the preferred semantic name for :func:`get_quenching_values`.  It
    does not assume diffusion-limited quenching: the finite distance-dependent
    rate ``q(r)`` is retained.  Existing cache behavior and keyword arguments
    are unchanged.
    """
    return get_quenching_values(data, q0, beta, a0, delta_t, **kwargs)


def get_reactive_quenching_correlation(
    data: Any,
    tres: Any = 100.0,
    **kwargs: Any,
) -> Any:
    """Return the finite-q(r) reactive quenching correlation/survival curve.

    This is the preferred semantic name for :func:`get_quenching_curves`.  The
    resulting decay is the simulation quantity intended for comparison with the
    experimental Trp-quenching ``k_obs`` under the reaction--diffusion model.
    """
    return get_quenching_curves(data, tres=tres, **kwargs)


def get_brightness_values(
    data: Any,
    q0: float,
    beta: float,
    a0: float,
    delta_t: Any,
    **kwargs: Any,
) -> Any:
    """Map distance trajectories to instantaneous relative Atto brightness.

    This is the preferred semantic name for :func:`get_quenching_fcs_values`.
    The resulting brightness signal is used only for the separate equilibrium
    brightness-autocorrelation diagnostic.
    """
    return get_quenching_fcs_values(data, q0, beta, a0, delta_t, **kwargs)


def get_brightness_autocorrelation(
    data: Any,
    tres: Any = None,
    **kwargs: Any,
) -> Any:
    """Calculate the equilibrium autocorrelation of instantaneous brightness.

    This is the preferred semantic name for :func:`get_quenching_fcs_curves`.
    Its fitted rate characterizes brightness/contact persistence and should not
    generally be identified with the experimental Trp-quenching ``k_obs``.
    """
    return get_quenching_fcs_curves(data, tres=tres, **kwargs)


def _set_sort_key(value: str) -> tuple[int, str]:
    """Sort ``set1``, ``set2``, ..., ``set10`` numerically."""
    text = str(value).strip().lower()
    if text.startswith("set"):
        digits = ""
        for character in text[3:]:
            if not character.isdigit():
                break
            digits += character
        if digits:
            return int(digits), text
    return 10**9, text


def _dataset_sort_key(value: Any) -> tuple[int, str, str]:
    """Sort composite ``<set>_<group>`` keys by the leading set label."""
    text = str(value)
    set_label = text.split("_", 1)[0]
    set_number, normalized_set_label = _set_sort_key(set_label)
    return set_number, normalized_set_label, text


def _result_field(result: Any, field: str) -> Any:
    """Retrieve a field from either an object or a mapping result."""
    if isinstance(result, Mapping):
        if field not in result:
            raise KeyError(f"result does not contain field {field!r}")
        return result[field]

    if not hasattr(result, field):
        raise AttributeError(f"result has no attribute {field!r}")
    return getattr(result, field)


def _result_has_field(result: Any, field: str) -> bool:
    """Return whether an object or mapping result contains *field*."""
    if isinstance(result, Mapping):
        return field in result
    return hasattr(result, field)


def _validate_filter_options(
    filter_by: Optional[Mapping[str, Any]],
    *,
    filter_min: Optional[float],
    filter_max: Optional[float],
    filter_prop: Optional[str],
) -> tuple[Optional[float], Optional[float]]:
    """Validate shared secondary-data filtering options."""
    if filter_by is None:
        if filter_min is not None or filter_max is not None or filter_prop is not None:
            raise ValueError(
                "filter_by is required when filter_min, filter_max, or " "filter_prop is specified"
            )
        return None, None

    if not isinstance(filter_by, Mapping):
        raise TypeError("filter_by must be a mapping with keys matching data_by_set")
    if filter_min is None and filter_max is None:
        raise ValueError(
            "at least one of filter_min or filter_max is required when " "filter_by is supplied"
        )
    if filter_prop is not None and (not isinstance(filter_prop, str) or not filter_prop.strip()):
        raise ValueError("filter_prop must be a non-empty string or None")

    def convert_limit(value: Optional[float], name: str) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, (bool, np.bool_)):
            raise TypeError(f"{name} must be a finite number or None")
        try:
            converted = float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{name} must be a finite number or None") from exc
        if not math.isfinite(converted):
            raise ValueError(f"{name} must be finite")
        return converted

    minimum = convert_limit(filter_min, "filter_min")
    maximum = convert_limit(filter_max, "filter_max")
    if minimum is not None and maximum is not None and maximum < minimum:
        raise ValueError("filter_max must be greater than or equal to filter_min")
    return minimum, maximum


def _candidate_filter_fields(result: Any) -> list[str]:
    """Return public field names that may hold a filter array."""
    if isinstance(result, Mapping):
        return [str(name) for name in result]

    try:
        names = vars(result).keys()
    except TypeError:
        names = ()
    return [str(name) for name in names if not str(name).startswith("_")]


def _row_labels_for_filter_alignment(
    result: Any,
    *,
    expected_rows: int,
    key: str,
    role: str,
) -> Optional[tuple[str, ...]]:
    """Return validated ``chain_labels`` for row alignment, when available."""
    if not _result_has_field(result, "chain_labels"):
        return None

    labels = tuple(str(label) for label in _result_field(result, "chain_labels"))
    if len(labels) != int(expected_rows):
        raise ValueError(
            f"{key!r}: {role} chain_labels contains {len(labels)} labels, "
            f"but the corresponding array contains {expected_rows} rows"
        )
    if len(set(labels)) != len(labels):
        raise ValueError(f"{key!r}: {role} chain_labels contains duplicate labels")
    return labels


def _align_filter_array_to_primary(
    array: np.ndarray,
    *,
    filter_result: Any,
    primary_result: Any,
    expected_shape: tuple[int, int],
    key: str,
    field: str,
) -> np.ndarray:
    """Align filter rows to primary rows using ``chain_labels`` when possible.

    This permits, for example, an interchain-distance result with the reference
    chain excluded to be filtered by a contact result that still contains all
    chains. The filter rows are selected and reordered to match the target-chain
    labels in the primary result.
    """
    candidate = np.asarray(array, dtype=float)
    if candidate.ndim != 2:
        raise ValueError(
            f"{key!r}: filter property {field!r} must be two-dimensional; "
            f"got shape {candidate.shape}"
        )
    if candidate.shape[1] != expected_shape[1]:
        raise ValueError(
            f"{key!r}: filter property {field!r} has {candidate.shape[1]} frames, "
            f"expected {expected_shape[1]}"
        )

    primary_labels = _row_labels_for_filter_alignment(
        primary_result,
        expected_rows=expected_shape[0],
        key=key,
        role="primary",
    )
    filter_labels = _row_labels_for_filter_alignment(
        filter_result,
        expected_rows=candidate.shape[0],
        key=key,
        role="filter",
    )

    if primary_labels is not None and filter_labels is not None:
        filter_index = {label: index for index, label in enumerate(filter_labels)}
        missing = [label for label in primary_labels if label not in filter_index]
        if missing:
            raise ValueError(
                f"{key!r}: filter property {field!r} is missing chain labels "
                f"required by the primary data: {missing}"
            )
        row_indices = np.asarray(
            [filter_index[label] for label in primary_labels],
            dtype=np.int64,
        )
        aligned = candidate[row_indices, :]
        if aligned.shape != expected_shape:
            raise RuntimeError(
                f"{key!r}: internal filter-alignment error produced shape "
                f"{aligned.shape}, expected {expected_shape}"
            )
        return aligned

    if candidate.shape != expected_shape:
        raise ValueError(
            f"{key!r}: filter property {field!r} has shape {candidate.shape}, "
            f"expected {expected_shape}. Row counts differ and both results do "
            "not provide compatible chain_labels for alignment."
        )
    return candidate


def _filter_array_for_key(
    filter_by: Mapping[str, Any],
    key: str,
    *,
    expected_shape: tuple[int, int],
    filter_prop: Optional[str],
    primary_result: Any,
) -> np.ndarray:
    """Retrieve, align, and validate the secondary array for *key*.

    Filter rows are aligned to primary rows by ``chain_labels`` when both
    results provide them. This allows a filter result containing all chains to
    be used with an interchain-distance result that excludes its reference
    chain.
    """
    if key not in filter_by:
        raise KeyError(f"filter_by does not contain a matching entry for {key!r}")

    result = filter_by[key]
    if filter_prop is not None:
        try:
            array = np.asarray(_result_field(result, filter_prop), dtype=float)
        except (AttributeError, KeyError) as exc:
            raise type(exc)(f"{key!r}: filter data {exc}") from exc
        return _align_filter_array_to_primary(
            array,
            filter_result=result,
            primary_result=primary_result,
            expected_shape=expected_shape,
            key=key,
            field=filter_prop,
        )

    preferred_fields = (
        "contacts_per_chain",
        "rg_per_chain_nm",
        "distance_per_chain_nm",
        "n_chains_above_threshold",
    )
    matching: list[tuple[str, np.ndarray]] = []
    seen: set[str] = set()
    fields = list(preferred_fields) + _candidate_filter_fields(result)
    for field in fields:
        if field in seen or not _result_has_field(result, field):
            continue
        seen.add(field)
        try:
            candidate = np.asarray(_result_field(result, field), dtype=float)
            aligned = _align_filter_array_to_primary(
                candidate,
                filter_result=result,
                primary_result=primary_result,
                expected_shape=expected_shape,
                key=key,
                field=field,
            )
        except (TypeError, ValueError):
            continue
        matching.append((field, aligned))

    if not matching:
        available = ", ".join(repr(field) for field in _candidate_filter_fields(result))
        suffix = f"; available fields: {available}" if available else ""
        raise ValueError(
            f"{key!r}: could not infer a two-dimensional filter property that "
            f"can be aligned to shape {expected_shape}{suffix}. Pass "
            "filter_prop explicitly."
        )

    preferred_matches = [item for item in matching if item[0] in preferred_fields]
    if len(preferred_matches) == 1:
        return preferred_matches[0][1]
    if len(matching) == 1:
        return matching[0][1]

    names = ", ".join(repr(name) for name, _ in matching)
    raise ValueError(
        f"{key!r}: multiple filter properties can be aligned to shape "
        f"{expected_shape}: {names}. Pass filter_prop explicitly."
    )


def _apply_secondary_filter(
    array: np.ndarray,
    filter_array: np.ndarray,
    *,
    filter_min: Optional[float],
    filter_max: Optional[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Mask *array* using inclusive limits on *filter_array*.

    Returns the filtered primary array and an eligibility mask indicating where
    both arrays had finite values before thresholding.
    """
    eligible = np.isfinite(array) & np.isfinite(filter_array)
    keep = eligible.copy()
    if filter_min is not None:
        keep &= filter_array >= filter_min
    if filter_max is not None:
        keep &= filter_array <= filter_max
    return np.where(keep, array, np.nan), eligible


def _normalize_interchain_filter_mode(mode: str) -> str:
    """Normalize how chain-level filters are applied to interchain RDFs."""
    if not isinstance(mode, str):
        raise TypeError("interchain_filter_mode must be a string")

    normalized = mode.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "target": "target",
        "targets": "target",
        "target_only": "target",
        "other": "target",
        "others": "target",
        "reference": "reference",
        "ref": "reference",
        "reference_only": "reference",
        "ref_only": "reference",
        "both": "both",
        "reference_and_target": "both",
        "reference_and_targets": "both",
        "ref_and_target": "both",
        "ref_and_targets": "both",
    }
    if normalized not in aliases:
        raise ValueError("interchain_filter_mode must be 'target', 'reference', or 'both'")
    return aliases[normalized]


def _normalize_reference_weighting(weighting: str) -> str:
    """Normalize weighting of reference-specific interchain RDF curves."""
    if not isinstance(weighting, str):
        raise TypeError("reference_weighting must be a string")

    normalized = weighting.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "auto": "auto",
        "equal": "equal",
        "uniform": "equal",
        "references": "equal",
        "reference": "equal",
        "observations": "observations",
        "observation": "observations",
        "frames": "observations",
        "qualifying_frames": "observations",
        "reference_observations": "observations",
    }
    if normalized not in aliases:
        raise ValueError("reference_weighting must be 'auto', 'equal', or 'observations'")
    return aliases[normalized]


def _filter_threshold_mask(
    values: np.ndarray,
    *,
    filter_min: Optional[float],
    filter_max: Optional[float],
) -> np.ndarray:
    """Return the finite-value mask satisfying inclusive filter limits."""
    array = np.asarray(values, dtype=float)
    keep = np.isfinite(array)
    if filter_min is not None:
        keep &= array >= filter_min
    if filter_max is not None:
        keep &= array <= filter_max
    return keep


def _reference_filter_series_for_key(
    filter_by: Mapping[str, Any],
    key: str,
    *,
    filter_prop: Optional[str],
    primary_result: Any,
    expected_frames: int,
) -> np.ndarray:
    """Return the filter time series for an interchain reference chain."""
    if key not in filter_by:
        raise KeyError(f"filter_by does not contain a matching entry for {key!r}")
    if not _result_has_field(primary_result, "reference_chain_label"):
        raise ValueError(
            f"{key!r}: reference-chain filtering requires an interchain result "
            "with reference_chain_label"
        )

    reference_label = str(_result_field(primary_result, "reference_chain_label"))
    filter_result = filter_by[key]
    if filter_prop is not None:
        fields = [filter_prop]
    else:
        fields = [
            "contacts_per_chain",
            "rg_per_chain_nm",
            "distance_per_chain_nm",
            "n_chains_above_threshold",
            *_candidate_filter_fields(filter_result),
        ]

    matches: list[tuple[str, np.ndarray]] = []
    seen: set[str] = set()
    for field in fields:
        if field in seen or not _result_has_field(filter_result, field):
            continue
        seen.add(field)
        try:
            candidate = np.asarray(_result_field(filter_result, field), dtype=float)
        except (TypeError, ValueError):
            continue
        if candidate.ndim != 2 or candidate.shape[1] != int(expected_frames):
            continue
        labels = _row_labels_for_filter_alignment(
            filter_result,
            expected_rows=candidate.shape[0],
            key=key,
            role="filter",
        )
        if labels is None:
            continue
        label_to_index = {label: index for index, label in enumerate(labels)}
        if reference_label in label_to_index:
            matches.append((field, candidate[label_to_index[reference_label], :]))

    if not matches:
        raise ValueError(
            f"{key!r}: could not find reference chain {reference_label!r} in "
            "the filter result. Regenerate cached filter data if chain_labels "
            "are missing, and pass filter_prop explicitly if needed."
        )
    if filter_prop is not None or len(matches) == 1:
        return matches[0][1]

    preferred = [
        item
        for item in matches
        if item[0]
        in {
            "contacts_per_chain",
            "rg_per_chain_nm",
            "distance_per_chain_nm",
            "n_chains_above_threshold",
        }
    ]
    if len(preferred) == 1:
        return preferred[0][1]
    names = ", ".join(repr(name) for name, _ in matches)
    raise ValueError(
        f"{key!r}: multiple filter properties contain reference chain "
        f"{reference_label!r}: {names}. Pass filter_prop explicitly."
    )


@dataclass
class HistogramResult:
    """A one-dimensional histogram and its shared bin edges.

    When histograms are averaged over simulation sets, ``values`` is the
    equal-weight mean of the independently normalized set histograms.
    ``n_samples`` is the total number of included observations across those
    sets, while ``n_sets`` and ``set_labels`` describe the replicate sets.
    """

    values: np.ndarray
    edges: np.ndarray
    density: bool
    n_samples: int
    n_sets: int = 1
    set_labels: tuple[str, ...] = ()

    @property
    def centers(self) -> np.ndarray:
        return 0.5 * (self.edges[:-1] + self.edges[1:])

    @property
    def widths(self) -> np.ndarray:
        return np.diff(self.edges)


@dataclass
class RadialNumberDensityResult:
    """Spherically normalized radial number density.

    ``values`` contains the number density in each spherical shell, in
    ``nm^-3``. For equal-weight set averaging, ``values`` is the mean of the
    independently normalized set curves and ``stderr`` is the standard error
    across those curves.

    The shell integral ``sum(values * shell_volumes)`` is the mean number of
    counted target atoms per reference observation within the represented
    radial range. It approaches one for a complete intrachain pair-distance
    distribution and the number of target chains for a complete interchain
    distribution.
    """

    values: np.ndarray
    stderr: np.ndarray
    edges: np.ndarray
    n_distances: int
    n_in_range: int
    n_reference_observations: int
    n_frames: Optional[int]
    reference_mode: str
    n_sets: int = 1
    set_labels: tuple[str, ...] = ()
    per_set_values: Optional[np.ndarray] = None
    n_distances_per_set: tuple[int, ...] = ()
    n_reference_observations_per_set: tuple[int, ...] = ()
    n_frames_per_set: tuple[int, ...] = ()
    n_reference_segments: int = 0
    reference_segment_labels: tuple[str, ...] = ()
    n_reference_segments_per_set: tuple[int, ...] = ()
    reference_observations_per_segment: tuple[int, ...] = ()
    aggregation: str = "individual"
    units: str = "nm^-3"

    @property
    def centers(self) -> np.ndarray:
        return 0.5 * (self.edges[:-1] + self.edges[1:])

    @property
    def widths(self) -> np.ndarray:
        return np.diff(self.edges)

    @property
    def shell_volumes(self) -> np.ndarray:
        return (4.0 * math.pi / 3.0) * (np.power(self.edges[1:], 3) - np.power(self.edges[:-1], 3))

    @property
    def cumulative_number(self) -> np.ndarray:
        """Cumulative mean target count per reference through each shell."""
        return np.cumsum(self.values * self.shell_volumes)

    @property
    def integrated_number(self) -> float:
        """Mean target count per reference within the represented range."""
        return float(np.sum(self.values * self.shell_volumes))

    @property
    def density(self) -> bool:
        """Compatibility flag for :func:`plot_histograms`."""
        return True


@dataclass
class AverageResult:
    """Summary of a two-dimensional trajectory property.

    In the default chain-aggregation mode, each chain is first averaged over
    the selected frames and ``mean``, ``std``, and ``stderr`` are calculated
    across those per-chain averages.

    With ``calculate_averages(..., average_over_sets=True)``, each set is first
    reduced to its own chain-averaged mean.  The reported ``mean``, ``std``, and
    ``stderr`` are then calculated across the set means, giving every set equal
    statistical weight.  ``aggregation`` records which convention was used.
    """

    mean: float
    std: float
    stderr: float
    per_chain_mean: np.ndarray
    n_chains: int
    n_total_chains: int
    n_frames: Optional[int]
    frame_start: Optional[int]
    frame_stop: Optional[int]
    per_set_mean: Optional[np.ndarray] = None
    n_sets: int = 1
    set_labels: tuple[str, ...] = ()
    aggregation: str = "chains"
    n_frames_per_set: tuple[int, ...] = ()

    @property
    def sem(self) -> float:
        """Alias for :attr:`stderr`."""
        return self.stderr


def _ordered_set_groups(
    keys: Sequence[str],
    *,
    group_by: Optional[Mapping[str, str]] = None,
) -> dict[str, list[str]]:
    """Group input dataset keys for equal-weight averaging over sets.

    ``group_by`` may explicitly map every input key to its desired output group.
    Without it, keys containing an underscore are interpreted as
    ``<set>_<group>`` and grouped by the suffix.  If no key contains an
    underscore, all keys are treated as replicate sets of one group named
    ``"all_sets"``.
    """
    ordered_keys = list(keys)
    groups: dict[str, list[str]] = {}

    if group_by is not None:
        missing = [key for key in ordered_keys if key not in group_by]
        if missing:
            formatted = ", ".join(repr(key) for key in missing)
            raise KeyError(f"group_by does not contain entries for: {formatted}")

        for key in ordered_keys:
            group = str(group_by[key]).strip()
            if not group:
                raise ValueError(f"group_by[{key!r}] must be non-empty")
            groups.setdefault(group, []).append(key)
        return groups

    if ordered_keys and all("_" in str(key) for key in ordered_keys):
        for key in ordered_keys:
            group = str(key).split("_", 1)[1]
            if not group:
                raise ValueError(
                    f"cannot infer a group name from dataset key {key!r}; "
                    "supply group_by explicitly"
                )
            groups.setdefault(group, []).append(key)
    else:
        groups["all_sets"] = ordered_keys

    return groups


def _common_or_none(values: Sequence[int]) -> Optional[int]:
    """Return the shared integer value, or ``None`` when values differ."""
    if not values:
        return None
    first = int(values[0])
    return first if all(int(value) == first for value in values[1:]) else None


def _time_value_to_ns(value: Any, *, name: str) -> float:
    """Convert a numeric nanosecond time or OpenMM Quantity to ns."""
    if isinstance(value, (float, int, np.floating, np.integer)):
        time_ns = float(value)
    else:
        try:
            from openmm import unit

            time_ns = float(value.value_in_unit(unit.nanoseconds))
        except Exception as exc:
            raise TypeError(f"{name} must be a number in ns or an OpenMM Quantity") from exc

    if not math.isfinite(time_ns):
        raise ValueError(f"{name} must be finite")
    return time_ns


def _frame_index(value: Any, *, name: str) -> int:
    """Validate and convert one frame index without silently truncating."""
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer or None")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be an integer or None") from exc
    if not math.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(f"{name} must be an integer or None")
    return int(numeric)


def _resolve_frame_slice(
    n_frames: int,
    *,
    frame_start: Optional[int] = None,
    frame_stop: Optional[int] = None,
    time_start: Any = None,
    time_stop: Any = None,
    delta_t: Any = None,
) -> tuple[int, int]:
    """Resolve index- or time-based limits to an inclusive/exclusive slice.

    Frame limits follow ordinary Python slicing: ``frame_start`` is included and
    ``frame_stop`` is excluded.  Time limits use the same convention: frames at
    times ``t >= time_start`` and ``t < time_stop`` are retained.  Numeric time
    values and numeric ``delta_t`` values are interpreted as nanoseconds; OpenMM
    time quantities are also accepted.
    """
    n_frames_i = int(n_frames)
    if n_frames_i <= 0:
        raise ValueError("the selected property contains no frames")

    uses_frames = frame_start is not None or frame_stop is not None
    uses_time = time_start is not None or time_stop is not None
    if uses_frames and uses_time:
        raise ValueError("specify either frame_start/frame_stop or time_start/time_stop, not both")

    if uses_time:
        if delta_t is None:
            raise ValueError("delta_t is required when time_start or time_stop is specified")
        dt_ns = _delta_t_to_ns(delta_t)
        start_time_ns = (
            0.0 if time_start is None else _time_value_to_ns(time_start, name="time_start")
        )
        if start_time_ns < 0.0:
            raise ValueError("time_start must be >= 0")

        if time_stop is None:
            stop_time_ns = None
        else:
            stop_time_ns = _time_value_to_ns(time_stop, name="time_stop")
            if stop_time_ns < 0.0:
                raise ValueError("time_stop must be >= 0")
            if stop_time_ns <= start_time_ns:
                raise ValueError("time_stop must be greater than time_start")

        frame_times_ns = dt_ns * np.arange(n_frames_i, dtype=np.float64)
        start = int(np.searchsorted(frame_times_ns, start_time_ns, side="left"))
        stop = (
            n_frames_i
            if stop_time_ns is None
            else int(np.searchsorted(frame_times_ns, stop_time_ns, side="left"))
        )
    else:
        start = 0 if frame_start is None else _frame_index(frame_start, name="frame_start")
        stop = n_frames_i if frame_stop is None else _frame_index(frame_stop, name="frame_stop")

        # Preserve normal Python negative-index behavior, while clipping limits
        # that extend beyond the available frame range.
        if start < 0:
            start += n_frames_i
        if stop < 0:
            stop += n_frames_i
        start = max(0, min(start, n_frames_i))
        stop = max(0, min(stop, n_frames_i))

    if stop <= start:
        raise ValueError(
            "the requested frame/time range selects no frames "
            f"(resolved slice {start}:{stop} for {n_frames_i} frames)"
        )
    return start, stop


def calculate_histograms(
    data_by_set: Mapping[str, Any],
    prop: str = "rg_per_chain_nm",
    *,
    bins: Union[int, Sequence[float], np.ndarray] = 60,
    value_range: Optional[tuple[float, float]] = (0.0, 8.0),
    density: bool = True,
    per_chain_time_avg: bool = False,
    average_over_sets: bool = False,
    group_by: Optional[Mapping[str, str]] = None,
    filter_by: Optional[Mapping[str, Any]] = None,
    filter_min: Optional[float] = None,
    filter_max: Optional[float] = None,
    filter_prop: Optional[str] = None,
    frame_start: Optional[int] = None,
    frame_stop: Optional[int] = None,
    time_start: Any = None,
    time_stop: Any = None,
    delta_t: Any = None,
) -> dict[str, HistogramResult]:
    """Calculate shared-bin histograms for a 2D result property.

    The selected property must have shape ``(n_chains, n_frames)``.  With
    ``per_chain_time_avg=False``, all finite chain-by-frame observations are
    pooled within each set.  With ``per_chain_time_avg=True``, each chain
    contributes one finite time-average.

    By default, one histogram is returned for every input key.  With
    ``average_over_sets=True``, a histogram is first calculated and normalized
    independently for each set, and those histograms are then averaged with
    equal weight.  This prevents sets with more chains or frames from dominating
    the result.  Input keys of the form ``<set>_<group>`` are grouped by the
    suffix; use ``group_by`` to supply an explicit ``input_key -> output_group``
    mapping.  If the keys contain no underscores, all entries are treated as
    replicate sets and returned under ``"all_sets"``.

    Limit the analyzed frames either with ``frame_start``/``frame_stop`` or with
    ``time_start``/``time_stop`` plus ``delta_t``.  Stop limits are exclusive.
    Numeric time values are interpreted as nanoseconds; OpenMM time quantities
    are accepted as well.

    ``filter_by`` may supply a second result mapping keyed like ``data_by_set``.
    Primary observations are retained only where the corresponding filter value
    is finite and satisfies the inclusive ``filter_min`` and/or ``filter_max``
    limits. The filter array must have exactly the same shape as the primary
    property. ``filter_prop`` may name its field explicitly; when omitted, a
    unique matching two-dimensional numeric field is inferred.
    """
    if not data_by_set:
        raise ValueError("data_by_set is empty")
    if not isinstance(prop, str) or not prop:
        raise ValueError("prop must be a non-empty string")

    filter_min, filter_max = _validate_filter_options(
        filter_by,
        filter_min=filter_min,
        filter_max=filter_max,
        filter_prop=filter_prop,
    )

    keys = sorted(data_by_set, key=_dataset_sort_key)
    values_by_set: dict[str, np.ndarray] = {}

    for key in keys:
        array = np.asarray(_result_field(data_by_set[key], prop), dtype=float)
        if array.ndim != 2:
            raise ValueError(
                f"{key!r}: property {prop!r} must have shape "
                f"(n_chains, n_frames); got {array.shape}"
            )
        full_shape = array.shape

        start, stop = _resolve_frame_slice(
            array.shape[1],
            frame_start=frame_start,
            frame_stop=frame_stop,
            time_start=time_start,
            time_stop=time_stop,
            delta_t=delta_t,
        )
        array = array[:, start:stop]

        if filter_by is not None:
            filter_array = _filter_array_for_key(
                filter_by,
                key,
                expected_shape=full_shape,
                filter_prop=filter_prop,
                primary_result=data_by_set[key],
            )
            filter_array = filter_array[:, start:stop]
            array, _ = _apply_secondary_filter(
                array,
                filter_array,
                filter_min=filter_min,
                filter_max=filter_max,
            )

        if per_chain_time_avg:
            finite = np.isfinite(array)
            counts = np.sum(finite, axis=1)
            sums = np.sum(np.where(finite, array, 0.0), axis=1)
            values = np.full(array.shape[0], np.nan, dtype=float)
            valid = counts > 0
            values[valid] = sums[valid] / counts[valid]
        else:
            values = array.reshape(-1)

        values_by_set[key] = values[np.isfinite(values)]

    if np.isscalar(bins):
        if isinstance(bins, (bool, np.bool_)):
            raise TypeError("bins must be an integer or explicit bin edges")
        n_bins_float = float(bins)
        if not n_bins_float.is_integer():
            raise ValueError("bins must be an integer")
        n_bins = int(n_bins_float)
        if n_bins < 1:
            raise ValueError("bins must be at least 1")

        if value_range is None:
            nonempty = [values for values in values_by_set.values() if values.size]
            if not nonempty:
                raise ValueError(f"no finite values found for property {prop!r}")

            minimum = min(float(np.min(values)) for values in nonempty)
            maximum = max(float(np.max(values)) for values in nonempty)
            if maximum <= minimum:
                padding = 0.01 * max(abs(minimum), 1.0)
            else:
                padding = 0.02 * (maximum - minimum)
            value_range = (minimum - padding, maximum + padding)

        if len(value_range) != 2:
            raise ValueError("value_range must contain exactly two values")
        range_min, range_max = (float(value_range[0]), float(value_range[1]))
        if not math.isfinite(range_min) or not math.isfinite(range_max):
            raise ValueError("value_range values must be finite")
        if range_max <= range_min:
            raise ValueError("value_range must be an increasing (min, max) pair")
        edges = np.linspace(range_min, range_max, n_bins + 1)
    else:
        edges = np.asarray(bins, dtype=float)
        if edges.ndim != 1 or edges.size < 2:
            raise ValueError(
                "explicit bin edges must be a one-dimensional array with at " "least two values"
            )
        if not np.all(np.isfinite(edges)):
            raise ValueError("bin edges must all be finite")
        if not np.all(np.diff(edges) > 0.0):
            raise ValueError("bin edges must be strictly increasing")

    widths = np.diff(edges)

    def histogram_for_set(key: str) -> tuple[np.ndarray, int]:
        values = values_by_set[key]
        if not values.size:
            raise ValueError(f"{key!r}: no finite values were found in the selected frames")

        counts, _ = np.histogram(values, bins=edges, density=False)
        included = int(np.sum(counts))
        if included == 0:
            raise ValueError(
                f"{key!r}: no finite values fall inside the histogram range "
                f"({edges[0]}, {edges[-1]})"
            )

        if density:
            histogram_values = counts.astype(float) / (included * widths)
        else:
            histogram_values = counts
        return histogram_values, included

    histograms: dict[str, HistogramResult] = {}

    if not average_over_sets:
        for key in keys:
            values = values_by_set[key]
            if not values.size:
                continue
            histogram_values, included = histogram_for_set(key)
            histograms[key] = HistogramResult(
                values=histogram_values,
                edges=edges.copy(),
                density=bool(density),
                n_samples=included,
                n_sets=1,
                set_labels=(str(key),),
            )
    else:
        groups = _ordered_set_groups(keys, group_by=group_by)
        for group, member_keys in groups.items():
            set_histograms: list[np.ndarray] = []
            sample_counts: list[int] = []
            for key in member_keys:
                histogram_values, included = histogram_for_set(key)
                set_histograms.append(histogram_values)
                sample_counts.append(included)

            histograms[group] = HistogramResult(
                values=np.mean(np.stack(set_histograms, axis=0), axis=0),
                edges=edges.copy(),
                density=bool(density),
                n_samples=int(np.sum(sample_counts)),
                n_sets=len(member_keys),
                set_labels=tuple(str(key) for key in member_keys),
            )

    if not histograms:
        raise ValueError(f"no histograms could be calculated for property {prop!r}")
    return histograms


def _normalize_radial_reference_mode(reference_mode: str) -> str:
    """Normalize radial-density reference-counting mode names."""
    if not isinstance(reference_mode, str):
        raise TypeError("reference_mode must be a string")

    mode = reference_mode.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "auto": "auto",
        "per_row": "per_row",
        "row": "per_row",
        "rows": "per_row",
        "intrachain": "per_row",
        "pair": "per_row",
        "per_frame": "per_frame",
        "frame": "per_frame",
        "single": "per_frame",
        "single_reference": "per_frame",
        "interchain": "per_frame",
        "reference_center": "per_frame",
    }
    if mode not in aliases:
        raise ValueError("reference_mode must be 'auto', 'per_row', or 'per_frame'")
    return aliases[mode]


def _infer_radial_reference_mode(result: Any) -> str:
    """Infer whether rows represent references or targets around one reference."""
    # InterchainDistanceResult and ReferenceCenterDistanceResult both have one
    # reference object per frame and multiple target/query rows.
    if _result_has_field(result, "reference_chain_label") or _result_has_field(
        result, "reference_label"
    ):
        return "per_frame"

    # IntrachainDistanceResult contains one independent reference-target pair
    # per row. The atom_indices field is specific to that result class.
    if _result_has_field(result, "atom_indices"):
        return "per_row"

    class_name = type(result).__name__.lower()
    if "interchain" in class_name or "referencecenter" in class_name:
        return "per_frame"
    if "intrachain" in class_name:
        return "per_row"

    raise ValueError(
        "could not infer reference counting from the result type; pass "
        "reference_mode='per_row' for intrachain pair distances or "
        "reference_mode='per_frame' for distances from one reference to "
        "multiple targets"
    )


def _calculate_radial_number_density_flat(
    data_by_set: Mapping[str, Any],
    prop: str = "distance_per_chain_nm",
    *,
    bins: Union[int, Sequence[float], np.ndarray] = 60,
    value_range: Optional[tuple[float, float]] = (0.0, 8.0),
    reference_mode: str = "auto",
    average_over_sets: bool = False,
    group_by: Optional[Mapping[str, str]] = None,
    filter_by: Optional[Mapping[str, Any]] = None,
    filter_min: Optional[float] = None,
    filter_max: Optional[float] = None,
    filter_prop: Optional[str] = None,
    interchain_filter_mode: str = "target",
    frame_start: Optional[int] = None,
    frame_stop: Optional[int] = None,
    time_start: Any = None,
    time_stop: Any = None,
    delta_t: Any = None,
) -> dict[str, RadialNumberDensityResult]:
    """Flat-mapping implementation for radial number density in ``nm^-3``.

    The selected property must have shape ``(n_rows, n_frames)`` and normally
    should be ``distance_per_chain_nm`` from either
    ``intrachain_distances_from_dcd`` or ``interchain_distances_from_dcd``.
    Counts in shell ``i`` are normalized as::

        rho_i = counts_i / (N_reference_observations * shell_volume_i)

    where ``shell_volume_i = 4*pi/3 * (r_outer^3 - r_inner^3)``.

    Reference counting
    ------------------
    ``reference_mode='auto'`` distinguishes the standard distance result
    classes automatically:

    - intrachain result: ``'per_row'``. Every row and frame is one independent
      reference-target observation. Integrating a complete distribution over
      volume gives one target per reference.
    - interchain result: ``'per_frame'``. Each frame has one reference atom and
      every row is a target chain. Integrating a complete distribution gives
      the mean number of target chains per reference.

    Use ``reference_mode`` explicitly when passing a plain mapping/object that
    does not retain the original result metadata.

    Set averaging
    -------------
    By default, one curve is returned per input key. With
    ``average_over_sets=True``, each set is normalized independently and the
    resulting number-density curves are averaged with equal set weight. Input
    keys of the form ``<set>_<group>`` are grouped by suffix, matching
    :func:`calculate_histograms`; ``group_by`` can override that inference.

    Frame/time selection follows :func:`calculate_histograms`. Distances
    outside ``value_range`` remain part of the reference normalization but do
    not contribute to represented shells; the curve is therefore not
    artificially renormalized when the radial range is truncated.

    Secondary filtering follows :func:`calculate_histograms`. For interchain
    results, ``interchain_filter_mode`` applies limits to each target chain
    (``"target"``), to the reference chain (``"reference"``), or to both.
    Reference-conditioned modes normalize by the number of frames in which the
    reference chain satisfies the limits. Target-only mode retains all eligible
    reference frames, so frames with no passing targets contribute zero density.

    Notes
    -----
    This is a three-dimensional spherical radial number density, not a
    dimensionless bulk-normalized ``g(r)``. It assumes the supplied distances
    are ordinary 3D radial distances.
    """
    if not data_by_set:
        raise ValueError("data_by_set is empty")
    if not isinstance(prop, str) or not prop:
        raise ValueError("prop must be a non-empty string")

    filter_min, filter_max = _validate_filter_options(
        filter_by,
        filter_min=filter_min,
        filter_max=filter_max,
        filter_prop=filter_prop,
    )

    requested_mode = _normalize_radial_reference_mode(reference_mode)
    interchain_filter_mode_name = _normalize_interchain_filter_mode(interchain_filter_mode)
    keys = sorted(data_by_set, key=_dataset_sort_key)

    selected_by_set: dict[str, np.ndarray] = {}
    mode_by_set: dict[str, str] = {}
    frames_by_set: dict[str, int] = {}
    reference_observations_by_set: dict[str, int] = {}
    finite_distances_by_set: dict[str, int] = {}

    for key in keys:
        result = data_by_set[key]
        array = np.asarray(_result_field(result, prop), dtype=float)
        if array.ndim != 2:
            raise ValueError(
                f"{key!r}: property {prop!r} must have shape "
                f"(n_rows, n_frames); got {array.shape}"
            )
        if array.shape[0] == 0:
            raise ValueError(f"{key!r}: property {prop!r} contains no rows")
        full_shape = array.shape

        mode = _infer_radial_reference_mode(result) if requested_mode == "auto" else requested_mode
        is_interchain_result = _result_has_field(result, "reference_chain_label")
        if interchain_filter_mode_name != "target" and not (
            mode == "per_frame" and is_interchain_result
        ):
            raise ValueError(
                "interchain_filter_mode='reference' or 'both' requires "
                "interchain-distance input with reference_chain_label"
            )

        start, stop = _resolve_frame_slice(
            array.shape[1],
            frame_start=frame_start,
            frame_stop=frame_stop,
            time_start=time_start,
            time_stop=time_stop,
            delta_t=delta_t,
        )
        selected = array[:, start:stop]
        primary_finite = np.isfinite(selected)
        reference_frame_mask: Optional[np.ndarray] = None

        if filter_by is not None:
            if mode == "per_frame" and is_interchain_result:
                if interchain_filter_mode_name in {"target", "both"}:
                    target_filter = _filter_array_for_key(
                        filter_by,
                        key,
                        expected_shape=full_shape,
                        filter_prop=filter_prop,
                        primary_result=result,
                    )[:, start:stop]
                    target_eligible = primary_finite & np.isfinite(target_filter)
                    target_pass = _filter_threshold_mask(
                        target_filter,
                        filter_min=filter_min,
                        filter_max=filter_max,
                    )
                else:
                    target_eligible = primary_finite
                    target_pass = np.ones_like(primary_finite, dtype=bool)

                if interchain_filter_mode_name in {"reference", "both"}:
                    reference_filter = _reference_filter_series_for_key(
                        filter_by,
                        key,
                        filter_prop=filter_prop,
                        primary_result=result,
                        expected_frames=full_shape[1],
                    )[start:stop]
                    reference_pass = _filter_threshold_mask(
                        reference_filter,
                        filter_min=filter_min,
                        filter_max=filter_max,
                    )
                else:
                    reference_pass = np.ones(selected.shape[1], dtype=bool)

                keep = primary_finite & target_pass & reference_pass[None, :]
                selected = np.where(keep, selected, np.nan)
                if interchain_filter_mode_name == "target":
                    reference_frame_mask = np.any(target_eligible, axis=0)
                else:
                    reference_frame_mask = reference_pass & np.any(target_eligible, axis=0)
            else:
                filter_array = _filter_array_for_key(
                    filter_by,
                    key,
                    expected_shape=full_shape,
                    filter_prop=filter_prop,
                    primary_result=result,
                )[:, start:stop]
                selected, eligible = _apply_secondary_filter(
                    selected,
                    filter_array,
                    filter_min=filter_min,
                    filter_max=filter_max,
                )
                reference_frame_mask = np.any(eligible, axis=0)
        else:
            reference_frame_mask = np.any(primary_finite, axis=0)

        finite = np.isfinite(selected)
        n_finite = int(np.sum(finite))
        if n_finite == 0:
            raise ValueError(
                f"{key!r}: no finite distances were found after applying the "
                "selected frame range and filters"
            )

        if mode == "per_row":
            n_reference_observations = n_finite
        else:
            assert reference_frame_mask is not None
            n_reference_observations = int(np.sum(reference_frame_mask))

        if n_reference_observations <= 0:
            raise ValueError(f"{key!r}: no valid reference observations")

        selected_by_set[key] = selected
        mode_by_set[key] = mode
        frames_by_set[key] = int(stop - start)
        reference_observations_by_set[key] = n_reference_observations
        finite_distances_by_set[key] = n_finite

    if np.isscalar(bins):
        if isinstance(bins, (bool, np.bool_)):
            raise TypeError("bins must be an integer or explicit bin edges")
        n_bins_float = float(bins)
        if not n_bins_float.is_integer():
            raise ValueError("bins must be an integer")
        n_bins = int(n_bins_float)
        if n_bins < 1:
            raise ValueError("bins must be at least 1")

        if value_range is None:
            maxima = []
            for selected in selected_by_set.values():
                finite_values = selected[np.isfinite(selected)]
                if finite_values.size:
                    maxima.append(float(np.max(finite_values)))
            if not maxima:
                raise ValueError(f"no finite values found for property {prop!r}")
            maximum = max(maxima)
            padding = 0.02 * maximum if maximum > 0.0 else 0.01
            value_range = (0.0, maximum + padding)

        if len(value_range) != 2:
            raise ValueError("value_range must contain exactly two values")
        range_min, range_max = (float(value_range[0]), float(value_range[1]))
        if not math.isfinite(range_min) or not math.isfinite(range_max):
            raise ValueError("value_range values must be finite")
        if range_min < 0.0:
            raise ValueError("the lower radial limit must be >= 0")
        if range_max <= range_min:
            raise ValueError("value_range must be an increasing (min, max) pair")
        edges = np.linspace(range_min, range_max, n_bins + 1)
    else:
        edges = np.asarray(bins, dtype=float)
        if edges.ndim != 1 or edges.size < 2:
            raise ValueError(
                "explicit bin edges must be a one-dimensional array with at " "least two values"
            )
        if not np.all(np.isfinite(edges)):
            raise ValueError("bin edges must all be finite")
        if edges[0] < 0.0:
            raise ValueError("radial bin edges must be >= 0")
        if not np.all(np.diff(edges) > 0.0):
            raise ValueError("bin edges must be strictly increasing")

    shell_volumes = (4.0 * math.pi / 3.0) * (np.power(edges[1:], 3) - np.power(edges[:-1], 3))
    if np.any(shell_volumes <= 0.0):
        raise ValueError("radial shell volumes must be positive")

    def density_for_set(key: str) -> tuple[np.ndarray, int]:
        selected = selected_by_set[key]
        values = selected[np.isfinite(selected)]
        counts, _ = np.histogram(values, bins=edges, density=False)
        n_in_range = int(np.sum(counts))
        density_values = counts.astype(float) / (
            float(reference_observations_by_set[key]) * shell_volumes
        )
        return density_values, n_in_range

    individual: dict[str, RadialNumberDensityResult] = {}
    for key in keys:
        density_values, n_in_range = density_for_set(key)
        individual[key] = RadialNumberDensityResult(
            values=density_values,
            stderr=np.zeros_like(density_values),
            edges=edges.copy(),
            n_distances=finite_distances_by_set[key],
            n_in_range=n_in_range,
            n_reference_observations=reference_observations_by_set[key],
            n_frames=frames_by_set[key],
            reference_mode=mode_by_set[key],
            n_sets=1,
            set_labels=(str(key),),
            per_set_values=density_values.reshape(1, -1),
            n_distances_per_set=(finite_distances_by_set[key],),
            n_reference_observations_per_set=(reference_observations_by_set[key],),
            n_frames_per_set=(frames_by_set[key],),
            aggregation="individual",
        )

    if not average_over_sets:
        return individual

    output: dict[str, RadialNumberDensityResult] = {}
    groups = _ordered_set_groups(keys, group_by=group_by)
    for group, member_keys in groups.items():
        modes = {mode_by_set[key] for key in member_keys}
        if len(modes) != 1:
            raise ValueError(
                f"group {group!r} mixes incompatible reference modes: " f"{sorted(modes)}"
            )

        per_set_values = np.stack([individual[key].values for key in member_keys], axis=0)
        mean_values = np.mean(per_set_values, axis=0)
        n_sets = int(per_set_values.shape[0])
        if n_sets < 2:
            stderr = np.zeros_like(mean_values)
        else:
            stderr = np.std(per_set_values, axis=0, ddof=1) / math.sqrt(float(n_sets))

        frame_counts = [frames_by_set[key] for key in member_keys]
        output[group] = RadialNumberDensityResult(
            values=mean_values,
            stderr=stderr,
            edges=edges.copy(),
            n_distances=int(sum(finite_distances_by_set[key] for key in member_keys)),
            n_in_range=int(sum(individual[key].n_in_range for key in member_keys)),
            n_reference_observations=int(
                sum(reference_observations_by_set[key] for key in member_keys)
            ),
            n_frames=_common_or_none(frame_counts),
            reference_mode=next(iter(modes)),
            n_sets=n_sets,
            set_labels=tuple(str(key) for key in member_keys),
            per_set_values=per_set_values,
            n_distances_per_set=tuple(finite_distances_by_set[key] for key in member_keys),
            n_reference_observations_per_set=tuple(
                reference_observations_by_set[key] for key in member_keys
            ),
            n_frames_per_set=tuple(frame_counts),
            aggregation="sets",
        )

    return output


def _radial_input_is_nested_by_reference(
    data: Mapping[str, Any],
    *,
    prop: str,
) -> bool:
    """Return whether *data* is ``reference -> set -> result``.

    A flat input maps set keys directly to result objects containing ``prop``.
    A nested interchain input maps each reference-segment label to a second
    mapping whose values contain ``prop``. Mixed flat/nested inputs are rejected
    because their averaging semantics would be ambiguous.
    """
    kinds: list[str] = []
    for key, value in data.items():
        if _result_has_field(value, prop):
            kinds.append("flat")
        elif isinstance(value, Mapping):
            if not value:
                raise ValueError(f"reference entry {key!r} is empty")
            bad = [
                inner_key
                for inner_key, result in value.items()
                if not _result_has_field(result, prop)
            ]
            if bad:
                raise ValueError(
                    f"reference entry {key!r} contains values without property " f"{prop!r}: {bad}"
                )
            kinds.append("nested")
        else:
            raise TypeError(
                f"{key!r}: expected a result containing {prop!r} or a mapping "
                "of simulation-set results"
            )

    unique = set(kinds)
    if len(unique) > 1:
        raise ValueError("data_by_set mixes flat results with reference-segment mappings")
    return unique == {"nested"}


def _average_reference_radial_results(
    results_by_reference: Mapping[str, Mapping[str, RadialNumberDensityResult]],
    *,
    reference_weighting: str = "equal",
) -> dict[str, RadialNumberDensityResult]:
    """Average available reference-specific interchain curves within each set.

    A reference/set combination may be absent when a reference-conditioned
    filter selects no frames for that reference. Such an empty conditional
    sample is omitted rather than treated as a zero-density curve or causing
    the full calculation to fail.

    With ``reference_weighting='equal'``, every retained reference curve has
    equal weight. With ``'observations'``, each curve is weighted by its number
    of valid reference observations (qualifying reference frames for
    reference-conditioned interchain filters). Observation weighting is
    equivalent to pooling the shell counts and denominators across references
    and prevents a sparsely sampled reference from receiving the same weight as
    one sampled in many frames.
    """
    reference_weighting_name = _normalize_reference_weighting(reference_weighting)
    if reference_weighting_name == "auto":
        reference_weighting_name = "equal"

    reference_labels = tuple(str(label) for label in results_by_reference)
    if not reference_labels:
        raise ValueError("no reference segments with valid observations were supplied")

    set_keys = sorted(
        {
            set_key
            for reference_results in results_by_reference.values()
            for set_key in reference_results
        },
        key=_dataset_sort_key,
    )
    if not set_keys:
        raise ValueError("no simulation sets contain valid reference observations")

    averaged: dict[str, RadialNumberDensityResult] = {}
    for set_key in set_keys:
        labels_for_set = tuple(
            reference_label
            for reference_label in reference_labels
            if set_key in results_by_reference[reference_label]
        )
        members = [
            results_by_reference[reference_label][set_key] for reference_label in labels_for_set
        ]
        if not members:
            continue

        modes = {member.reference_mode for member in members}
        if len(modes) != 1:
            raise ValueError(
                f"set {set_key!r} mixes incompatible reference modes: " f"{sorted(modes)}"
            )

        first_edges = members[0].edges
        if any(
            member.edges.shape != first_edges.shape or not np.array_equal(member.edges, first_edges)
            for member in members[1:]
        ):
            raise ValueError(f"reference-specific curves for set {set_key!r} do not share bins")

        per_reference_values = np.stack([member.values for member in members], axis=0)
        if reference_weighting_name == "equal":
            reference_weights = np.ones(len(members), dtype=np.float64)
        else:
            reference_weights = np.asarray(
                [member.n_reference_observations for member in members],
                dtype=np.float64,
            )
            if np.any(reference_weights <= 0.0):
                raise ValueError(
                    f"set {set_key!r} contains a reference with no valid " "reference observations"
                )
        mean_values = np.average(
            per_reference_values,
            axis=0,
            weights=reference_weights,
        )
        frame_counts = [int(member.n_frames) for member in members if member.n_frames is not None]
        common_frames = _common_or_none(frame_counts) if len(frame_counts) == len(members) else None

        averaged[set_key] = RadialNumberDensityResult(
            values=mean_values,
            # Reference segments are sampling origins, not independent
            # trajectory replicates. Replicate SEM is calculated across sets.
            stderr=np.zeros_like(mean_values),
            edges=first_edges.copy(),
            n_distances=int(sum(member.n_distances for member in members)),
            n_in_range=int(sum(member.n_in_range for member in members)),
            n_reference_observations=int(
                sum(member.n_reference_observations for member in members)
            ),
            n_frames=common_frames,
            reference_mode=next(iter(modes)),
            n_sets=1,
            set_labels=(str(set_key),),
            per_set_values=mean_values.reshape(1, -1),
            n_distances_per_set=(int(sum(member.n_distances for member in members)),),
            n_reference_observations_per_set=(
                int(sum(member.n_reference_observations for member in members)),
            ),
            n_frames_per_set=(int(common_frames) if common_frames is not None else 0,),
            n_reference_segments=len(labels_for_set),
            reference_segment_labels=labels_for_set,
            n_reference_segments_per_set=(len(labels_for_set),),
            reference_observations_per_segment=tuple(
                int(member.n_reference_observations) for member in members
            ),
            aggregation=(
                "reference_segments_observation_weighted"
                if reference_weighting_name == "observations"
                else "reference_segments_equal_weight"
            ),
        )

    if not averaged:
        raise ValueError("no simulation set has a reference chain passing the filter")
    return averaged


def _average_radial_results_over_sets(
    individual: Mapping[str, RadialNumberDensityResult],
    *,
    group_by: Optional[Mapping[str, str]],
) -> dict[str, RadialNumberDensityResult]:
    """Equal-weight average of already normalized set-level curves."""
    keys = sorted(individual, key=_dataset_sort_key)
    groups = _ordered_set_groups(keys, group_by=group_by)
    output: dict[str, RadialNumberDensityResult] = {}

    for group, member_keys in groups.items():
        members = [individual[key] for key in member_keys]
        modes = {member.reference_mode for member in members}
        if len(modes) != 1:
            raise ValueError(
                f"group {group!r} mixes incompatible reference modes: " f"{sorted(modes)}"
            )

        first_edges = members[0].edges
        if any(
            member.edges.shape != first_edges.shape or not np.array_equal(member.edges, first_edges)
            for member in members[1:]
        ):
            raise ValueError(f"group {group!r} does not share radial bins")

        per_set_values = np.stack([member.values for member in members], axis=0)
        mean_values = np.mean(per_set_values, axis=0)
        n_sets = int(per_set_values.shape[0])
        if n_sets < 2:
            stderr = np.zeros_like(mean_values)
        else:
            stderr = np.std(per_set_values, axis=0, ddof=1) / math.sqrt(float(n_sets))

        frame_counts = [int(member.n_frames) for member in members if member.n_frames is not None]
        common_frames = _common_or_none(frame_counts) if len(frame_counts) == len(members) else None

        # Reference-conditioned filters may leave different valid reference
        # chains in different sets. Each set has already been averaged over its
        # own available references, so retain the union only as metadata.
        reference_labels = tuple(
            dict.fromkeys(label for member in members for label in member.reference_segment_labels)
        )
        reference_observations_by_label = {label: 0 for label in reference_labels}
        for member in members:
            for label, count in zip(
                member.reference_segment_labels,
                member.reference_observations_per_segment,
            ):
                reference_observations_by_label[label] += int(count)
        reference_observations_per_segment = tuple(
            reference_observations_by_label[label] for label in reference_labels
        )

        output[group] = RadialNumberDensityResult(
            values=mean_values,
            stderr=stderr,
            edges=first_edges.copy(),
            n_distances=int(sum(member.n_distances for member in members)),
            n_in_range=int(sum(member.n_in_range for member in members)),
            n_reference_observations=int(
                sum(member.n_reference_observations for member in members)
            ),
            n_frames=common_frames,
            reference_mode=next(iter(modes)),
            n_sets=n_sets,
            set_labels=tuple(str(key) for key in member_keys),
            per_set_values=per_set_values,
            n_distances_per_set=tuple(int(member.n_distances) for member in members),
            n_reference_observations_per_set=tuple(
                int(member.n_reference_observations) for member in members
            ),
            n_frames_per_set=tuple(
                int(member.n_frames) if member.n_frames is not None else 0 for member in members
            ),
            n_reference_segments=(len(reference_labels) if reference_labels else 0),
            reference_segment_labels=reference_labels,
            n_reference_segments_per_set=tuple(
                int(member.n_reference_segments) for member in members
            ),
            reference_observations_per_segment=(reference_observations_per_segment),
            aggregation=("reference_segments_then_sets" if reference_labels else "sets"),
        )

    return output


def calculate_radial_number_density(
    data_by_set: Mapping[str, Any],
    prop: str = "distance_per_chain_nm",
    *,
    bins: Union[int, Sequence[float], np.ndarray] = 60,
    value_range: Optional[tuple[float, float]] = (0.0, 8.0),
    reference_mode: str = "auto",
    average_over_sets: bool = False,
    group_by: Optional[Mapping[str, str]] = None,
    filter_by: Optional[Mapping[str, Any]] = None,
    filter_min: Optional[float] = None,
    filter_max: Optional[float] = None,
    filter_prop: Optional[str] = None,
    interchain_filter_mode: str = "target",
    reference_weighting: str = "auto",
    min_reference_observations: int = 1,
    frame_start: Optional[int] = None,
    frame_stop: Optional[int] = None,
    time_start: Any = None,
    time_stop: Any = None,
    delta_t: Any = None,
) -> dict[str, RadialNumberDensityResult]:
    """Convert intra- or interchain pair distances to radial number density.

    Accepted input layouts
    ----------------------
    Flat intrachain or interchain data::

        {set_key: distance_result}

    Nested interchain data calculated once per reference segment::

        {reference_segment: {set_key: interchain_distance_result}}

    Nested interchain input is reduced in this order:

    1. Calculate one normalized number-density curve for every reference
       segment within every simulation set.
    2. Combine those curves over reference segments within each set using
       ``reference_weighting``.
    3. If ``average_over_sets=True``, average the resulting set curves with
       equal set weight and calculate SEM across sets.

    Thus, reference segments define sampling origins within a trajectory, while
    simulation sets define independent replicates for uncertainty estimation.

    Other arguments, shell normalization, reference-mode inference, and time
    slicing follow the flat implementation described in
    :func:`_calculate_radial_number_density_flat`.

    For nested interchain input, ``filter_by`` may either mirror the nested
    ``reference_segment -> set -> result`` layout or be one flat
    ``set -> result`` mapping reused for every reference segment. Filter rows
    are aligned by ``chain_labels``. ``interchain_filter_mode`` may be
    ``"target"``, ``"reference"``, or ``"both"``.

    ``reference_weighting='auto'`` uses observation weighting for
    reference-conditioned modes (``"reference"`` and ``"both"``) and equal
    reference weighting for target-only or unfiltered RDFs. Use
    ``reference_weighting='equal'`` to reproduce the older behavior, or
    ``'observations'`` to pool reference observations explicitly.
    ``min_reference_observations`` can exclude reference/set combinations with
    too few valid reference frames. Extra filter entries are ignored.
    """
    if not data_by_set:
        raise ValueError("data_by_set is empty")
    if not isinstance(prop, str) or not prop:
        raise ValueError("prop must be a non-empty string")

    filter_min, filter_max = _validate_filter_options(
        filter_by,
        filter_min=filter_min,
        filter_max=filter_max,
        filter_prop=filter_prop,
    )
    interchain_filter_mode_name = _normalize_interchain_filter_mode(interchain_filter_mode)
    reference_weighting_name = _normalize_reference_weighting(reference_weighting)
    if reference_weighting_name == "auto":
        reference_weighting_name = (
            "observations" if interchain_filter_mode_name in {"reference", "both"} else "equal"
        )

    if isinstance(min_reference_observations, (bool, np.bool_)):
        raise TypeError("min_reference_observations must be an integer >= 1")
    try:
        min_reference_observations_float = float(min_reference_observations)
    except (TypeError, ValueError) as exc:
        raise TypeError("min_reference_observations must be an integer >= 1") from exc
    if (
        not math.isfinite(min_reference_observations_float)
        or not min_reference_observations_float.is_integer()
        or min_reference_observations_float < 1
    ):
        raise ValueError("min_reference_observations must be an integer >= 1")
    min_reference_observations_i = int(min_reference_observations_float)

    is_nested = _radial_input_is_nested_by_reference(data_by_set, prop=prop)
    if not is_nested:
        return _calculate_radial_number_density_flat(
            data_by_set,
            prop=prop,
            bins=bins,
            value_range=value_range,
            reference_mode=reference_mode,
            average_over_sets=average_over_sets,
            group_by=group_by,
            filter_by=filter_by,
            filter_min=filter_min,
            filter_max=filter_max,
            filter_prop=filter_prop,
            interchain_filter_mode=interchain_filter_mode,
            frame_start=frame_start,
            frame_stop=frame_stop,
            time_start=time_start,
            time_stop=time_stop,
            delta_t=delta_t,
        )

    results_by_reference: dict[str, dict[str, RadialNumberDensityResult]] = {}
    skipped_reference_sets: list[tuple[str, str, int]] = []

    for reference_label, set_results in data_by_set.items():
        assert isinstance(set_results, Mapping)
        reference_filter: Optional[Mapping[str, Any]] = None
        if filter_by is not None:
            # Accept either a fully nested filter mapping
            #   reference_label -> set_key -> result
            # or one flat set mapping
            #   set_key -> result
            # that is reused for every reference chain.
            candidate_filter = filter_by.get(reference_label)
            if isinstance(candidate_filter, Mapping) and all(
                set_key in candidate_filter for set_key in set_results
            ):
                reference_filter = candidate_filter
            elif all(set_key in filter_by for set_key in set_results):
                reference_filter = filter_by
            else:
                missing_flat = [set_key for set_key in set_results if set_key not in filter_by]
                raise KeyError(
                    "filter_by must either mirror the nested reference layout "
                    f"or contain the simulation-set keys directly. For reference "
                    f"{reference_label!r}, missing flat set keys: {missing_flat}"
                )

        reference_results: dict[str, RadialNumberDensityResult] = {}
        for set_key, set_result in set_results.items():
            one_filter = None if reference_filter is None else {set_key: reference_filter[set_key]}
            try:
                one_result = _calculate_radial_number_density_flat(
                    {set_key: set_result},
                    prop=prop,
                    bins=bins,
                    value_range=value_range,
                    reference_mode=reference_mode,
                    average_over_sets=False,
                    group_by=None,
                    filter_by=one_filter,
                    filter_min=filter_min,
                    filter_max=filter_max,
                    filter_prop=filter_prop,
                    interchain_filter_mode=interchain_filter_mode_name,
                    frame_start=frame_start,
                    frame_stop=frame_stop,
                    time_start=time_start,
                    time_stop=time_stop,
                    delta_t=delta_t,
                )
            except ValueError as exc:
                message = str(exc)
                empty_conditional_sample = (
                    "no finite distances were found after applying" in message
                    or "no valid reference observations" in message
                )
                if (
                    interchain_filter_mode_name in {"reference", "both"}
                    and empty_conditional_sample
                ):
                    skipped_reference_sets.append((str(reference_label), str(set_key), 0))
                    continue
                raise

            result_for_set = one_result[set_key]
            if result_for_set.n_reference_observations < min_reference_observations_i:
                skipped_reference_sets.append(
                    (
                        str(reference_label),
                        str(set_key),
                        int(result_for_set.n_reference_observations),
                    )
                )
                continue

            reference_results.update(one_result)

        if reference_results:
            results_by_reference[str(reference_label)] = reference_results

    if not results_by_reference:
        detail = (
            "; skipped reference/set combinations: "
            + ", ".join(
                f"{ref}/{set_key} (n={n_obs})" for ref, set_key, n_obs in skipped_reference_sets
            )
            if skipped_reference_sets
            else ""
        )
        raise ValueError(
            "no reference chain satisfies the selected reference filter and "
            "minimum observation requirement in any "
            f"simulation set{detail}"
        )

    # All nested calls must use identical radial edges. When value_range=None,
    # independently inferred ranges could differ, so reject that ambiguous use.
    if value_range is None and np.isscalar(bins):
        edge_arrays = [
            result.edges
            for reference_results in results_by_reference.values()
            for result in reference_results.values()
        ]
        first_edges = edge_arrays[0]
        if any(
            edges.shape != first_edges.shape or not np.array_equal(edges, first_edges)
            for edges in edge_arrays[1:]
        ):
            raise ValueError(
                "nested reference-segment input with value_range=None produced "
                "different bin edges; supply an explicit value_range or explicit "
                "bin edges"
            )

    per_set = _average_reference_radial_results(
        results_by_reference,
        reference_weighting=reference_weighting_name,
    )
    if not average_over_sets:
        return per_set
    return _average_radial_results_over_sets(per_set, group_by=group_by)


def calculate_averages(
    data_by_set: Mapping[str, Any],
    prop: str = "rg_per_chain_nm",
    *,
    average_over_sets: bool = False,
    group_by: Optional[Mapping[str, str]] = None,
    filter_by: Optional[Mapping[str, Any]] = None,
    filter_min: Optional[float] = None,
    filter_max: Optional[float] = None,
    filter_prop: Optional[str] = None,
    frame_start: Optional[int] = None,
    frame_stop: Optional[int] = None,
    time_start: Any = None,
    time_stop: Any = None,
    delta_t: Any = None,
    n_blocks: int = 3,
) -> dict[str, AverageResult]:
    """Average a 2D property over selected frames and then across chains or sets.

    For every input dataset, one finite-value time average is first calculated
    for each chain.  In the default mode, the returned mean and uncertainty are
    calculated across those per-chain averages independently for each dataset.

    With ``average_over_sets=True``, each dataset is first reduced to its own
    chain-averaged mean.  The final mean, sample standard deviation, and standard
    error are calculated across those set means, giving every set equal weight.
    Input keys of the form ``<set>_<group>`` are grouped by the suffix; use
    ``group_by`` to explicitly map input keys to output groups.  If the keys
    contain no underscores, all entries are returned under ``"all_sets"``.

    Frame selection is identical to :func:`calculate_histograms`: specify either
    ``frame_start``/``frame_stop`` or ``time_start``/``time_stop`` together with
    ``delta_t``.  Stop limits are exclusive.  Numeric times are nanoseconds.

    ``filter_by`` may supply a second result mapping keyed like ``data_by_set``.
    Each primary chain/frame value is included only where the corresponding
    filter value is finite and satisfies the inclusive ``filter_min`` and/or
    ``filter_max`` limits. The filter property must have exactly the same shape
    as the primary property. Use ``filter_prop`` to select its field explicitly;
    otherwise a unique matching two-dimensional numeric field is inferred.

    When a dataset contains only one valid chain/time series, uncertainty cannot
    be estimated across chains. In that case the selected time series is divided
    into ``n_blocks`` contiguous frame blocks (3 by default). The mean of each
    populated block is calculated, and ``std`` and ``stderr`` are the sample
    standard deviation and standard error across those block means. Filtering is
    applied before block averaging. At least two populated blocks are required.
    """
    if not data_by_set:
        raise ValueError("data_by_set is empty")
    if not isinstance(prop, str) or not prop:
        raise ValueError("prop must be a non-empty string")

    if isinstance(n_blocks, (bool, np.bool_)):
        raise TypeError("n_blocks must be an integer >= 2")
    try:
        n_blocks_float = float(n_blocks)
    except (TypeError, ValueError) as exc:
        raise TypeError("n_blocks must be an integer >= 2") from exc
    if not math.isfinite(n_blocks_float) or not n_blocks_float.is_integer() or n_blocks_float < 2:
        raise ValueError("n_blocks must be an integer >= 2")
    n_blocks_i = int(n_blocks_float)

    filter_min, filter_max = _validate_filter_options(
        filter_by,
        filter_min=filter_min,
        filter_max=filter_max,
        filter_prop=filter_prop,
    )

    keys = sorted(data_by_set, key=_dataset_sort_key)
    individual: dict[str, AverageResult] = {}

    for key in keys:
        array = np.asarray(_result_field(data_by_set[key], prop), dtype=float)
        if array.ndim != 2:
            raise ValueError(
                f"{key!r}: property {prop!r} must have shape "
                f"(n_chains, n_frames); got {array.shape}"
            )
        if array.shape[0] == 0:
            raise ValueError(f"{key!r}: property {prop!r} contains no chains")
        full_shape = array.shape

        start, stop = _resolve_frame_slice(
            array.shape[1],
            frame_start=frame_start,
            frame_stop=frame_stop,
            time_start=time_start,
            time_stop=time_stop,
            delta_t=delta_t,
        )
        selected = array[:, start:stop]
        if filter_by is not None:
            filter_array = _filter_array_for_key(
                filter_by,
                key,
                expected_shape=full_shape,
                filter_prop=filter_prop,
                primary_result=data_by_set[key],
            )[:, start:stop]
            selected, _ = _apply_secondary_filter(
                selected,
                filter_array,
                filter_min=filter_min,
                filter_max=filter_max,
            )

        finite = np.isfinite(selected)
        counts = np.sum(finite, axis=1)
        sums = np.sum(np.where(finite, selected, 0.0), axis=1)
        per_chain_mean = np.full(array.shape[0], np.nan, dtype=float)
        valid = counts > 0
        per_chain_mean[valid] = sums[valid] / counts[valid]

        chain_values = per_chain_mean[valid]
        n_valid = int(chain_values.size)
        if n_valid == 0:
            raise ValueError(f"{key!r}: no finite values were found in the selected frames")

        mean = float(np.mean(chain_values))
        aggregation = "chains"
        if n_valid >= 2:
            std = float(np.std(chain_values, ddof=1))
            stderr = std / math.sqrt(float(n_valid))
        else:
            # With only one chain, estimate uncertainty from contiguous time
            # blocks rather than reporting a misleading zero uncertainty.
            series = selected[int(np.flatnonzero(valid)[0]), :]
            frame_indices = np.arange(series.size, dtype=np.int64)
            blocks = np.array_split(frame_indices, min(n_blocks_i, series.size))
            block_means: list[float] = []
            for block in blocks:
                if block.size == 0:
                    continue
                block_values = series[block]
                finite_block = block_values[np.isfinite(block_values)]
                if finite_block.size:
                    block_means.append(float(np.mean(finite_block)))

            if len(block_means) < 2:
                raise ValueError(
                    f"{key!r}: only one valid chain is available and fewer than "
                    "two populated time blocks remain after selection/filtering; "
                    "cannot estimate uncertainty by block averaging"
                )

            block_values = np.asarray(block_means, dtype=float)
            std = float(np.std(block_values, ddof=1))
            stderr = std / math.sqrt(float(block_values.size))
            aggregation = "time_blocks"

        n_frames = int(stop - start)
        individual[key] = AverageResult(
            mean=mean,
            std=std,
            stderr=float(stderr),
            per_chain_mean=per_chain_mean,
            n_chains=n_valid,
            n_total_chains=int(array.shape[0]),
            n_frames=n_frames,
            frame_start=int(start),
            frame_stop=int(stop),
            per_set_mean=np.asarray([mean], dtype=float),
            n_sets=1,
            set_labels=(str(key),),
            aggregation=aggregation,
            n_frames_per_set=(n_frames,),
        )

    if not average_over_sets:
        return individual

    summaries: dict[str, AverageResult] = {}
    groups = _ordered_set_groups(keys, group_by=group_by)
    for group, member_keys in groups.items():
        member_results = [individual[key] for key in member_keys]
        per_set_mean = np.asarray([result.mean for result in member_results], dtype=float)
        n_sets = int(per_set_mean.size)
        mean = float(np.mean(per_set_mean))
        if n_sets >= 2:
            std = float(np.std(per_set_mean, ddof=1))
            stderr = std / math.sqrt(float(n_sets))
            summary_aggregation = "sets"
        else:
            # Preserve the single-set uncertainty estimate. This may be the
            # block-averaged uncertainty when that set contains one time series.
            std = float(member_results[0].std)
            stderr = float(member_results[0].stderr)
            summary_aggregation = member_results[0].aggregation

        frame_starts = [int(result.frame_start) for result in member_results]
        frame_stops = [int(result.frame_stop) for result in member_results]
        frame_counts = [int(result.n_frames) for result in member_results]

        summaries[group] = AverageResult(
            mean=mean,
            std=std,
            stderr=float(stderr),
            per_chain_mean=np.concatenate([result.per_chain_mean for result in member_results]),
            n_chains=int(sum(result.n_chains for result in member_results)),
            n_total_chains=int(sum(result.n_total_chains for result in member_results)),
            n_frames=_common_or_none(frame_counts),
            frame_start=_common_or_none(frame_starts),
            frame_stop=_common_or_none(frame_stops),
            per_set_mean=per_set_mean,
            n_sets=n_sets,
            set_labels=tuple(str(key) for key in member_keys),
            aggregation=summary_aggregation,
            n_frames_per_set=tuple(frame_counts),
        )

    return summaries


def _delta_t_to_ns(delta_t: Union[float, int, object]) -> float:
    """Convert a numeric nanosecond interval or OpenMM Quantity to ns."""
    dt_ns = _time_value_to_ns(delta_t, name="delta_t")
    if dt_ns <= 0.0:
        raise ValueError("delta_t must be > 0")
    return dt_ns


# --- two-state contact kinetics from distance time series ----------------------


@dataclass(frozen=True)
class OnOffRatesResult:
    """Two-state kinetics and plot_timeseries-compatible state trajectories.

    State arrays have shape (n_chains, n_frames), with 1 = ON/contact,
    0 = OFF/no contact, and NaN = unknown or outside the analysis window.
    ``raw_state_per_chain`` is the hysteresis-classified state before minimum
    residence filtering; ``state_per_chain`` is the final, filtered state.

    Pooled ``k_on_per_s`` and ``k_off_per_s`` are event counts divided by the
    total OFF and ON exposure time, respectively, across all chains. These are
    first-order transition frequencies, not bimolecular association constants.
    Per-chain rates and exposures are also supplied. A rate is NaN when its
    originating state has no measurable exposure, and zero when it has exposure
    but no observed transitions.

    ``on_dwell_times_ns`` / ``off_dwell_times_ns`` include only fully observed
    runs bounded by transitions on *both* sides. Runs cut off by trajectory
    endpoints, missing data, or an analysis-window boundary are censored and
    excluded from these distributions, but their observed intervals contribute
    to the exposure-based rates.
    """

    state_per_chain: np.ndarray
    raw_state_per_chain: np.ndarray
    chain_labels: tuple[str, ...]
    k_on_per_s: float
    k_off_per_s: float
    k_on_per_chain_per_s: np.ndarray
    k_off_per_chain_per_s: np.ndarray
    n_on: int
    n_off: int
    n_on_per_chain: np.ndarray
    n_off_per_chain: np.ndarray
    t_on_ns: float
    t_off_ns: float
    t_on_per_chain_ns: np.ndarray
    t_off_per_chain_ns: np.ndarray
    fraction_on: float
    fraction_on_per_chain: np.ndarray
    tau_on_ns: float
    tau_off_ns: float
    on_dwell_times_ns: np.ndarray
    off_dwell_times_ns: np.ndarray
    on_dwell_times_per_chain_ns: tuple[np.ndarray, ...]
    off_dwell_times_per_chain_ns: tuple[np.ndarray, ...]
    on_cutoff_nm: float
    off_cutoff_nm: float
    min_on_time_ns: float
    min_off_time_ns: float
    delta_t_ns: float
    frame_start: int
    frame_stop: int
    initial_state: str

    @property
    def fraction_bound(self) -> float:
        """Alias for fraction_on."""
        return self.fraction_on

    @property
    def tau_bound_ns(self) -> float:
        """ON-state mean lifetime inferred from the OFF transition rate."""
        return self.tau_on_ns

    @property
    def tau_unbound_ns(self) -> float:
        """OFF-state mean lifetime inferred from the ON transition rate."""
        return self.tau_off_ns


def _on_off_cutoffs(
    cutoff_nm: Optional[float],
    *,
    hysteresis_nm: float,
    on_cutoff_nm: Optional[float],
    off_cutoff_nm: Optional[float],
) -> tuple[float, float]:
    """Resolve single cutoff, symmetric hysteresis, or two explicit cutoffs."""
    h = float(hysteresis_nm)
    if not math.isfinite(h) or h < 0.0:
        raise ValueError("hysteresis_nm must be finite and >= 0")

    if (on_cutoff_nm is None) != (off_cutoff_nm is None):
        raise ValueError("on_cutoff_nm and off_cutoff_nm must be supplied together")
    if on_cutoff_nm is not None:
        if h != 0.0:
            raise ValueError("use either explicit on/off cutoffs or hysteresis_nm")
        lower, upper = float(on_cutoff_nm), float(off_cutoff_nm)
    else:
        if cutoff_nm is None:
            raise ValueError("cutoff_nm or both explicit on/off cutoffs are required")
        center = float(cutoff_nm)
        lower, upper = center - 0.5 * h, center + 0.5 * h

    if not (math.isfinite(lower) and math.isfinite(upper)):
        raise ValueError("distance cutoffs must be finite")
    if lower < 0.0 or upper <= 0.0 or upper < lower:
        raise ValueError("cutoffs must satisfy 0 <= on_cutoff_nm <= off_cutoff_nm")
    return lower, upper


def _on_off_minimum_frames(value_ns: float, dt_ns: float) -> int:
    """Convert a minimum dwell to the number of sampled frame intervals."""
    # Roundoff tolerance avoids ceil(3.0000000000000004) = 4.
    return max(1, int(math.ceil(value_ns / dt_ns - 1.0e-12)))


def _on_off_states_one_series(
    distances: np.ndarray,
    *,
    on_cutoff_nm: float,
    off_cutoff_nm: float,
    min_on_frames: int,
    min_off_frames: int,
    initial_state: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Classify one series; missing values break continuity and reset history.

    Minimum dwell filtering is a retrospective debounce: a proposed transition
    is accepted only if the *provisional* new state persists for enough sampled
    frames. Accepted transitions are assigned to the initial cutoff crossing,
    not to the end of the confirmation period. Short excursions are replaced by
    the previously confirmed state. The first identified state in each valid
    segment is accepted as initial, never counted as a transition.
    """
    n_frames = distances.size
    raw = np.full(n_frames, np.nan, dtype=float)
    filtered = np.full(n_frames, np.nan, dtype=float)
    is_finite = np.isfinite(distances)
    edges = np.flatnonzero(np.diff(np.r_[False, is_finite, False].astype(np.int8)))

    for start, stop in zip(edges[::2], edges[1::2]):
        values = distances[start:stop]
        n = stop - start
        # -1 is unresolved in the hysteresis interval. At equal cutoffs,
        # a value exactly at the cutoff is OFF.
        forced = np.full(n, -1, dtype=np.int8)
        forced[values < on_cutoff_nm] = 1
        forced[values >= off_cutoff_nm] = 0
        if forced[0] == -1 and initial_state != "auto":
            forced[0] = 1 if initial_state == "on" else 0

        prior = np.maximum.accumulate(np.where(forced != -1, np.arange(n, dtype=np.int64), -1))
        known = prior >= 0
        provisional = np.full(n, -1, dtype=np.int8)
        provisional[known] = forced[prior[known]]
        raw_segment = raw[start:stop]
        raw_segment[known] = provisional[known]

        if not np.any(known):
            continue
        first = int(np.flatnonzero(known)[0])
        changes = np.flatnonzero(np.diff(provisional[first:]) != 0) + first + 1
        boundaries = np.r_[first, changes, n]
        accepted = int(provisional[first])
        filtered_segment = filtered[start:stop]
        filtered_segment[first : boundaries[1]] = accepted
        for run in range(1, boundaries.size - 1):
            left = int(boundaries[run])
            right = int(boundaries[run + 1])
            proposed = int(provisional[left])
            needed = min_on_frames if proposed == 1 else min_off_frames
            if proposed != accepted and (right - left) >= needed:
                accepted = proposed
            filtered_segment[left:right] = accepted

    return raw, filtered


def _on_off_complete_dwells(states: np.ndarray, dt_ns: float) -> tuple[np.ndarray, np.ndarray]:
    """Return only dwell runs bracketed by observed transitions."""
    complete_on: list[float] = []
    complete_off: list[float] = []
    known = np.isfinite(states)
    edges = np.flatnonzero(np.diff(np.r_[False, known, False].astype(np.int8)))
    for start, stop in zip(edges[::2], edges[1::2]):
        values = states[start:stop]
        changes = np.flatnonzero(np.diff(values) != 0) + 1
        boundaries = np.r_[0, changes, values.size]
        # Exclude first and last run of each uninterrupted observation span.
        for run in range(1, boundaries.size - 2):
            left = int(boundaries[run])
            right = int(boundaries[run + 1])
            lifetime_ns = (right - left) * dt_ns
            if values[left] == 1:
                complete_on.append(lifetime_ns)
            else:
                complete_off.append(lifetime_ns)
    return np.asarray(complete_on, dtype=float), np.asarray(complete_off, dtype=float)


def _on_off_rate(count: int, exposure_ns: float) -> float:
    """Maximum-likelihood first-order rate (s^-1), given observed exposure."""
    return float(count * 1.0e9 / exposure_ns) if exposure_ns > 0.0 else float("nan")


def _on_off_lifetime_ns(rate_per_s: float) -> float:
    """Inverse rate in nanoseconds, preserving undefined zero-exposure cases."""
    if math.isnan(rate_per_s):
        return float("nan")
    return float("inf") if rate_per_s == 0.0 else float(1.0e9 / rate_per_s)


def calculate_on_off_rates(
    data: Any,
    cutoff_nm: Optional[float] = None,
    delta_t: Any = None,
    *,
    prop: str = "distance_per_chain_nm",
    hysteresis_nm: float = 0.0,
    on_cutoff_nm: Optional[float] = None,
    off_cutoff_nm: Optional[float] = None,
    min_residence_time: Any = None,
    min_on_time: Any = None,
    min_off_time: Any = None,
    initial_state: str = "auto",
    frame_start: Optional[int] = None,
    frame_stop: Optional[int] = None,
    time_start: Any = None,
    time_stop: Any = None,
) -> Any:
    """Calculate contact ON/OFF rates from distance time series.

    ``data`` can be a result with a ``(n_chains, n_frames)`` distance property,
    a plain 1D/2D distance array, or a (possibly nested) ``get_values`` mapping.
    A single result returns :class:`OnOffRatesResult`; mappings return matching
    mappings of results. ``prop`` selects the distance field (in nm). A 1D
    array is interpreted as one chain.

    State definitions (strict at the ON threshold):

    * Single cutoff: ON when r < cutoff_nm, otherwise OFF.
    * Symmetric hysteresis: ON for r < cutoff_nm-hysteresis_nm/2;
      OFF for r >= cutoff_nm+hysteresis_nm/2; otherwise retain state.
    * Explicit hysteresis: provide both on_cutoff_nm and off_cutoff_nm.
      Explicit thresholds supersede cutoff_nm and cannot be combined with
      nonzero hysteresis_nm.

    ``initial_state='auto'`` leaves samples inside the hysteresis band unknown
    until the first threshold crossing. ``'on'`` or ``'off'`` instead assumes
    that state for an initially ambiguous sample. NaN/inf distances reset state
    history; transitions are never inferred across missing samples.

    ``min_residence_time`` specifies a common minimum run duration for a newly
    entered state. ``min_on_time`` and ``min_off_time`` override it for each
    candidate state. Numeric times are in ns (OpenMM quantities are accepted).
    A state change must persist for at least ceil(minimum/delta_t) consecutive
    sampled states, and is then assigned back to its first threshold-crossing
    frame. Short excursions are suppressed. The first classified state of a
    continuous valid interval is not subject to this minimum; it is left-censored.

    Rates use observed adjacent-frame exposure only:
    k_on = (# OFF->ON) / total OFF exposure;
    k_off = (# ON->OFF) / total ON exposure.
    Output rates are s^-1 and exposures/dwells are ns. This is a sampled,
    operational two-state rate and need not equal a microscopic kinetic rate.

    Frame/time limits follow ``calculate_averages`` (stop exclusive). Returned
    state arrays retain the *original full frame count*, with NaN outside the
    chosen window, so ``plot_timeseries`` preserves the original time axis.

    Example::

        kinetics = calculate_on_off_rates(
            distances, cutoff_nm=0.8, hysteresis_nm=0.2,
            min_on_time=0.5, min_off_time=1.0, delta_t=0.1,
        )
        ax = plot_timeseries(kinetics["set1_asn"], "state_per_chain", 0.1,
                             ylim=(-0.1, 1.1))
    """
    if delta_t is None:
        raise ValueError("delta_t is required (in ns for numeric values)")
    dt_ns = _delta_t_to_ns(delta_t)
    if not isinstance(prop, str) or not prop:
        raise ValueError("prop must be a non-empty distance-field name")
    on_nm, off_nm = _on_off_cutoffs(
        cutoff_nm,
        hysteresis_nm=hysteresis_nm,
        on_cutoff_nm=on_cutoff_nm,
        off_cutoff_nm=off_cutoff_nm,
    )

    state_name = str(initial_state).strip().lower()
    if state_name not in {"auto", "on", "off"}:
        raise ValueError("initial_state must be 'auto', 'on', or 'off'")
    common_min_ns = (
        0.0
        if min_residence_time is None
        else _time_value_to_ns(min_residence_time, name="min_residence_time")
    )
    on_min_ns = (
        common_min_ns if min_on_time is None else _time_value_to_ns(min_on_time, name="min_on_time")
    )
    off_min_ns = (
        common_min_ns
        if min_off_time is None
        else _time_value_to_ns(min_off_time, name="min_off_time")
    )
    if min(common_min_ns, on_min_ns, off_min_ns) < 0.0:
        raise ValueError("minimum residence times must be >= 0")
    on_frames = _on_off_minimum_frames(on_min_ns, dt_ns)
    off_frames = _on_off_minimum_frames(off_min_ns, dt_ns)

    # A dictionary of distance arrays is a result; a dictionary of named
    # simulation results is a container. Keep the original nesting and keys.
    if isinstance(data, Mapping) and not _result_has_field(data, prop):
        if not data:
            raise ValueError("data mapping is empty")
        return {
            key: calculate_on_off_rates(
                item,
                cutoff_nm=cutoff_nm,
                delta_t=delta_t,
                prop=prop,
                hysteresis_nm=hysteresis_nm,
                on_cutoff_nm=on_cutoff_nm,
                off_cutoff_nm=off_cutoff_nm,
                min_residence_time=min_residence_time,
                min_on_time=min_on_time,
                min_off_time=min_off_time,
                initial_state=state_name,
                frame_start=frame_start,
                frame_stop=frame_stop,
                time_start=time_start,
                time_stop=time_stop,
            )
            for key, item in data.items()
        }

    if isinstance(data, (np.ndarray, list, tuple)):
        distances = np.asarray(data, dtype=float)
        labels = None
    else:
        distances = np.asarray(_result_field(data, prop), dtype=float)
        labels = (
            tuple(str(x) for x in _result_field(data, "chain_labels"))
            if _result_has_field(data, "chain_labels")
            else None
        )
    if distances.ndim == 1:
        distances = distances.reshape(1, -1)
    if distances.ndim != 2:
        raise ValueError(f"{prop!r} must have shape (n_chains, n_frames)")
    n_chains, n_frames = distances.shape
    if n_chains == 0 or n_frames < 2:
        raise ValueError("distance input requires >=1 chain and >=2 frames")
    if labels is None:
        labels = tuple(f"chain {i}" for i in range(n_chains))
    if len(labels) != n_chains or len(set(labels)) != n_chains:
        raise ValueError("chain_labels must have one unique label per distance row")

    start, stop = _resolve_frame_slice(
        n_frames,
        frame_start=frame_start,
        frame_stop=frame_stop,
        time_start=time_start,
        time_stop=time_stop,
        delta_t=dt_ns,
    )
    if stop - start < 2:
        raise ValueError("selected analysis window requires at least two frames")

    raw_states = np.full((n_chains, n_frames), np.nan, dtype=float)
    states = np.full((n_chains, n_frames), np.nan, dtype=float)
    n_on_chain = np.zeros(n_chains, dtype=np.int64)
    n_off_chain = np.zeros(n_chains, dtype=np.int64)
    t_on_chain = np.zeros(n_chains, dtype=float)
    t_off_chain = np.zeros(n_chains, dtype=float)
    dwell_on_by_chain: list[np.ndarray] = []
    dwell_off_by_chain: list[np.ndarray] = []

    for chain in range(n_chains):
        raw, accepted = _on_off_states_one_series(
            distances[chain, start:stop],
            on_cutoff_nm=on_nm,
            off_cutoff_nm=off_nm,
            min_on_frames=on_frames,
            min_off_frames=off_frames,
            initial_state=state_name,
        )
        raw_states[chain, start:stop] = raw
        states[chain, start:stop] = accepted
        valid_pair = np.isfinite(accepted[:-1]) & np.isfinite(accepted[1:])
        was_on = valid_pair & (accepted[:-1] == 1)
        was_off = valid_pair & (accepted[:-1] == 0)
        n_on_chain[chain] = np.count_nonzero(was_off & (accepted[1:] == 1))
        n_off_chain[chain] = np.count_nonzero(was_on & (accepted[1:] == 0))
        t_on_chain[chain] = np.count_nonzero(was_on) * dt_ns
        t_off_chain[chain] = np.count_nonzero(was_off) * dt_ns
        on_dwells, off_dwells = _on_off_complete_dwells(accepted, dt_ns)
        dwell_on_by_chain.append(on_dwells)
        dwell_off_by_chain.append(off_dwells)

    k_on_by_chain = np.array(
        [_on_off_rate(int(n), float(t)) for n, t in zip(n_on_chain, t_off_chain)]
    )
    k_off_by_chain = np.array(
        [_on_off_rate(int(n), float(t)) for n, t in zip(n_off_chain, t_on_chain)]
    )
    total_on = float(np.sum(t_on_chain))
    total_off = float(np.sum(t_off_chain))
    n_on = int(np.sum(n_on_chain))
    n_off = int(np.sum(n_off_chain))
    k_on = _on_off_rate(n_on, total_off)
    k_off = _on_off_rate(n_off, total_on)
    denominator = t_on_chain + t_off_chain
    fractions = np.full(n_chains, np.nan, dtype=float)
    np.divide(t_on_chain, denominator, out=fractions, where=denominator > 0.0)
    fraction_total = (
        float(total_on / (total_on + total_off)) if (total_on + total_off) > 0.0 else float("nan")
    )
    dwell_on = np.concatenate(dwell_on_by_chain) if dwell_on_by_chain else np.empty(0, dtype=float)
    dwell_off = (
        np.concatenate(dwell_off_by_chain) if dwell_off_by_chain else np.empty(0, dtype=float)
    )

    return OnOffRatesResult(
        state_per_chain=states,
        raw_state_per_chain=raw_states,
        chain_labels=labels,
        k_on_per_s=k_on,
        k_off_per_s=k_off,
        k_on_per_chain_per_s=k_on_by_chain,
        k_off_per_chain_per_s=k_off_by_chain,
        n_on=n_on,
        n_off=n_off,
        n_on_per_chain=n_on_chain,
        n_off_per_chain=n_off_chain,
        t_on_ns=total_on,
        t_off_ns=total_off,
        t_on_per_chain_ns=t_on_chain,
        t_off_per_chain_ns=t_off_chain,
        fraction_on=fraction_total,
        fraction_on_per_chain=fractions,
        tau_on_ns=_on_off_lifetime_ns(k_off),
        tau_off_ns=_on_off_lifetime_ns(k_on),
        on_dwell_times_ns=dwell_on,
        off_dwell_times_ns=dwell_off,
        on_dwell_times_per_chain_ns=tuple(dwell_on_by_chain),
        off_dwell_times_per_chain_ns=tuple(dwell_off_by_chain),
        on_cutoff_nm=on_nm,
        off_cutoff_nm=off_nm,
        min_on_time_ns=on_min_ns,
        min_off_time_ns=off_min_ns,
        delta_t_ns=dt_ns,
        frame_start=start,
        frame_stop=stop,
        initial_state=state_name,
    )


def jackknife_rate_sem(counts, exposure_ns):
    """Jackknife SEM of pooled transition rate across chains."""
    counts = np.asarray(counts, dtype=float)
    exposure = np.asarray(exposure_ns, dtype=float)

    valid = np.isfinite(exposure) & (exposure > 0)
    counts, exposure = counts[valid], exposure[valid]
    n = len(counts)

    if n < 2:
        return np.nan

    # Leave-one-chain-out rates
    denom = exposure.sum() - exposure
    if np.any(denom <= 0):
        return np.nan

    rates = 1e9 * (counts.sum() - counts) / denom
    return np.sqrt((n - 1) / n * np.sum((rates - rates.mean()) ** 2))


def calculate_distance_msd_1d(
    data: Any,
    delta_t: Any,
    *,
    distance_field: str = "distance_per_chain_nm",
    lags_frames: Optional[Sequence[int]] = None,
    max_lag_frames: int = 200,
    lag_stride: int = 1,
    origin_stride: int = 1,
    frame_start: Optional[int] = None,
    frame_stop: Optional[int] = None,
    time_start: Any = None,
    time_stop: Any = None,
) -> Any:
    """Calculate unconditional 1D distance MSDs from cached distance series.

    Accepts a single distance result from get_values(calc='intrachain'), a
    raw (n_chains,n_frames) array, or a mapping {set_group: distance_result}.
    Dictionary keys and nesting are preserved. Per-chain MSDs are averaged
    equally and SEM is estimated across chains. Numeric times are in ns.
    ``delta_t`` is the actual spacing of the supplied frames, including any
    stride applied when reading the original DCD.

    This does not re-read the trajectory or require a pairsel argument; the
    input must already contain its pairsel-resolved distance time series.
    To supply pairsel directly, use get_values(calc='distmsd1d', pairsel=...).
    """
    dt_ns = _delta_t_to_ns(delta_t)
    if _result_has_field(data, distance_field) or isinstance(data, np.ndarray):
        values = (
            np.asarray(_result_field(data, distance_field), dtype=np.float64)
            if _result_has_field(data, distance_field)
            else np.asarray(data, dtype=np.float64)
        )
        if values.ndim == 1:
            values = values[None, :]
        if values.ndim != 2:
            raise ValueError("distance series must have shape (n_chains,n_frames)")
        start, stop = _resolve_frame_slice(
            values.shape[1],
            frame_start=frame_start,
            frame_stop=frame_stop,
            time_start=time_start,
            time_stop=time_stop,
            delta_t=dt_ns,
        )
        field_values: dict[str, Any] = {distance_field: values[:, start:stop]}
        for field in ("chain_labels", "selection", "atom_labels", "selection_mode", "pbc"):
            if _result_has_field(data, field):
                field_values[field] = _result_field(data, field)
        return distance_msd_1d_from_series(
            field_values,
            dt_ns=dt_ns,
            distance_field=distance_field,
            lags_frames=lags_frames,
            max_lag_frames=max_lag_frames,
            lag_stride=lag_stride,
            origin_stride=origin_stride,
        )
    if isinstance(data, Mapping):
        if not data:
            raise ValueError("data mapping is empty")
        return {
            key: calculate_distance_msd_1d(
                value,
                delta_t,
                distance_field=distance_field,
                lags_frames=lags_frames,
                max_lag_frames=max_lag_frames,
                lag_stride=lag_stride,
                origin_stride=origin_stride,
                frame_start=frame_start,
                frame_stop=frame_stop,
                time_start=time_start,
                time_stop=time_stop,
            )
            for key, value in data.items()
        }
    raise TypeError("data must be a distance series/result or mapping of results")


def calculate_distance_dependent_msd(
    data: Any,
    delta_t: Any,
    *,
    distance_field: str = "distance_per_chain_nm",
    bins: Union[int, Sequence[float], np.ndarray] = 6,
    value_range: Optional[tuple[float, float]] = (0.3, 8.0),
    lags_frames: Optional[Sequence[int]] = None,
    max_lag_frames: int = 200,
    lag_stride: int = 1,
    origin_stride: int = 1,
    frame_start: Optional[int] = None,
    frame_stop: Optional[int] = None,
    time_start: Any = None,
    time_stop: Any = None,
) -> Any:
    """Initial-distance-dependent 1D MSD calculated from existing distance series.

    Operates on the SAME scalar r(t) values as calculate_distance_msd_1d.
    Accepts one intrachain result, an array (chain,frame), or the flat/nested
    mappings returned by get_values(calc='intrachain'). Mapping layout and
    keys are preserved. Numeric time values are in ns; delta_t is the actual
    spacing between adjacent input frames (accounting for any DCD stride).

    bins=6 with value_range=(0.3,8.0) corresponds to the old distmsd window,
    but NOT to the old origin-dependent distance_image='hybrid' coordinates:
    cached PBC distances lose the periodic-image information needed for that.

    Each leaf returns DistMSDPairResult(intra=DistMSDBinnedResult(...),inter=None)
    with the same fields as get_values(calc='distmsd'). Consequently,
    fit_dist_msd_linear_bins(result, mode='intra',...) also works.
    """
    dt_ns = _delta_t_to_ns(delta_t)
    if np.isscalar(bins):
        if isinstance(bins, (bool, np.bool_)):
            raise TypeError("bins must be a positive integer or explicit edges")
        try:
            bins_num = float(bins)
        except (TypeError, ValueError) as exc:
            raise TypeError("bins must be a positive integer or explicit edges") from exc
        if not np.isfinite(bins_num) or bins_num != int(bins_num) or bins_num < 1:
            raise ValueError("bins must be a positive integer")
        if value_range is None or len(value_range) != 2:
            raise ValueError("value_range=(min,max) is required for integer bins")
        rmin, rmax = float(value_range[0]), float(value_range[1])
        if not np.isfinite(rmin) or not np.isfinite(rmax) or rmin < 0 or rmax <= rmin:
            raise ValueError("value_range must be an increasing nonnegative interval")
        edges = None
        n_bins = int(bins_num)
    else:
        edges = np.asarray(bins, dtype=np.float64)
        if edges.ndim != 1 or edges.size < 2:
            raise ValueError("explicit bin edges must have shape (n_bins+1,)")
        if not np.all(np.isfinite(edges)) or edges[0] < 0 or np.any(np.diff(edges) <= 0):
            raise ValueError("bin edges must be nonnegative, finite and increasing")
        rmin, rmax = float(edges[0]), float(edges[-1])
        n_bins = len(edges) - 1

    if _result_has_field(data, distance_field) or isinstance(data, (np.ndarray, list, tuple)):
        values = (
            np.asarray(_result_field(data, distance_field), dtype=np.float64)
            if _result_has_field(data, distance_field)
            else np.asarray(data, dtype=np.float64)
        )
        if values.ndim == 1:
            values = values[None, :]
        if values.ndim != 2:
            raise ValueError("distance series must have shape (n_chains,n_frames)")
        start, stop = _resolve_frame_slice(
            values.shape[1],
            frame_start=frame_start,
            frame_stop=frame_stop,
            time_start=time_start,
            time_stop=time_stop,
            delta_t=dt_ns,
        )
        selected: dict[str, Any] = {distance_field: values[:, start:stop]}
        for field in ("chain_labels", "selection", "atom_labels", "pbc"):
            if _result_has_field(data, field):
                selected[field] = _result_field(data, field)
        return dist_msd_binned_from_series(
            selected,
            dt_ns=dt_ns,
            distance_field=distance_field,
            r_edges_nm=edges,
            r_min_nm=rmin,
            r_max_nm=rmax,
            n_bins=n_bins,
            lags_frames=lags_frames,
            max_lag_frames=max_lag_frames,
            lag_stride=lag_stride,
            origin_stride=origin_stride,
        )
    if isinstance(data, Mapping):
        if not data:
            raise ValueError("data mapping is empty")
        return {
            key: calculate_distance_dependent_msd(
                item,
                delta_t,
                distance_field=distance_field,
                bins=bins,
                value_range=value_range,
                lags_frames=lags_frames,
                max_lag_frames=max_lag_frames,
                lag_stride=lag_stride,
                origin_stride=origin_stride,
                frame_start=frame_start,
                frame_stop=frame_stop,
                time_start=time_start,
                time_stop=time_stop,
            )
            for key, item in data.items()
        }
    raise TypeError("data must contain a distance series or a mapping of results")


# Alternative name emphasizing the scalar, 1D coordinate.
calculate_distance_msd_binned_1d = calculate_distance_dependent_msd


def _time_scale_from_ns(time_unit: str) -> tuple[float, str]:
    """Return the scale from ns and a normalized axis label."""
    normalized = str(time_unit).strip().lower().replace("μ", "u").replace("µ", "u")
    scales = {
        "ps": (1.0e3, "ps"),
        "ns": (1.0, "ns"),
        "us": (1.0e-3, "µs"),
        "ms": (1.0e-6, "ms"),
        "s": (1.0e-9, "s"),
    }
    if normalized not in scales:
        raise ValueError("time_unit must be one of 'ps', 'ns', 'us', 'ms', or 's'")
    return scales[normalized]


def _save_figure_if_requested(
    figure: plt.Figure,
    save_path: Optional[PathLike] = None,
    *,
    save_dpi: Optional[float] = None,
    save_bbox_inches: Optional[str] = "tight",
    save_kwargs: Optional[Mapping[str, Any]] = None,
) -> None:
    """Save *figure* when ``save_path`` is provided.

    Parent directories are created automatically. Additional keyword
    arguments for :meth:`matplotlib.figure.Figure.savefig` can be supplied via
    ``save_kwargs``. ``save_dpi`` and ``save_bbox_inches`` are convenience
    options that populate the corresponding ``savefig`` arguments unless those
    keys are already present in ``save_kwargs``.
    """
    if save_path is None:
        return

    destination = Path(save_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    kwargs: dict[str, Any] = {}
    if save_kwargs is not None:
        kwargs.update(dict(save_kwargs))
    if save_dpi is not None and "dpi" not in kwargs:
        kwargs["dpi"] = save_dpi
    if save_bbox_inches is not None and "bbox_inches" not in kwargs:
        kwargs["bbox_inches"] = save_bbox_inches

    figure.savefig(destination, **kwargs)


def _torsion_residue_series(
    result: Any,
    angle: str,
    resnum: int,
) -> tuple[np.ndarray, tuple[str, ...], str]:
    """Extract one residue's torsion series from each chain containing it.

    Chain residue-number arrays are consulted independently, so different
    sequences and chain lengths are supported. A residue is addressed by its
    PDB residue number, not by zero-based row index.
    """
    if angle not in {"phi", "psi", "omega", "chi1", "chi2"}:
        raise ValueError("torsion data_field must be phi, psi, omega, chi1, or chi2")
    if isinstance(resnum, (bool, np.bool_)) or not isinstance(resnum, (int, np.integer)):
        raise TypeError("resnum must be an integer PDB residue number")

    rows: list[np.ndarray] = []
    labels: list[str] = []
    for label, chain in result.chains.items():
        numbers = np.asarray(chain.residue_numbers)
        matches = np.flatnonzero(numbers == int(resnum))
        if matches.size == 0:
            continue
        if matches.size > 1:
            raise ValueError(
                f"chain {label!r} has repeated residue number {resnum}; "
                "cannot choose one torsion unambiguously"
            )
        angles = np.asarray(getattr(chain, angle), dtype=np.float64)
        if angles.ndim != 2 or angles.shape != (numbers.size, int(result.n_frames)):
            raise ValueError(f"chain {label!r}: invalid {angle} array shape {angles.shape}")
        rows.append(angles[int(matches[0]), :])
        labels.append(str(label))

    if not rows:
        raise ValueError(f"no atomistic chains contain residue {resnum}")
    data = np.stack(rows, axis=0)
    if not np.any(np.isfinite(data)):
        raise ValueError(f"{angle} is undefined for residue {resnum} in every selected chain")

    unit = str(result.unit).strip().lower()
    if unit not in {"radians", "degrees"}:
        raise ValueError(f"unsupported torsion angle unit {result.unit!r}")
    return data, tuple(labels), unit


def plot_timeseries(
    result: Any,
    data_field: str,
    delta_t: Union[float, int, object],
    *,
    resnum: Optional[int] = None,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    title: Optional[str] = None,
    chain_indices: Optional[Sequence[int]] = None,
    chain_labels: Optional[Sequence[str]] = None,
    include_chains: Optional[Sequence[str]] = None,
    exclude_chains: Optional[Sequence[str]] = None,
    strict_chain_selection: bool = True,
    ax: Optional[plt.Axes] = None,
    figsize: tuple[float, float] = (10.0, 4.0),
    time_unit: str = "ns",
    alpha: float = 0.8,
    linewidth: float = 1.0,
    color: Optional[Any] = None,
    colors: Optional[Union[Mapping[Any, Any], Sequence[Any]]] = None,
    show_mean: bool = False,
    mean_linewidth: float = 2.0,
    mean_color: str = "black",
    legend: bool = False,
    ylim: Optional[tuple[float, float]] = None,
    xlim: Optional[tuple[float, float]] = None,
    xticks: Optional[Sequence[float]] = None,
    yticks: Optional[Sequence[float]] = None,
    save_path: Optional[PathLike] = None,
    save_dpi: Optional[float] = None,
    save_bbox_inches: Optional[str] = "tight",
    save_kwargs: Optional[Mapping[str, Any]] = None,
) -> plt.Axes:
    """Plot selected rows of a ``(n_chains, n_frames)`` result field.

    ``time_unit`` controls both conversion of the x values and the default
    x-axis label. Supported units are ``"ps"``, ``"ns"``, ``"us"``, ``"ms"``,
    and ``"s"``; ``"us"`` is displayed as ``µs``.

    ``color`` assigns one color to every selected trace. ``colors`` assigns
    colors individually and is mutually exclusive with ``color``. A ``colors``
    sequence may contain either one entry per selected trace or one entry per
    chain in the full result. A ``colors`` mapping may be keyed by chain label
    or integer chain index.

    ``x_label`` and ``y_label`` override the default axis labels.
    ``xticks`` and ``yticks`` optionally specify exact major tick positions.
    X-axis values use the selected ``time_unit``. If omitted, Matplotlib
    determines tick locations automatically.

    When ``result`` is a PeptideTorsionResult, supply ``resnum`` and an angle
    name (``phi``, ``psi``, ``omega``, ``chi1``, or ``chi2``). This plots the
    corresponding PDB residue number across all chains containing it. For
    torsions, ``show_mean=True`` plots the *circular* mean across chains rather
    than an incorrect arithmetic mean near the -pi/pi (or -180/180) boundary.

    ``include_chains`` and ``exclude_chains`` accept exact physical chain/segment
    IDs such as ["P001", "P003"] and are mutually exclusive. If ``chain_indices``
    is also provided, both filters apply (intersection); ``chain_indices`` refers
    to the plotted row order, after any torsion residue extraction. By default,
    unknown chain names raise an error. Set ``strict_chain_selection=False``
    to ignore unavailable names, e.g. when different chains lack a residue.
    ``chain_labels`` remains a display-label override, not a selection filter.

    When ``save_path`` is provided, the figure is written with
    :meth:`matplotlib.figure.Figure.savefig`. Additional keyword arguments may
    be passed via ``save_kwargs``.
    """
    if include_chains is not None and exclude_chains is not None:
        raise ValueError("include_chains and exclude_chains are mutually exclusive")
    if not isinstance(strict_chain_selection, (bool, np.bool_)):
        raise TypeError("strict_chain_selection must be a boolean")

    is_torsion = hasattr(result, "chains") and hasattr(result, "unit")
    torsion_unit: Optional[str] = None
    physical_chain_labels: Optional[tuple[str, ...]] = None
    if is_torsion:
        if resnum is None:
            raise ValueError("resnum is required to plot a torsion time series")
        data, selected_chain_labels, torsion_unit = _torsion_residue_series(
            result, data_field, resnum
        )
        physical_chain_labels = selected_chain_labels
        if chain_labels is None:
            chain_labels = selected_chain_labels
    else:
        if resnum is not None:
            raise ValueError("resnum is only valid for peptide torsion results")
        data = np.asarray(_result_field(result, data_field), dtype=np.float64)
        if data.ndim != 2:
            raise ValueError(f"{data_field} must have shape (n_chains, n_frames)")
        if hasattr(result, "chain_labels"):
            candidate = tuple(str(label) for label in result.chain_labels)
            if len(candidate) == data.shape[0]:
                physical_chain_labels = candidate

    n_chains, n_frames = data.shape
    if n_chains == 0 or n_frames == 0:
        raise ValueError(f"{data_field} is empty")

    dt_ns = _delta_t_to_ns(delta_t)
    time_scale, time_label = _time_scale_from_ns(time_unit)
    time = dt_ns * time_scale * np.arange(n_frames, dtype=np.float64)

    if chain_indices is None:
        selected = np.arange(n_chains, dtype=np.int64)
    else:
        selected = np.asarray(chain_indices, dtype=np.int64)
        if selected.ndim != 1:
            raise ValueError("chain_indices must be a one-dimensional sequence")
        if selected.size == 0:
            raise ValueError("chain_indices is empty")
        if np.any(selected < 0) or np.any(selected >= n_chains):
            raise ValueError("chain_indices contains an out-of-range index")

    if include_chains is not None or exclude_chains is not None:
        if physical_chain_labels is None:
            raise ValueError(
                "include_chains/exclude_chains requires a result with valid physical "
                "chain_labels (or a peptide-torsion result)"
            )
        if len(set(physical_chain_labels)) != n_chains:
            raise ValueError("result chain_labels must be unique for named chain selection")

        spec = include_chains if include_chains is not None else exclude_chains
        if isinstance(spec, (str, bytes)) or not isinstance(spec, Sequence):
            raise TypeError("include_chains/exclude_chains must be a sequence of segment IDs")
        wanted = tuple(str(label) for label in spec)
        if len(set(wanted)) != len(wanted):
            raise ValueError("chain selection contains duplicate segment IDs")
        missing = sorted(set(wanted) - set(physical_chain_labels))
        if missing and strict_chain_selection:
            raise ValueError(
                f"unknown chain IDs {missing}; available for this property/residue: "
                f"{list(physical_chain_labels)}"
            )

        wanted_set = set(wanted)
        if include_chains is not None:
            selected = np.asarray(
                [i for i in selected if physical_chain_labels[int(i)] in wanted_set],
                dtype=np.int64,
            )
        else:
            selected = np.asarray(
                [i for i in selected if physical_chain_labels[int(i)] not in wanted_set],
                dtype=np.int64,
            )
        if selected.size == 0:
            raise ValueError("chain selection leaves no chains to plot")

    if chain_labels is None and hasattr(result, "chain_labels"):
        candidate_labels = list(getattr(result, "chain_labels"))
        if len(candidate_labels) == n_chains:
            chain_labels = [str(label) for label in candidate_labels]

    if chain_labels is not None:
        labels = [str(label) for label in chain_labels]
        if len(labels) != n_chains:
            raise ValueError(f"chain_labels contains {len(labels)} labels, expected {n_chains}")
    else:
        labels = [f"chain {index}" for index in range(n_chains)]

    if color is not None and colors is not None:
        raise ValueError("color and colors are mutually exclusive")

    selected_colors: list[Any]
    if colors is None:
        selected_colors = [color] * int(selected.size)
    elif isinstance(colors, Mapping):
        selected_colors = []
        for chain_index in selected:
            index = int(chain_index)
            label = labels[index]
            if label in colors:
                selected_colors.append(colors[label])
            elif index in colors:
                selected_colors.append(colors[index])
            else:
                selected_colors.append(None)
    else:
        color_values = list(colors)
        if len(color_values) == int(selected.size):
            selected_colors = color_values
        elif len(color_values) == n_chains:
            selected_colors = [color_values[int(index)] for index in selected]
        else:
            raise ValueError(
                "colors must contain either one entry per selected trace "
                f"({int(selected.size)}) or one entry per chain ({n_chains})"
            )

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    selected_data = data[selected, :]
    for row, chain_index in enumerate(selected):
        ax.plot(
            time,
            selected_data[row],
            linewidth=float(linewidth),
            alpha=float(alpha),
            color=selected_colors[row],
            label=labels[int(chain_index)],
        )

    if show_mean:
        finite = np.isfinite(selected_data)
        counts = np.sum(finite, axis=0)
        valid = counts > 0
        mean = np.full(n_frames, np.nan, dtype=float)
        if is_torsion:
            angles_rad = np.deg2rad(selected_data) if torsion_unit == "degrees" else selected_data
            # Invalid angles contribute neither sin nor cos.
            safe_angles = np.where(finite, angles_rad, 0.0)
            sin_sum = np.sum(np.where(finite, np.sin(safe_angles), 0.0), axis=0)
            cos_sum = np.sum(np.where(finite, np.cos(safe_angles), 0.0), axis=0)
            # The circular mean is undefined when the resultant vector vanishes.
            valid &= np.hypot(sin_sum, cos_sum) > 1.0e-12 * counts
            mean[valid] = np.arctan2(sin_sum[valid], cos_sum[valid])
            if torsion_unit == "degrees":
                mean = np.rad2deg(mean)
        else:
            sums = np.sum(np.where(finite, selected_data, 0.0), axis=0)
            mean[valid] = sums[valid] / counts[valid]
        ax.plot(
            time,
            mean,
            linewidth=float(mean_linewidth),
            color=mean_color,
            label="circular mean" if is_torsion else "mean",
        )

    ax.set_xlabel(x_label if x_label is not None else f"time ({time_label})")
    if is_torsion:
        unit_label = "°" if torsion_unit == "degrees" else "rad"
        default_ylabel = f"{data_field} ({unit_label})"
        default_title = f"{data_field} residue {resnum} vs time"
    else:
        default_ylabel = data_field
        default_title = f"{data_field} vs time"
    ax.set_ylabel(y_label or default_ylabel)
    ax.set_title(title or default_title)

    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)
    if legend:
        ax.legend()

    _save_figure_if_requested(
        ax.figure,
        save_path,
        save_dpi=save_dpi,
        save_bbox_inches=save_bbox_inches,
        save_kwargs=save_kwargs,
    )

    return ax


def plot_errorbar_datasets(
    datasets: Union[
        Sequence[Sequence[float]],
        np.ndarray,
        Mapping[str, Union[Sequence[Sequence[float]], np.ndarray]],
    ],
    *,
    x_col: int = 0,
    y_col: int = 1,
    yerr_col: Optional[int] = 2,
    dataset_label: Optional[str] = None,
    labels: Optional[Union[Mapping[str, str], Sequence[str]]] = None,
    colors: Optional[Union[Mapping[str, Any], Sequence[Any]]] = None,
    markers: Optional[Union[Mapping[str, str], Sequence[str]]] = None,
    connect: bool = False,
    fill_uncertainty: bool = False,
    uncertainty_alpha: float = 0.25,
    linewidth: float = 1.5,
    markersize: float = 6.0,
    capsize: float = 3.0,
    alpha: float = 1.0,
    reference_value: Optional[float] = None,
    reference_values: Optional[Sequence[Union[float, Mapping[str, Any]]]] = None,
    reference_error: Optional[Union[float, Sequence[float], np.ndarray]] = None,
    reference_label: Optional[str] = None,
    reference_color: Optional[Any] = None,
    reference_band_color: Optional[Any] = None,
    reference_linestyle: str = "--",
    reference_linewidth: float = 1.5,
    reference_alpha: float = 0.2,
    vertical_reference_value: Optional[float] = None,
    vertical_reference_label: Optional[str] = None,
    vertical_reference_color: Optional[Any] = None,
    vertical_reference_linestyle: str = "--",
    vertical_reference_linewidth: float = 1.5,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    xlim: Optional[tuple[float, float]] = None,
    ylim: Optional[tuple[float, float]] = None,
    xscale: str = "linear",
    yscale: str = "linear",
    figsize: tuple[float, float] = (6.0, 4.0),
    ax: Optional[plt.Axes] = None,
    legend: bool = True,
    save_path: Optional[PathLike] = None,
    save_dpi: Optional[float] = None,
    save_bbox_inches: Optional[str] = "tight",
    save_kwargs: Optional[Mapping[str, Any]] = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot one or more ``x, y, error`` datasets with error bars.

    Parameters
    ----------
    datasets
        Either one two-dimensional array or a mapping ``label -> array``. By
        default, columns 0, 1, and 2 contain x, y, and symmetric y uncertainty.
        Additional columns, such as sample counts, are ignored. Set
        ``yerr_col=None`` to plot without error bars.
    x_col, y_col, yerr_col
        Column indices containing x, y, and symmetric y uncertainty.
        Uncertainty values must be finite and nonnegative. Rows with nonfinite
        x or y values are omitted; rows with nonfinite uncertainty are also
        omitted when ``yerr_col`` is used.
    fill_uncertainty
        If False, use conventional point error bars. If True, draw a continuous
        line through the mean values with no point markers and use a filled
        ``y - error`` to ``y + error`` band.
    uncertainty_alpha
        Transparency of the filled uncertainty band. Used only when
        ``fill_uncertainty=True``.
    reference_value
        Optional single horizontal reference line (legacy interface).
    reference_values
        Optional additional horizontal reference lines. Supply a sequence of
        numeric y-values, e.g. ``[0.3, 0.6, 0.9]``, or dictionaries for
        individual styles, e.g. ``[{"value": 0.3, "label": "experiment",
        "color": "red", "error": 0.05}]``. Each dictionary supports the keys
        ``value``, ``error`` (scalar or (lower, upper)), ``label``, ``color``,
        ``band_color``, ``linestyle``, ``linewidth``, and ``alpha`` (band opacity).
        The global ``reference_color``, ``reference_band_color``,
        ``reference_linestyle``, ``reference_linewidth``, and ``reference_alpha``
        apply as defaults. This argument can coexist with ``reference_value``.
    reference_error
        Optional uncertainty around ``reference_value``. A scalar gives a
        symmetric band. A two-element sequence gives ``(lower, upper)`` errors.
    reference_color
        Color of the horizontal reference line. The uncertainty band uses the
        same color unless ``reference_band_color`` is supplied.
    reference_band_color
        Optional separate color for the reference uncertainty band.
    reference_alpha
        Transparency of the reference uncertainty band.
    vertical_reference_value
        Optional x-coordinate for a vertical reference line. Must be positive
        when ``xscale="log"``.
    vertical_reference_label, vertical_reference_color
        Optional legend label and line color for the vertical reference.
    vertical_reference_linestyle, vertical_reference_linewidth
        Line style and width for the vertical reference (default dashed, 1.5).
    xscale, yscale
        Axis scales. Each must be ``"linear"`` or ``"log"``. When a log
        scale is requested, nonpositive values on that axis are omitted. For a
        filled uncertainty band on a log y-axis, the band is drawn only where
        ``y - error > 0`` and ``y + error > 0``.

    Returns
    -------
    figure, axes
        The Matplotlib figure and axes containing the plot.

    When ``save_path`` is provided, the figure is written with
    :meth:`matplotlib.figure.Figure.savefig`. Additional keyword arguments may
    be passed via ``save_kwargs``.
    """
    if isinstance(datasets, Mapping):
        if not datasets:
            raise ValueError("datasets is empty")
        dataset_items = [(str(key), value) for key, value in datasets.items()]
    else:
        key = "data" if dataset_label is None else str(dataset_label)
        dataset_items = [(key, datasets)]

    def _column_index(value: Any, *, name: str) -> int:
        if isinstance(value, (bool, np.bool_)):
            raise TypeError(f"{name} must be an integer")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{name} must be an integer") from exc
        if not math.isfinite(numeric) or not numeric.is_integer():
            raise ValueError(f"{name} must be an integer")
        index = int(numeric)
        if index < 0:
            raise ValueError(f"{name} must be >= 0")
        return index

    def _normalize_scale(value: str, *, name: str) -> str:
        if not isinstance(value, str):
            raise TypeError(f"{name} must be 'linear' or 'log'")
        normalized = value.strip().lower()
        aliases = {"linear": "linear", "lin": "linear", "log": "log", "log10": "log"}
        if normalized not in aliases:
            raise ValueError(f"{name} must be 'linear' or 'log'")
        return aliases[normalized]

    x_index = _column_index(x_col, name="x_col")
    y_index = _column_index(y_col, name="y_col")
    error_index = None if yerr_col is None else _column_index(yerr_col, name="yerr_col")
    xscale_name = _normalize_scale(xscale, name="xscale")
    yscale_name = _normalize_scale(yscale, name="yscale")

    if float(linewidth) < 0.0:
        raise ValueError("linewidth must be >= 0")
    if float(markersize) < 0.0:
        raise ValueError("markersize must be >= 0")
    if float(capsize) < 0.0:
        raise ValueError("capsize must be >= 0")
    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("alpha must be between 0 and 1")
    if not 0.0 <= float(uncertainty_alpha) <= 1.0:
        raise ValueError("uncertainty_alpha must be between 0 and 1")
    if bool(fill_uncertainty) and error_index is None:
        raise ValueError("yerr_col is required when fill_uncertainty=True")
    if float(reference_linewidth) < 0.0:
        raise ValueError("reference_linewidth must be >= 0")
    if (
        not math.isfinite(float(vertical_reference_linewidth))
        or float(vertical_reference_linewidth) < 0.0
    ):
        raise ValueError("vertical_reference_linewidth must be finite and >= 0")
    if not 0.0 <= float(reference_alpha) <= 1.0:
        raise ValueError("reference_alpha must be between 0 and 1")

    def _style_value(
        specification: Optional[Union[Mapping[str, Any], Sequence[Any]]],
        key: str,
        index: int,
        default: Any,
        *,
        name: str,
    ) -> Any:
        if specification is None:
            return default
        if isinstance(specification, Mapping):
            return specification.get(key, default)
        if (
            isinstance(specification, Sequence) or isinstance(specification, np.ndarray)
        ) and not isinstance(specification, (str, bytes)):
            return specification[index] if index < len(specification) else default
        raise TypeError(f"{name} must be a mapping, sequence, or None")

    if ax is None:
        figure, ax = plt.subplots(figsize=figsize)
    else:
        figure = ax.figure

    default_markers = ("o", "s", "^", "D", "v", "P", "X", "<", ">")

    for dataset_index, (key, values) in enumerate(dataset_items):
        array = np.asarray(values, dtype=float)
        if array.ndim != 2:
            raise ValueError(
                f"{key!r}: dataset must be a two-dimensional array; " f"got shape {array.shape}"
            )
        if array.shape[0] == 0:
            raise ValueError(f"{key!r}: dataset contains no rows")

        required_columns = [x_index, y_index]
        if error_index is not None:
            required_columns.append(error_index)
        maximum_column = max(required_columns)
        if array.shape[1] <= maximum_column:
            raise ValueError(
                f"{key!r}: dataset has {array.shape[1]} columns, but column "
                f"index {maximum_column} was requested"
            )

        x_values = array[:, x_index]
        y_values = array[:, y_index]
        valid = np.isfinite(x_values) & np.isfinite(y_values)

        if error_index is None:
            error_values = None
        else:
            error_values = array[:, error_index]
            if np.any(np.isfinite(error_values) & (error_values < 0.0)):
                raise ValueError(f"{key!r}: y uncertainties must be nonnegative")
            valid &= np.isfinite(error_values)

        if xscale_name == "log":
            valid &= x_values > 0.0
        if yscale_name == "log":
            valid &= y_values > 0.0

        if not np.any(valid):
            raise ValueError(f"{key!r}: dataset contains no finite plottable rows")

        x_values = x_values[valid]
        y_values = y_values[valid]
        if error_values is not None:
            error_values = error_values[valid]

        label = _style_value(
            labels,
            key,
            dataset_index,
            key if (isinstance(datasets, Mapping) or dataset_label is not None) else None,
            name="labels",
        )
        color = _style_value(colors, key, dataset_index, None, name="colors")
        marker = _style_value(
            markers,
            key,
            dataset_index,
            default_markers[dataset_index % len(default_markers)],
            name="markers",
        )

        if bool(fill_uncertainty):
            assert error_values is not None
            line_kwargs: dict[str, Any] = {
                "linewidth": float(linewidth),
                "alpha": float(alpha),
                "label": label,
            }
            if color is not None:
                line_kwargs["color"] = color
            line = ax.plot(x_values, y_values, **line_kwargs)[0]
            band_color = color if color is not None else line.get_color()
            if yscale_name == "log":
                band_mask = (y_values - error_values) > 0.0
                band_mask &= (y_values + error_values) > 0.0
                if np.any(band_mask):
                    ax.fill_between(
                        x_values[band_mask],
                        (y_values - error_values)[band_mask],
                        (y_values + error_values)[band_mask],
                        color=band_color,
                        alpha=float(uncertainty_alpha),
                        linewidth=0.0,
                    )
            else:
                ax.fill_between(
                    x_values,
                    y_values - error_values,
                    y_values + error_values,
                    color=band_color,
                    alpha=float(uncertainty_alpha),
                    linewidth=0.0,
                )
        else:
            errorbar_kwargs: dict[str, Any] = {
                "x": x_values,
                "y": y_values,
                "yerr": error_values,
                "fmt": marker,
                "linestyle": "-" if connect else "none",
                "linewidth": float(linewidth),
                "markersize": float(markersize),
                "capsize": float(capsize),
                "alpha": float(alpha),
                "label": label,
            }
            if color is not None:
                errorbar_kwargs["color"] = color
            ax.errorbar(**errorbar_kwargs)

    def draw_horizontal_reference(
        value: float,
        *,
        error: Optional[Union[float, Sequence[float], np.ndarray]],
        label: Optional[str],
        color: Optional[Any],
        band_color: Optional[Any],
        linestyle: str,
        line_width: float,
        band_alpha: float,
    ) -> None:
        reference = float(value)
        if not math.isfinite(reference):
            raise ValueError("horizontal reference values must be finite")
        if yscale_name == "log" and reference <= 0.0:
            raise ValueError("horizontal reference values must be > 0 when yscale='log'")
        if not math.isfinite(float(line_width)) or float(line_width) < 0.0:
            raise ValueError("reference linewidth must be finite and >= 0")
        if not math.isfinite(float(band_alpha)) or not 0.0 <= float(band_alpha) <= 1.0:
            raise ValueError("reference alpha must be between 0 and 1")

        line_kwargs: dict[str, Any] = {
            "y": reference,
            "linestyle": linestyle,
            "linewidth": float(line_width),
            "label": label,
        }
        if color is not None:
            line_kwargs["color"] = color
        reference_line = ax.axhline(**line_kwargs)
        shade_color = (
            band_color
            if band_color is not None
            else color if color is not None else reference_line.get_color()
        )

        if error is not None:
            error_array = np.asarray(error, dtype=float)
            if error_array.ndim == 0:
                lower_error = upper_error = float(error_array)
            elif error_array.ndim == 1 and error_array.size == 2:
                lower_error = float(error_array[0])
                upper_error = float(error_array[1])
            else:
                raise ValueError(
                    "reference error must be a nonnegative scalar or "
                    "two-element (lower, upper) sequence"
                )

            if (
                not math.isfinite(lower_error)
                or not math.isfinite(upper_error)
                or lower_error < 0.0
                or upper_error < 0.0
            ):
                raise ValueError("reference error values must be finite and nonnegative")

            lower = reference - lower_error
            upper = reference + upper_error
            if yscale_name != "log" or (lower > 0.0 and upper > 0.0):
                ax.axhspan(
                    lower,
                    upper,
                    color=shade_color,
                    alpha=float(band_alpha),
                    linewidth=0.0,
                )

    # Retain the existing one-reference behavior and its keyword arguments.
    if reference_error is not None and reference_value is None:
        raise ValueError("reference_value is required when reference_error is supplied")
    if reference_value is not None:
        draw_horizontal_reference(
            reference_value,
            error=reference_error,
            label=reference_label,
            color=reference_color,
            band_color=reference_band_color,
            linestyle=reference_linestyle,
            line_width=reference_linewidth,
            band_alpha=reference_alpha,
        )

    if reference_values is not None:
        if isinstance(reference_values, (str, bytes, Mapping)) or np.isscalar(reference_values):
            raise TypeError("reference_values must be a sequence of numbers or dictionaries")
        supported_keys = {
            "value",
            "error",
            "label",
            "color",
            "band_color",
            "linestyle",
            "linewidth",
            "alpha",
        }
        for index, entry in enumerate(reference_values):
            if isinstance(entry, Mapping):
                unknown = set(entry) - supported_keys
                if unknown:
                    raise ValueError(
                        f"reference_values[{index}] has unsupported keys: {sorted(unknown)}"
                    )
                if "value" not in entry:
                    raise ValueError(f"reference_values[{index}] needs a 'value' key")
                values = entry
            else:
                values = {"value": entry}
            draw_horizontal_reference(
                values["value"],
                error=values.get("error"),
                label=values.get("label"),
                color=values.get("color", reference_color),
                band_color=values.get("band_color", reference_band_color),
                linestyle=values.get("linestyle", reference_linestyle),
                line_width=values.get("linewidth", reference_linewidth),
                band_alpha=values.get("alpha", reference_alpha),
            )

    if vertical_reference_value is not None:
        x_reference = float(vertical_reference_value)
        if not math.isfinite(x_reference):
            raise ValueError("vertical_reference_value must be finite")
        if xscale_name == "log" and x_reference <= 0.0:
            raise ValueError("vertical_reference_value must be > 0 when xscale='log'")
        vertical_kwargs: dict[str, Any] = {
            "x": x_reference,
            "linestyle": vertical_reference_linestyle,
            "linewidth": float(vertical_reference_linewidth),
            "label": vertical_reference_label,
        }
        if vertical_reference_color is not None:
            vertical_kwargs["color"] = vertical_reference_color
        ax.axvline(**vertical_kwargs)

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    ax.set_xscale(xscale_name)
    ax.set_yscale(yscale_name)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    if legend:
        handles, legend_labels = ax.get_legend_handles_labels()
        if any(label and not str(label).startswith("_") for label in legend_labels):
            ax.legend()

    _save_figure_if_requested(
        figure,
        save_path,
        save_dpi=save_dpi,
        save_bbox_inches=save_bbox_inches,
        save_kwargs=save_kwargs,
    )

    return figure, ax


def plot_histograms(
    histograms: Mapping[str, HistogramResult],
    *,
    colors: Optional[Union[Mapping[str, Any], Sequence[Any]]] = None,
    labels: Optional[Union[Mapping[str, str], Sequence[str]]] = None,
    alpha: float = 0.35,
    fill: bool = True,
    linewidth: float = 1.5,
    smooth: bool = False,
    smooth_sigma: float = 1.5,
    smooth_linewidth: float = 2.0,
    show_histogram: bool = True,
    title: Optional[str] = None,
    xlabel: str = "Rg (nm)",
    ylabel: Optional[str] = None,
    xlim: Optional[tuple[float, float]] = None,
    ylim: Optional[tuple[float, float]] = None,
    figsize: tuple[float, float] = (6.0, 4.0),
    ax: Optional[plt.Axes] = None,
    legend: bool = True,
    save_path: Optional[PathLike] = None,
    save_dpi: Optional[float] = None,
    save_bbox_inches: Optional[str] = "tight",
    save_kwargs: Optional[Mapping[str, Any]] = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot histograms returned by :func:`calculate_histograms`.

    ``figsize`` controls figure dimensions in inches when ``ax`` is not supplied.
    When ``ax`` is supplied, its existing figure is used unchanged.

    Set ``smooth=True`` to superimpose a continuous smoothed curve on each
    histogram. Bin heights are averaged with a Gaussian kernel of standard
    deviation ``smooth_sigma`` **in bins** (1.5 by default); larger values
    produce smoother curves. A shape-preserving cubic interpolation connects
    the smoothed bin centers without introducing extra peaks or negative values.
    This is a nonparametric visual smoothing operation, *not* a model fit; the
    original :class:`HistogramResult` data are left unchanged. For nonuniform
    bin edges, smoothing is by neighboring bin index rather than physical x
    distance. End bins use reflected boundary conditions.

    ``show_histogram=False`` with ``smooth=True`` plots just the smooth curves.
    The smoothed curve and corresponding histogram use matching colors, with a
    single legend entry per dataset. Requires SciPy only if smoothing is enabled.

    When ``save_path`` is provided, the figure is written with
    :meth:`matplotlib.figure.Figure.savefig`. Additional keyword arguments may
    be passed via ``save_kwargs``.
    """
    if not histograms:
        raise ValueError("histograms is empty")
    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("alpha must be between 0 and 1")
    if not math.isfinite(float(linewidth)) or float(linewidth) < 0.0:
        raise ValueError("linewidth must be finite and >= 0")
    if not math.isfinite(float(smooth_linewidth)) or float(smooth_linewidth) < 0.0:
        raise ValueError("smooth_linewidth must be finite and >= 0")
    if not math.isfinite(float(smooth_sigma)) or float(smooth_sigma) <= 0.0:
        raise ValueError("smooth_sigma must be finite and > 0")
    if not show_histogram and not smooth:
        raise ValueError("enable smooth=True when show_histogram=False")

    if smooth:
        from scipy.interpolate import PchipInterpolator
        from scipy.ndimage import gaussian_filter1d

    keys = list(histograms)

    def style_value(
        specification: Optional[Union[Mapping[str, Any], Sequence[Any]]],
        key: str,
        index: int,
        default: Any = None,
    ) -> Any:
        if specification is None:
            return default
        if isinstance(specification, Mapping):
            return specification.get(key, default)
        if (
            isinstance(specification, Sequence) or isinstance(specification, np.ndarray)
        ) and not isinstance(specification, (str, bytes)):
            return specification[index] if index < len(specification) else default
        raise TypeError("colors and labels must be mappings, sequences, or None")

    if ax is None:
        figure, ax = plt.subplots(figsize=figsize)
    else:
        figure = ax.figure

    density_values = {result.density for result in histograms.values()}
    if len(density_values) > 1 and ylabel is None:
        raise ValueError("density and count histograms require an explicit common ylabel")

    for index, key in enumerate(keys):
        result = histograms[key]
        label = style_value(labels, key, index, str(key))
        color = style_value(colors, key, index)
        if show_histogram:
            stairs_artist = ax.stairs(
                result.values,
                result.edges,
                label=label,
                color=color,
                fill=bool(fill),
                alpha=float(alpha) if fill else 1.0,
                linewidth=float(linewidth),
            )
            # The filled histogram may have reduced alpha; reuse its RGB for a
            # fully opaque overlaid line, including automatic cycle colors.
            curve_color = stairs_artist.get_edgecolor()[:3]
        else:
            curve_color = color

        if smooth:
            centers = np.asarray(result.centers, dtype=float)
            values = np.asarray(result.values, dtype=float)
            if values.ndim != 1 or centers.shape != values.shape or values.size == 0:
                raise ValueError(f"{key!r}: histogram values/edges have incompatible shapes")
            if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(values)):
                raise ValueError(f"{key!r}: smoothing requires finite histogram data")
            if not np.all(np.diff(centers) > 0.0):
                raise ValueError(f"{key!r}: histogram bin centers must be increasing")

            smoothed = gaussian_filter1d(values, sigma=float(smooth_sigma), mode="reflect")
            if centers.size == 1:
                # A single bin cannot support cubic interpolation.
                curve_x = np.asarray(result.edges, dtype=float)
                curve_y = np.full(curve_x.shape, float(smoothed[0]))
            else:
                curve_x = np.linspace(centers[0], centers[-1], max(200, 10 * centers.size))
                curve_y = PchipInterpolator(centers, smoothed)(curve_x)

            ax.plot(
                curve_x,
                curve_y,
                color=curve_color,
                linewidth=float(smooth_linewidth),
                label="_nolegend_" if show_histogram else label,
            )

    ax.set_xlabel(xlabel)
    if ylabel is None:
        is_density = next(iter(histograms.values())).density
        ylabel = "Probability density" if is_density else "Count"
    ax.set_ylabel(ylabel)

    if title is not None:
        ax.set_title(title)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if legend:
        ax.legend()

    _save_figure_if_requested(
        figure,
        save_path,
        save_dpi=save_dpi,
        save_bbox_inches=save_bbox_inches,
        save_kwargs=save_kwargs,
    )

    return figure, ax


def get_group_averages(
    values_by_group: Any,
    group_labels: Union[Sequence[str], Mapping[str, Sequence[str]]],
    *,
    average_over_sets: bool = False,
    frame_start: Optional[int] = None,
    frame_stop: Optional[int] = None,
    time_start: Any = None,
    time_stop: Any = None,
    delta_t: Any = None,
    value_name: str = "Value",
    unit: Optional[str] = None,
    precision: int = 4,
    printout: bool = False,
) -> dict[str, float]:
    """Return finite-value time averages for labeled groups.

    In the default mode, ``values_by_group`` must have shape
    ``(n_groups, n_frames)`` and one time average is returned for every group.

    With ``average_over_sets=True``, set-replicate means are averaged with equal
    weight.  Three input forms are accepted:

    * a mapping ``set_key -> (n_groups, n_frames)``;
    * a three-dimensional array ``(n_sets, n_groups, n_frames)``;
    * a two-dimensional array whose repeated group labels represent replicate
      rows to be averaged.

    For mapping input, ``group_labels`` may be one shared label sequence or a
    mapping ``set_key -> label sequence``.  Missing groups are allowed; their
    mean is calculated from the sets in which they occur.  A group with no
    finite set mean is returned as ``nan``.

    Frame selection is identical to :func:`calculate_histograms`: specify
    either ``frame_start``/``frame_stop`` or ``time_start``/``time_stop``
    together with ``delta_t``. Stop limits are exclusive. Numeric times and
    numeric ``delta_t`` values are interpreted as nanoseconds; OpenMM time
    quantities are also accepted.
    """
    if int(precision) < 0:
        raise ValueError("precision must be >= 0")

    datasets: list[tuple[str, np.ndarray, list[str]]] = []

    if isinstance(values_by_group, Mapping):
        if not average_over_sets:
            raise ValueError("mapping input requires average_over_sets=True")
        set_keys = sorted(
            values_by_group,
            key=lambda key: (_set_sort_key(str(key)), str(key)),
        )
        labels_are_mapping = isinstance(group_labels, Mapping)
        if labels_are_mapping:
            missing = [key for key in set_keys if key not in group_labels]
            if missing:
                formatted = ", ".join(repr(key) for key in missing)
                raise KeyError(f"group_labels does not contain entries for: {formatted}")

        for key in set_keys:
            values = np.asarray(values_by_group[key], dtype=float)
            if values.ndim != 2:
                raise ValueError(
                    f"{key!r}: values must have shape (n_groups, n_frames); " f"got {values.shape}"
                )
            labels_source = group_labels[key] if labels_are_mapping else group_labels
            labels = [str(label) for label in labels_source]
            if len(labels) != values.shape[0]:
                raise ValueError(
                    f"{key!r}: group_labels contains {len(labels)} labels, "
                    f"but values contain {values.shape[0]} groups"
                )
            if len(set(labels)) != len(labels):
                raise ValueError(
                    f"{key!r}: group_labels contains duplicate labels within " "one set"
                )
            datasets.append((str(key), values, labels))

    else:
        values = np.asarray(values_by_group, dtype=float)
        if values.ndim == 2:
            labels = [str(label) for label in group_labels]
            if len(labels) != values.shape[0]:
                raise ValueError(
                    f"group_labels contains {len(labels)} labels, but "
                    f"values_by_group contains {values.shape[0]} groups"
                )
            if not average_over_sets and len(set(labels)) != len(labels):
                raise ValueError("group_labels contains duplicate labels")
            datasets.append(("set1", values, labels))
        elif values.ndim == 3 and average_over_sets:
            labels = [str(label) for label in group_labels]
            if len(labels) != values.shape[1]:
                raise ValueError(
                    f"group_labels contains {len(labels)} labels, but "
                    f"values_by_group contains {values.shape[1]} groups per set"
                )
            if len(set(labels)) != len(labels):
                raise ValueError("group_labels contains duplicate labels")
            for set_index in range(values.shape[0]):
                datasets.append((f"set{set_index + 1}", values[set_index], labels))
        else:
            expected = (
                "(n_groups, n_frames), or with average_over_sets=True, "
                "(n_sets, n_groups, n_frames)"
            )
            raise ValueError(f"values_by_group must have shape {expected}")

    set_means_by_group: dict[str, list[float]] = {}
    group_order: list[str] = []
    frame_slices: list[tuple[str, int, int]] = []

    for set_key, values, labels in datasets:
        n_groups, n_frames = values.shape
        start, stop = _resolve_frame_slice(
            n_frames,
            frame_start=frame_start,
            frame_stop=frame_stop,
            time_start=time_start,
            time_stop=time_stop,
            delta_t=delta_t,
        )
        frame_slices.append((set_key, start, stop))

        selected = values[:, start:stop]
        finite = np.isfinite(selected)
        counts = np.sum(finite, axis=1)
        sums = np.sum(np.where(finite, selected, 0.0), axis=1)
        row_means = np.full(n_groups, np.nan, dtype=float)
        valid = counts > 0
        row_means[valid] = sums[valid] / counts[valid]

        for label, mean in zip(labels, row_means):
            if label not in set_means_by_group:
                set_means_by_group[label] = []
                group_order.append(label)
            set_means_by_group[label].append(float(mean))

    suffix = "" if unit is None else f" {unit}"
    result: dict[str, float] = {}

    if not average_over_sets:
        set_key, start, stop = frame_slices[0]
        if printout:
            print(f"Frames {start}:{stop} ({stop - start} frames)")
            print(f"{'Group':<20s} {value_name}")

        for label in group_order:
            mean = set_means_by_group[label][0]
            if printout:
                print(f"{label:<20s} {mean:.{int(precision)}f}{suffix}")
            result[label] = float(mean)
        return result

    if printout:
        if len(frame_slices) == 1:
            _, start, stop = frame_slices[0]
            print(f"Frames {start}:{stop} ({stop - start} frames)")
        else:
            slice_text = ", ".join(f"{key}={start}:{stop}" for key, start, stop in frame_slices)
            print(f"Frames by set: {slice_text}")
        print(f"{'Group':<20s} {value_name:<14s} {'N sets':>6s}")

    for label in group_order:
        set_values = np.asarray(set_means_by_group[label], dtype=float)
        finite_set_values = set_values[np.isfinite(set_values)]
        if finite_set_values.size:
            mean = float(np.mean(finite_set_values))
        else:
            mean = float("nan")

        if printout:
            print(
                f"{label:<20s} {mean:.{int(precision)}f}{suffix:<8s} "
                f"{finite_set_values.size:>6d}"
            )
        result[label] = mean

    return result


def _looks_like_residue_selection(fragment: str) -> bool:
    """Return whether a selector fragment represents residues rather than atoms."""
    text = str(fragment).strip().lower()
    if text == "all":
        return True
    return bool(re.fullmatch(r"[0-9][0-9:.+\-]*", text))


def _selection_with_atom_name(selection: Any, atom_name: Optional[str]) -> Any:
    """Add an atom selector to chain/residue selections when one is absent.

    Examples with ``atom_name="CA"``::

        "A"               -> "A.CA"
        "A:B:C"           -> "A:B:C.CA"
        "A.19-205"        -> "A.19-205.CA"
        "protein"         -> "protein.CA"
        ["A", "B.19-205"] -> ["A.CA", "B.19-205.CA"]

    Selections that already contain an atom selector are left unchanged.  This
    helper also preserves comma/whitespace-separated terms and the ``;``/``_``
    group separators understood by :class:`StructureSelector`.  Explicit atom
    index groups are returned unchanged.
    """
    if atom_name is None or selection is None:
        return selection

    atom = str(atom_name).strip()
    if not atom:
        raise ValueError("selection_atom_name must be non-empty or None")

    if isinstance(selection, str):
        # Preserve all separators while applying the rewrite term by term.
        tokens = re.split(r"([;_,\s]+)", selection)
        rewritten: list[str] = []

        for token in tokens:
            if not token or re.fullmatch(r"[;_,\s]+", token):
                rewritten.append(token)
                continue

            term = token.strip()
            parts = term.split(".")

            if len(parts) == 1:
                term = f"{term}.{atom}"
            elif len(parts) == 2 and _looks_like_residue_selection(parts[1]):
                term = f"{term}.{atom}"
            # len(parts) >= 3, or a two-part selector ending in an atom
            # name, already contains an atom-level selection.

            rewritten.append(term)

        return "".join(rewritten)

    if (
        isinstance(selection, Sequence)
        and not isinstance(selection, (str, bytes))
        and all(isinstance(item, str) for item in selection)
    ):
        converted = [_selection_with_atom_name(item, atom) for item in selection]
        return tuple(converted) if isinstance(selection, tuple) else converted

    return selection


def _common_trajectory_kwargs(
    *,
    stride: int,
    chunk: int,
    frame_start: int,
    frame_stop: Optional[int],
    box_nm: Optional[Sequence[float]],
) -> dict[str, Any]:
    if int(stride) <= 0:
        raise ValueError("stride must be >= 1")
    if int(chunk) <= 0:
        raise ValueError("chunk must be >= 1")
    if int(frame_start) < 0:
        raise ValueError("frame_start must be >= 0")

    return {
        "stride": int(stride),
        "chunk": int(chunk),
        "frame_start": int(frame_start),
        "frame_stop": frame_stop,
        "box_nm": box_nm,
    }


def _normalize_distance_axes(distance_axes: str) -> str:
    """Return a canonical Cartesian-axis selection for center distances."""
    if not isinstance(distance_axes, str):
        raise TypeError("distance_axes must be a string")

    text = distance_axes.strip().lower().replace(" ", "")
    if text == "3d":
        text = "xyz"

    if not text or any(axis not in "xyz" for axis in text):
        raise ValueError(
            "distance_axes must be one of 'x', 'y', 'z', 'xy', 'xz', " "'yz', or 'xyz'"
        )
    if len(set(text)) != len(text):
        raise ValueError("distance_axes must not contain repeated axes")

    canonical = "".join(axis for axis in "xyz" if axis in text)
    if canonical not in {"x", "y", "z", "xy", "xz", "yz", "xyz"}:
        raise ValueError(
            "distance_axes must be one of 'x', 'y', 'z', 'xy', 'xz', " "'yz', or 'xyz'"
        )
    return canonical


def _normalize_reference_image_mode(reference_image_mode: str) -> str:
    """Return ``as_is`` or ``cluster`` for reference-chain placement."""
    if not isinstance(reference_image_mode, str):
        raise TypeError("reference_image_mode must be a string")

    mode = reference_image_mode.strip().lower().replace("-", "_")
    aliases = {
        "asis": "as_is",
        "stored": "as_is",
        "direct": "as_is",
        "whole": "as_is",
        "common_cluster": "cluster",
        "periodic_cluster": "cluster",
        "wrapped": "cluster",
        "periodic": "cluster",
    }
    mode = aliases.get(mode, mode)
    if mode not in {"as_is", "cluster"}:
        raise ValueError("reference_image_mode must be 'as_is' or 'cluster'")
    return mode


def _is_center_distance_calculation(calc: str) -> bool:
    """Return whether *calc* selects the reference-center distance analysis."""
    key = str(calc).strip().lower().replace("-", "_")
    return key in {
        "centerdist",
        "center_distance",
        "reference_center_distance",
    }


def _is_density_profile_calculation(calc: str) -> bool:
    """Return whether *calc* selects the mass-density profile analysis."""
    key = str(calc).strip().lower().replace("-", "_")
    return key in {
        "densityprofile",
        "density_profile",
        "mass_density",
        "mass_density_profile",
        "massdensity",
        "density",
    }


def _selected_peptide_torsions(
    result: PeptideTorsionResult,
    pdb_file: PathLike,
    selection: Any,
) -> PeptideTorsionResult:
    """Keep all residues of each atomistic chain touched by a group selection.

    mdsim.peptide_torsions_from_dcd currently analyzes every atomistic chain;
    notebook ``segs`` values identify which physical chains belong to a group.
    A residue- or atom-restricted selector is used to *identify chains*, not to
    remove other residues from those chains.
    """
    template = PDBReader().read(pdb_file)
    model = template.model

    if isinstance(selection, str) or (
        isinstance(selection, Sequence)
        and not isinstance(selection, (str, bytes))
        and all(isinstance(item, str) for item in selection)
    ):
        selected_atoms = set(StructureSelector(selection).atom_indices(template))
    elif isinstance(selection, Sequence) and not isinstance(selection, (str, bytes)):
        # As elsewhere in this module, explicit groups of template atom indices
        # are accepted in addition to StructureSelector strings.
        selected_atoms = {int(index) for group in selection for index in group}
    else:
        raise TypeError("torsion chain selection must be a selector or sequence of selectors")

    if not selected_atoms:
        raise ValueError(f"torsion selection {selection!r} selected no atoms")

    global_index = {id(atom): i for i, atom in enumerate(model.atoms)}
    chains = {}
    for label, chain_result in result.chains.items():
        physical_chain = model.chain[label]
        if any(
            global_index[id(atom)] in selected_atoms
            for residue in physical_chain.residues
            for atom in residue.atoms
        ):
            chains[label] = chain_result

    # Coarse-grained-only groups are legitimate in mixed-resolution systems.
    # Retain their dataset key with n_chains=0 instead of failing the entire call.
    return replace(result, chains=chains)


def calculate(
    calc: str,
    pdb: PathLike,
    dcd: PathLike,
    selection: Any,
    *,
    selection2: Any = None,
    refsel: Any = None,
    pairsel: Any = None,
    option: Optional[str] = None,
    include_hydrogens: bool = False,
    sidechain_only: bool = False,
    workers: int = 1,
    query_site_chunk: int = 4096,
    stride: int = 1,
    chunk: int = 200,
    frame_start: int = 0,
    frame_stop: Optional[int] = None,
    time_start: Any = None,
    time_stop: Any = None,
    delta_t: Any = None,
    bins: Union[int, Sequence[float], np.ndarray] = 60,
    value_range: Optional[tuple[float, float]] = None,
    box_nm: Optional[Sequence[float]] = None,
    mode: str = "cog",
    cutoff_nm: float = 0.8,
    pbc: bool = True,
    diffusion_mode: str = "both",
    distance_image: str = "hybrid",
    lags_frames: Optional[Sequence[int]] = None,
    max_lag_frames: int = 200,
    lag_stride: int = 1,
    origin_stride: int = 1,
    inter_targets_per_ref: Optional[int] = None,
    random_seed: int = 0,
    backend: str = "serial",
    n_jobs: int = 0,
    mp_start_method: Optional[str] = None,
    min_intra_sequence_separation: int = 2,
    pair_block_size: int = 8,
    origin_chunk: int = 16,
    reference_chunk: int = 2,
    map_work_dtype: str = "float32",
    map_work_dir: Optional[PathLike] = None,
    exclude_reference: bool = True,
    reference_label: str = "reference_rlp",
    distance_axes: str = "xyz",
    reference_image_mode: str = "as_is",
    selection_atom_name: Optional[str] = "CA",
    residue_site: Optional[str] = None,
    remove_reference_motion: bool = False,
    unwrap_within_frame: bool = False,
    degrees: bool = False,
    check_peptide_bonds: bool = True,
    peptide_bond_cutoff_nm: float = 0.22,
    torsion_dtype: Any = "float32",
) -> Any:
    """Run one of the common notebook analyses.

    Supported values for ``calc`` are ``rgyr``, ``contacts``, ``centerdist``,
    ``densityprofile``, ``intrachain``, ``interchain``, ``mincontact``,
    ``distmsd``, ``distmsdmap``, ``contactmap``, and ``peptide-torsions``.
    A few descriptive aliases are accepted.

    For ``rgyr``, ``contacts``, and ``centerdist``, chain/residue selections
    are restricted to ``selection_atom_name`` before being passed to mdsim.
    The default is ``"CA"``. Set ``selection_atom_name=None`` to leave supplied
    selections unchanged. Explicit atom selections are never overwritten.

    For ``centerdist``, ``distance_axes`` selects a one-dimensional axis
    (``"x"``, ``"y"``, or ``"z"``), a two-dimensional plane (``"xy"``,
    ``"xz"``, or ``"yz"``), or the default full 3D distance (``"xyz"``).
    All selected chains are assumed to be whole. ``reference_image_mode``
    controls only how different reference-chain centers are placed across PBC:
    ``"as_is"`` preserves stored images, while ``"cluster"`` assembles one
    connected periodic cluster before calculating its center.

    For ``residuedensity``, ``residue_site='CA'`` samples C-alpha atoms,
    ``residue_site='COG'`` samples each residue's center of geometry, and
    ``residue_site='COM'`` samples each residue's center of mass. Omitting
    ``residue_site`` retains the original ``selection_atom_name`` atom (CA by
    default). The reference center remains independently controlled by
    ``mode`` and the reference selection uses ``selection_atom_name``.

    For ``densityprofile``, the analyzed ``selection`` retains all selected
    atoms and their masses. Only ``refsel`` is restricted to
    ``selection_atom_name`` (CA by default), preserving the same reference-center
    definition used by ``centerdist``. One-dimensional profiles are signed;
    two- and three-dimensional profiles are radial.

    For ``intrachain``, ``interchain``, and ``mincontact``, each side of
    ``pairsel`` is a site and the reported distance is the minimum over all atom
    pairs between the two sites. ``pairsel="39.CE2,69.SG"`` is the one-atom
    special case. Multiple explicit atoms are joined with ``+``, for example
    ``pairsel="39.CE2+CD2,69.SG+CB"``. A structured equivalent such as
    ``pairsel=(["39.CE2", "CD2"], ["69.SG", "CB"])`` is also accepted.
    Two bare residue numbers such as ``pairsel="39,69"`` compare all retained
    atoms in the two residues. Residue mode uses heavy atoms by default;
    ``include_hydrogens=True`` includes hydrogen atoms, and
    ``sidechain_only=True`` excludes CA, N, C, O, HN, HA, HA1, HA2, OT1, OT2,
    OXT, HT1, HT2, HT3, and H.

    For ``mincontact``, every selected reference chain is reduced over the
    permitted target chains. The default includes the same chain and all other
    chains. Set ``option="intraonly"`` to retain only the same-chain pair or
    ``option="interonly"`` to minimize only over other chains.

    For ``distmsd1d``, ``selection`` identifies chains and ``pairsel`` uses
    the same 1-atom, multi-atom minimum-distance, or residue-pair syntax as
    ``intrachain``. Its unconditional 1D MSD does not require bins or value_range.
    For this scalar minimum-distance coordinate, ``pbc`` controls per-frame
    minimum-image distances; ``distance_image`` does not apply.

    For ``distmsd``, ``selection`` identifies the chains and ``pairsel`` contains
    two ordered StructureSelector atom selectors, exactly like the pair-distance
    analyses. The result contains intrachain and/or interchain distance-coordinate
    MSDs binned by the initial pair distance. Numeric ``delta_t`` is in ns and is
    interpreted as the spacing of stored trajectory frames; ``stride`` is applied
    automatically to obtain the effective lag-time spacing. ``bins``/``value_range``
    define the initial-distance bins.

    For ``distmsdmap``, ``selection`` identifies the full homologous-chain pool
    and ``selection_atom_name`` identifies one site atom per retained residue (CA
    by default). Optional ``refsel`` restricts only the chain replicates averaged
    into the map: intrachain MSDs are averaged over those physical chains, while
    interchain MSDs use those chains as references but retain every chain in
    ``selection`` as a possible target (excluding self). Thus ``refsel`` can apply
    a chain-property filter without unnecessarily restricting neighboring chains.
    ``value_range=(r_min, r_max)`` independently selects the single initial-distance
    window; ``value_range=None`` accepts all initial distances. Intrachain values
    use the lower triangle and obey ``min_intra_sequence_separation``; interchain
    values use the upper triangle including the diagonal. Coordinates are staged
    in a temporary disk-backed memmap so RAM does not scale with the full selected
    trajectory. ``backend='process'`` parallelizes residue-pair blocks.

    For ``peptide-torsions``, mdsim calculates phi, psi, omega, chi1, and chi2
    for all atomistic chains. The supplied ``selection`` then restricts the
    physical chains kept in the result, retaining every residue of each chain.
    The result has a ``chains`` mapping (chain label -> per-residue arrays).
    ``degrees``, ``check_peptide_bonds``, ``peptide_bond_cutoff_nm``, and
    ``torsion_dtype`` are forwarded to mdsim. ``pbc`` controls minimum-imaging
    of individual bond vectors.

    For ``contactmap``, the function caches compact sparse per-frame CA contact events
    up to ``cutoff_nm``. ``query_site_chunk`` limits the number of query CA sites
    processed at once and therefore controls peak neighbor-search memory.
    ``selection`` defines query chains and optional
    ``selection2`` defines partner chains; omitting ``selection2`` gives a
    homotypic map. Time/property filtering and final map normalization are
    performed later by :func:`calculate_contact_maps`.
    """
    calc_key = str(calc).strip().lower().replace("-", "_")
    aliases = {
        "rg": "rgyr",
        "radius_of_gyration": "rgyr",
        "contact": "contacts",
        "center_distance": "centerdist",
        "reference_center_distance": "centerdist",
        "density": "densityprofile",
        "residue_density": "residuedensity",
        "residue_density_profile": "residuedensity",
        "density_profile": "densityprofile",
        "mass_density": "densityprofile",
        "mass_density_profile": "densityprofile",
        "massdensity": "densityprofile",
        "intra": "intrachain",
        "inter": "interchain",
        "min_contact": "mincontact",
        "minimum_contact": "mincontact",
        "minimum_distance": "mincontact",
        "contact_map": "contactmap",
        "contactmap_raw": "contactmap",
        "ca_contact_map": "contactmap",
        "ca_contacts": "contactmap",
        "dist_msd": "distmsd",
        "distance_msd": "distmsd",
        "distance_diffusion": "distmsd",
        "distance_dependent_diffusion": "distmsd",
        "distdiff": "distmsd",
        "dist_msd_map": "distmsdmap",
        "distance_msd_map": "distmsdmap",
        "distance_diffusion_map": "distmsdmap",
        "distance_dependent_diffusion_map": "distmsdmap",
        "distdiffmap": "distmsdmap",
        "diffusionmap": "distmsdmap",
        "com_trajectory": "comtraj",
        "chain_com": "comtraj",
        "translational_trajectory": "comtraj",
        "distance_msd_1d": "distmsd1d",
        "unconditional_msd": "distmsd1d",
        "dist_msd_1d": "distmsd1d",
        "peptide_torsions": "peptide_torsions",
        "peptidetorsions": "peptide_torsions",
        "torsions": "peptide_torsions",
        "torsion": "peptide_torsions",
    }
    calc_key = aliases.get(calc_key, calc_key)

    common = _common_trajectory_kwargs(
        stride=stride,
        chunk=chunk,
        frame_start=frame_start,
        frame_stop=frame_stop,
        box_nm=box_nm,
    )
    dcd_files = [Path(dcd)]

    analysis_selection = selection
    partner_selection = selection2
    reference_selection = refsel

    if calc_key in {"rgyr", "contacts", "centerdist"}:
        analysis_selection = _selection_with_atom_name(
            selection,
            selection_atom_name,
        )

    if calc_key == "contacts" and selection2 is not None:
        partner_selection = _selection_with_atom_name(
            selection2,
            selection_atom_name,
        )

    if (
        calc_key in {"centerdist", "densityprofile", "residuedensity", "comtraj"}
        and refsel is not None
    ):
        reference_selection = _selection_with_atom_name(
            refsel,
            selection_atom_name,
        )

    if calc_key == "peptide_torsions":
        # Time limits are interpreted on the strided trajectory frame grid.
        uses_time = time_start is not None or time_stop is not None
        if uses_time and (int(frame_start) != 0 or frame_stop is not None):
            raise ValueError("use either time_start/time_stop or frame_start/frame_stop")
        first, last = int(frame_start), frame_stop
        if uses_time:
            if delta_t is None:
                raise ValueError("delta_t is required for time-based torsion selection")
            dt_ns = _delta_t_to_ns(delta_t) * int(stride)
            t0 = 0.0 if time_start is None else _time_value_to_ns(time_start, name="time_start")
            t1 = None if time_stop is None else _time_value_to_ns(time_stop, name="time_stop")
            if t0 < 0.0 or (t1 is not None and t1 <= t0):
                raise ValueError("time limits must satisfy 0 <= time_start < time_stop")
            first = max(0, int(math.ceil(t0 / dt_ns - 1.0e-12)))
            last = None if t1 is None else max(0, int(math.ceil(t1 / dt_ns - 1.0e-12)))
            if last is not None and last <= first:
                raise ValueError("time window selects no torsion frames")

        all_torsions = peptide_torsions_from_dcd(
            pdb_file=Path(pdb),
            dcd_files=dcd_files,
            degrees=bool(degrees),
            pbc=bool(pbc),
            box_nm=box_nm,
            check_peptide_bonds=bool(check_peptide_bonds),
            peptide_bond_cutoff_nm=float(peptide_bond_cutoff_nm),
            stride=int(stride),
            chunk=int(chunk),
            frame_start=first,
            frame_stop=last,
            dtype=torsion_dtype,
        )
        return _selected_peptide_torsions(all_torsions, pdb, selection)

    if calc_key == "comtraj":
        if delta_t is None:
            raise ValueError("delta_t (stored-frame spacing in ns) is required for calc='comtraj'")
        stored_dt_ns = _delta_t_to_ns(delta_t)
        effective_dt_ns = stored_dt_ns * int(stride)
        uses_time = time_start is not None or time_stop is not None
        if uses_time and (int(frame_start) != 0 or frame_stop is not None):
            raise ValueError("use either time_start/time_stop or frame_start/frame_stop")
        first, last = int(frame_start), frame_stop
        if uses_time:
            t0 = 0.0 if time_start is None else _time_value_to_ns(time_start, name="time_start")
            t1 = None if time_stop is None else _time_value_to_ns(time_stop, name="time_stop")
            if t0 < 0 or (t1 is not None and t1 <= t0):
                raise ValueError("invalid time window")
            # stop exclusive, same convention as centerdist
            first = max(0, int(math.ceil(t0 / effective_dt_ns - 1e-12)))
            last = None if t1 is None else max(0, int(math.ceil(t1 / effective_dt_ns - 1e-12)))
        # ``selection`` is not CA-restricted: use entire chains for true COMs.
        # ``refsel`` follows centerdist's optional selection_atom_name convention.
        return chain_com_trajectories_from_dcd(
            pdb_file=Path(pdb),
            dcd_files=dcd_files,
            selection=selection,
            reference_segments=reference_selection,
            mode=mode,
            remove_reference_motion=remove_reference_motion,
            reference_image_mode=reference_image_mode,
            dt_ns=effective_dt_ns,
            box_nm=box_nm,
            stride=int(stride),
            chunk=int(chunk),
            frame_start=first,
            frame_stop=last,
            unwrap_within_frame=unwrap_within_frame,
        )

    if calc_key == "distmsd1d":
        if pairsel is None:
            raise ValueError("pairsel is required for calc='distmsd1d'")
        if delta_t is None:
            raise ValueError("delta_t is required for calc='distmsd1d'")
        if diffusion_mode not in {"intra", "intrachain", "both"}:
            raise ValueError("calc='distmsd1d' currently supports intrachain pairs only")
        uses_time = time_start is not None or time_stop is not None
        uses_frames = int(frame_start) != 0 or frame_stop is not None
        if uses_time and uses_frames:
            raise ValueError(
                "for calc='distmsd1d', specify either frame_start/frame_stop "
                "or time_start/time_stop, not both"
            )
        dt_effective_ns = _delta_t_to_ns(delta_t) * int(stride)
        first_frame, last_frame = int(frame_start), frame_stop
        if uses_time:
            start_ns = (
                0.0 if time_start is None else _time_value_to_ns(time_start, name="time_start")
            )
            stop_ns = None if time_stop is None else _time_value_to_ns(time_stop, name="time_stop")
            if start_ns < 0 or (stop_ns is not None and stop_ns <= start_ns):
                raise ValueError("time limits must satisfy 0 <= time_start < time_stop")

            # ceil with tolerance avoids off-by-one at exact frame boundaries.
            def to_frame(t_ns: float) -> int:
                ratio = float(t_ns) / dt_effective_ns
                nearest = round(ratio)
                if math.isclose(ratio, nearest, rel_tol=1e-12, abs_tol=1e-12):
                    return max(0, int(nearest))
                return max(0, int(math.ceil(ratio)))

            first_frame = to_frame(start_ns)
            last_frame = None if stop_ns is None else to_frame(stop_ns)
            if last_frame is not None and last_frame <= first_frame:
                raise ValueError("time range selects no frames")
        # pbc controls the per-frame minimum image, as in calc='intrachain'.
        # distance_image is intentionally not applied to a scalar minimum
        # distance time series; hybrid is origin-dependent and not a series.
        return distance_msd_1d_from_dcd(
            pdb_file=Path(pdb),
            dcd_files=dcd_files,
            chains=selection,
            selection=pairsel,
            dt_ns=dt_effective_ns,
            include_hydrogens=include_hydrogens,
            sidechain_only=sidechain_only,
            pbc=pbc,
            lags_frames=lags_frames,
            max_lag_frames=int(max_lag_frames),
            lag_stride=int(lag_stride),
            origin_stride=int(origin_stride),
            stride=int(stride),
            chunk=int(chunk),
            frame_start=first_frame,
            frame_stop=last_frame,
            box_nm=box_nm,
        )

    if calc_key == "distmsdmap":
        if delta_t is None:
            raise ValueError("delta_t is required for calc='distmsdmap'")
        if selection_atom_name is None or not str(selection_atom_name).strip():
            raise ValueError("selection_atom_name must be non-empty for calc='distmsdmap'")

        uses_time = time_start is not None or time_stop is not None
        uses_frames = int(frame_start) != 0 or frame_stop is not None
        if uses_time and uses_frames:
            raise ValueError(
                "for calc='distmsdmap', specify either frame_start/frame_stop or "
                "time_start/time_stop, not both"
            )

        stored_dt_ns = _delta_t_to_ns(delta_t)
        effective_dt_ns = stored_dt_ns * float(stride)
        map_frame_start = int(frame_start)
        map_frame_stop = frame_stop

        if uses_time:
            start_ns = (
                0.0 if time_start is None else _time_value_to_ns(time_start, name="time_start")
            )
            stop_ns = None if time_stop is None else _time_value_to_ns(time_stop, name="time_stop")
            if start_ns < 0.0:
                raise ValueError("time_start must be >= 0")
            if stop_ns is not None and stop_ns <= start_ns:
                raise ValueError("time_stop must be greater than time_start")

            def _map_time_to_left_frame(time_ns: float) -> int:
                ratio = float(time_ns) / float(effective_dt_ns)
                nearest = round(ratio)
                if math.isclose(ratio, nearest, rel_tol=1.0e-12, abs_tol=1.0e-12):
                    return max(0, int(nearest))
                return max(0, int(math.floor(ratio)) + 1)

            map_frame_start = _map_time_to_left_frame(start_ns)
            map_frame_stop = None if stop_ns is None else _map_time_to_left_frame(stop_ns)
            if map_frame_stop is not None and map_frame_stop <= map_frame_start:
                raise ValueError("the requested time range selects no strided frames")

        if value_range is None:
            distance_min_nm = None
            distance_max_nm = None
        else:
            if len(value_range) != 2:
                raise ValueError("value_range must be (distance_min_nm, distance_max_nm) or None")
            distance_min_nm = float(value_range[0])
            distance_max_nm = float(value_range[1])
            if (
                not math.isfinite(distance_min_nm)
                or not math.isfinite(distance_max_nm)
                or distance_min_nm < 0.0
                or distance_max_nm <= distance_min_nm
            ):
                raise ValueError("value_range must be a finite increasing nonnegative pair")

        return dist_msd_map_from_dcd(
            pdb_file=Path(pdb),
            dcd_files=dcd_files,
            chains=selection,
            reference_chains=refsel,
            atom_name=str(selection_atom_name),
            mode=diffusion_mode,
            distance_image=distance_image,
            dt_ns=effective_dt_ns,
            distance_min_nm=distance_min_nm,
            distance_max_nm=distance_max_nm,
            min_intra_sequence_separation=int(min_intra_sequence_separation),
            lags_frames=lags_frames,
            max_lag_frames=int(max_lag_frames),
            lag_stride=int(lag_stride),
            origin_stride=int(origin_stride),
            inter_targets_per_ref=inter_targets_per_ref,
            random_seed=int(random_seed),
            backend=backend,
            n_jobs=int(n_jobs),
            mp_start_method=mp_start_method,
            pair_block_size=int(pair_block_size),
            origin_chunk=int(origin_chunk),
            reference_chunk=int(reference_chunk),
            work_dtype=str(map_work_dtype),
            work_dir=map_work_dir,
            stride=int(stride),
            chunk=int(chunk),
            frame_start=map_frame_start,
            frame_stop=map_frame_stop,
            box_nm=box_nm,
        )

    if calc_key == "distmsd":
        if pairsel is None:
            raise ValueError("pairsel is required for calc='distmsd'")
        if delta_t is None:
            raise ValueError("delta_t is required for calc='distmsd'")

        uses_time = time_start is not None or time_stop is not None
        uses_frames = int(frame_start) != 0 or frame_stop is not None
        if uses_time and uses_frames:
            raise ValueError(
                "for calc='distmsd', specify either frame_start/frame_stop or "
                "time_start/time_stop, not both"
            )

        stored_dt_ns = _delta_t_to_ns(delta_t)
        effective_dt_ns = stored_dt_ns * float(stride)

        dist_frame_start = int(frame_start)
        dist_frame_stop = frame_stop
        if uses_time:
            start_ns = (
                0.0 if time_start is None else _time_value_to_ns(time_start, name="time_start")
            )
            stop_ns = None if time_stop is None else _time_value_to_ns(time_stop, name="time_stop")
            if start_ns < 0.0:
                raise ValueError("time_start must be >= 0")
            if stop_ns is not None and stop_ns <= start_ns:
                raise ValueError("time_stop must be greater than time_start")

            def _time_to_left_frame(time_ns: float) -> int:
                ratio = float(time_ns) / float(effective_dt_ns)
                nearest = round(ratio)
                if math.isclose(ratio, nearest, rel_tol=1.0e-12, abs_tol=1.0e-12):
                    return max(0, int(nearest))
                return max(0, int(math.floor(ratio)) + 1)

            dist_frame_start = _time_to_left_frame(start_ns)
            dist_frame_stop = None if stop_ns is None else _time_to_left_frame(stop_ns)
            if dist_frame_stop is not None and dist_frame_stop <= dist_frame_start:
                raise ValueError("the requested time range selects no strided frames")

        if np.isscalar(bins):
            if isinstance(bins, (bool, np.bool_)):
                raise TypeError("bins must be an integer or explicit distance-bin edges")
            bins_float = float(bins)
            if not math.isfinite(bins_float) or not bins_float.is_integer() or bins_float < 1:
                raise ValueError("bins must be an integer >= 1")
            if value_range is None or len(value_range) != 2:
                raise ValueError(
                    "calc='distmsd' requires value_range=(r_min_nm, r_max_nm) "
                    "when bins is an integer"
                )
            r_min_nm = float(value_range[0])
            r_max_nm = float(value_range[1])
            if not math.isfinite(r_min_nm) or not math.isfinite(r_max_nm) or r_max_nm <= r_min_nm:
                raise ValueError("value_range must be a finite increasing pair")
            r_edges_nm = None
            n_dist_bins = int(bins_float)
        else:
            r_edges_nm = np.asarray(bins, dtype=float)
            if r_edges_nm.ndim != 1 or r_edges_nm.size < 2:
                raise ValueError("explicit distance-bin edges must be a 1D array with >=2 values")
            if not np.all(np.isfinite(r_edges_nm)) or not np.all(np.diff(r_edges_nm) > 0.0):
                raise ValueError(
                    "explicit distance-bin edges must be finite and strictly increasing"
                )
            r_min_nm = float(r_edges_nm[0])
            r_max_nm = None
            n_dist_bins = int(r_edges_nm.size - 1)

        return dist_msd_binned_from_dcd(
            pdb_file=Path(pdb),
            dcd_files=dcd_files,
            chains=selection,
            selection=pairsel,
            include_hydrogens=bool(include_hydrogens),
            sidechain_only=bool(sidechain_only),
            mode=diffusion_mode,
            distance_image=distance_image,
            dt_ns=effective_dt_ns,
            r_edges_nm=r_edges_nm,
            r_min_nm=r_min_nm,
            r_max_nm=r_max_nm,
            n_bins=n_dist_bins,
            lags_frames=lags_frames,
            max_lag_frames=int(max_lag_frames),
            lag_stride=int(lag_stride),
            origin_stride=int(origin_stride),
            inter_targets_per_ref=inter_targets_per_ref,
            random_seed=int(random_seed),
            backend=backend,
            n_jobs=int(n_jobs),
            mp_start_method=mp_start_method,
            stride=int(stride),
            chunk=int(chunk),
            frame_start=dist_frame_start,
            frame_stop=dist_frame_stop,
            box_nm=box_nm,
        )

    if calc_key == "contactmap":
        return ca_contact_trajectory_from_dcd(
            pdb_file=Path(pdb),
            dcd_files=dcd_files,
            query_chains=selection,
            partner_chains=selection2,
            atom_name="CA" if selection_atom_name is None else selection_atom_name,
            max_cutoff_nm=float(cutoff_nm),
            pbc=bool(pbc),
            workers=int(workers),
            query_site_chunk=int(query_site_chunk),
            **common,
        )

    if calc_key == "rgyr":
        return rg_from_dcd(
            pdb_file=Path(pdb),
            dcd_files=dcd_files,
            selection=analysis_selection,
            mode=mode,
            **common,
        )

    if calc_key == "contacts":
        kwargs: dict[str, Any] = {
            "pdb_file": Path(pdb),
            "dcd_files": dcd_files,
            "query_selection": analysis_selection,
            "cutoff_nm": float(cutoff_nm),
            "workers": int(workers),
            **common,
        }
        # Preserve the underlying mdsim default when no partner selection is
        # supplied instead of explicitly passing None to StructureSelector.
        if partner_selection is not None:
            kwargs["partner_selection"] = partner_selection
        return contacts_from_dcd(**kwargs)

    if calc_key == "centerdist":
        if refsel is None:
            raise ValueError("refsel is required for calc='centerdist'")
        return reference_center_distances_from_dcd(
            pdb_file=Path(pdb),
            dcd_files=dcd_files,
            reference_label=reference_label,
            reference_segments=reference_selection,
            query_segments=analysis_selection,
            mode=mode,
            distance_axes=_normalize_distance_axes(distance_axes),
            reference_image_mode=_normalize_reference_image_mode(reference_image_mode),
            **common,
        )

    if calc_key == "densityprofile":
        if refsel is None:
            raise ValueError("refsel is required for calc='densityprofile'")
        return mass_density_profile_from_dcd(
            pdb_file=Path(pdb),
            dcd_files=dcd_files,
            reference_label=reference_label,
            reference_segments=reference_selection,
            selection=analysis_selection,
            mode=mode,
            profile_axes=_normalize_distance_axes(distance_axes),
            reference_image_mode=_normalize_reference_image_mode(reference_image_mode),
            bins=bins,
            value_range=value_range,
            time_start=time_start,
            time_stop=time_stop,
            delta_t=delta_t,
            **common,
        )

    if calc_key == "residuedensity":
        if _normalize_distance_axes(distance_axes) != "xyz":
            raise ValueError("residuedensity currently supports radial 3D ('xyz') only")
        if refsel is None:
            raise ValueError("refsel is required for calc='residuedensity'")
        return residue_density_profile_from_dcd(
            pdb_file=Path(pdb),
            dcd_files=dcd_files,
            reference_segments=reference_selection,
            selection=analysis_selection,
            reference_label=reference_label,
            atom_name="CA" if selection_atom_name is None else selection_atom_name,
            residue_site=residue_site,
            mode=mode,
            reference_image_mode=_normalize_reference_image_mode(reference_image_mode),
            bins=bins,
            value_range=value_range,
            time_start=time_start,
            time_stop=time_stop,
            delta_t=delta_t,
            **common,
        )

    if calc_key == "intrachain":
        if pairsel is None:
            raise ValueError("pairsel is required for calc='intrachain'")
        return intrachain_distances_from_dcd(
            pdb_file=Path(pdb),
            dcd_files=dcd_files,
            chains=selection,
            selection=pairsel,
            include_hydrogens=bool(include_hydrogens),
            sidechain_only=bool(sidechain_only),
            pbc=bool(pbc),
            **common,
        )

    if calc_key == "mincontact":
        if pairsel is None:
            raise ValueError("pairsel is required for calc='mincontact'")
        return minimum_contact_distances_from_dcd(
            pdb_file=Path(pdb),
            dcd_files=dcd_files,
            chains=selection,
            selection=pairsel,
            option=option,
            include_hydrogens=bool(include_hydrogens),
            sidechain_only=bool(sidechain_only),
            pbc=bool(pbc),
            **common,
        )

    if calc_key == "interchain":
        if refsel is None:
            raise ValueError("refsel is required for calc='interchain'")
        if pairsel is None:
            raise ValueError("pairsel is required for calc='interchain'")
        return interchain_distances_from_dcd(
            pdb_file=Path(pdb),
            dcd_files=dcd_files,
            target_chains=selection,
            reference_chain=refsel,
            selection=pairsel,
            exclude_reference=bool(exclude_reference),
            include_hydrogens=bool(include_hydrogens),
            sidechain_only=bool(sidechain_only),
            pbc=bool(pbc),
            **common,
        )

    supported = "rgyr, contacts, centerdist, comtraj, densityprofile, intrachain, interchain, mincontact, distmsd, distmsd1d, distmsdmap, contactmap, peptide-torsions"
    raise ValueError(f"unknown calculation type {calc!r}; choose one of: {supported}")


def _filename_component(value: Any) -> str:
    """Make a user label safe as one filename component."""
    text = str(value).strip()
    if not text:
        raise ValueError("set and group labels must be non-empty")
    for separator in {os.sep, os.altsep} - {None}:
        text = text.replace(separator, "_")
    return text


def _get_values_serial(
    sets: Sequence[str],
    segs: Mapping[str, Any],
    calc: str = "rgyr",
    *,
    overwrite: bool = False,
    base: Optional[PathLike] = None,
    tag: str = "",
    sel2: Any = None,
    refsel: Any = None,
    pairsel: Any = None,
    option: Optional[str] = None,
    include_hydrogens: bool = False,
    sidechain_only: bool = False,
    workers: int = 1,
    query_site_chunk: int = 4096,
    pdb_name: str = "protein.pdb",
    dcd_name: str = "protein.dcd",
    results_subdir: str = "results",
    stride: int = 1,
    chunk: int = 200,
    frame_start: int = 0,
    frame_stop: Optional[int] = None,
    time_start: Any = None,
    time_stop: Any = None,
    delta_t: Any = None,
    bins: Union[int, Sequence[float], np.ndarray] = 60,
    value_range: Optional[tuple[float, float]] = None,
    box_nm: Optional[Sequence[float]] = None,
    mode: str = "cog",
    cutoff_nm: float = 0.8,
    pbc: bool = True,
    diffusion_mode: str = "both",
    distance_image: str = "hybrid",
    lags_frames: Optional[Sequence[int]] = None,
    max_lag_frames: int = 200,
    lag_stride: int = 1,
    origin_stride: int = 1,
    inter_targets_per_ref: Optional[int] = None,
    random_seed: int = 0,
    backend: str = "serial",
    n_jobs: int = 0,
    mp_start_method: Optional[str] = None,
    min_intra_sequence_separation: int = 2,
    pair_block_size: int = 8,
    origin_chunk: int = 16,
    reference_chunk: int = 2,
    map_work_dtype: str = "float32",
    map_work_dir: Optional[PathLike] = None,
    exclude_reference: bool = True,
    reference_label: str = "reference_rlp",
    distance_axes: str = "xyz",
    reference_image_mode: str = "as_is",
    selection_atom_name: Optional[str] = "CA",
    residue_site: Optional[str] = None,
    remove_reference_motion: bool = False,
    unwrap_within_frame: bool = False,
    degrees: bool = False,
    check_peptide_bonds: bool = True,
    peptide_bond_cutoff_nm: float = 0.22,
    torsion_dtype: Any = "float32",
) -> dict[str, Any]:
    """Calculate or load cached results for each set/group combination.

    ``residue_site='CA'`` samples C-alpha positions for residuedensity;
    ``'COM'`` and ``'COG'`` use all selected atoms of each residue. Distinct
    residue sites use distinct cache names, except that the previous default
    CA cache name is retained for compatibility.

    Cache filenames retain the original ``set.group.calc<tag>.pkl`` convention.
    When a matching cache file exists and ``overwrite=False``, it is loaded
    without requiring the original PDB or DCD files to still be present. PDB/DCD
    existence is checked only when a calculation must actually be performed.
    Because the filename does not encode every calculation parameter, use a
    distinct ``tag`` or ``overwrite=True`` after changing selections, cutoffs,
    frame ranges, ``selection_atom_name``, or other analysis settings. The
    For ``calc="distmsd"``, ``segs`` supplies the StructureSelector-compatible
    chain selection and ``pairsel`` supplies the ordered two-atom selection. The
    returned mapping uses the same ``<set>_<group>`` keys as the other analyses.
    ``bins`` and ``value_range`` define initial-distance bins; ``delta_t`` is required.

    For ``calc="peptide-torsions"``, each ``segs`` group retains all residues of
    matching atomistic chains (coarse-grained-only groups have zero chains).
    The DCD is read once per set across all groups in serial mode.
    The cache includes units, PBC, peptide-bond
    validation settings, and dtype in its filename. Use ``tag`` or ``overwrite``
    when changing the trajectory stride or frame/time window.

    For ``calc="distmsdmap"``, ``segs`` supplies the full homologous-chain pool.
    Optional ``refsel`` restricts only the physical/reference chains whose replicate
    MSDs are averaged; interchain targets remain the full ``segs`` selection.
    ``refsel`` may be one selection used for every dataset, or a mapping keyed by
    ``<set>_<group>`` so that each simulation set can use a different reference-chain
    subset, for example ``{f"{s1}_asn": chains1, f"{s2}_asn": chains2}``.
    ``selection_atom_name`` selects the site atom for all retained residues, and
    ``value_range`` is the single initial-distance conditioning window (or None for
    any distance). Changing ``refsel``, the distance window, or other map-generation
    parameters requires a distinct cache tag or ``overwrite=True``. The fit-time
    window is applied later by :func:`fit_dist_diffusion_maps` and does not require
    regenerating the cache.

    For ``calc="contactmap"``, the cached object contains compact sparse
    per-frame CA contact events up to ``cutoff_nm``. ``query_site_chunk`` defaults
    to 4096; reduce it if contact generation still uses too much peak RAM. Use
    :func:`calculate_contact_maps` for time/property filtering. The
    ``option`` value is intentionally not encoded in the cache filename; use
    distinct tags for ``intraonly`` and ``interonly``. By default, ``rgyr``,
    ``contacts``, and ``centerdist`` use CA atoms. For
    ``densityprofile``, the density selection uses all selected atoms while the
    reference selection uses CA atoms by default. Cartesian axes and
    ``reference_image_mode`` are encoded in cache names for both center-distance
    and density-profile calculations. Bin definitions and time windows are not
    encoded; use a distinct ``tag`` or ``overwrite=True`` when changing them.
    """
    if not sets:
        raise ValueError("sets is empty")
    if not segs:
        raise ValueError("segs is empty")

    root = Path(".") if base is None else Path(base)
    results_dir = root / results_subdir
    results_dir.mkdir(parents=True, exist_ok=True)

    is_centerdist = _is_center_distance_calculation(calc)
    is_densityprofile = _is_density_profile_calculation(calc)
    is_residuedensity = str(calc).strip().lower().replace("-", "_") in {
        "residuedensity",
        "residue_density",
        "residue_density_profile",
    }
    calc_key = str(calc).strip().lower().replace("-", "_")
    is_comtraj = calc_key in {"comtraj", "com_trajectory", "chain_com", "translational_trajectory"}
    is_peptide_torsions = calc_key in {"peptide_torsions", "peptidetorsions", "torsion", "torsions"}
    is_distmsdmap = calc_key in {
        "distmsdmap",
        "dist_msd_map",
        "distance_msd_map",
        "distance_diffusion_map",
        "distance_dependent_diffusion_map",
        "distdiffmap",
        "diffusionmap",
    }
    uses_reference_geometry = is_centerdist or is_densityprofile or is_residuedensity or is_comtraj
    distance_axes_name = (
        _normalize_distance_axes(distance_axes) if uses_reference_geometry else "xyz"
    )
    reference_image_mode_name = (
        _normalize_reference_image_mode(reference_image_mode)
        if uses_reference_geometry
        else "as_is"
    )

    data: dict[str, Any] = {}
    for set_label in sets:
        set_name = str(set_label)
        pdb_file = root / set_name / pdb_name
        dcd_file = root / set_name / dcd_name
        # mdsim evaluates all atomistic chains, irrespective of the group.
        # Reuse one calculation across groups within this simulation set.
        all_torsions_for_set: Optional[PeptideTorsionResult] = None

        for group_label, selection in segs.items():
            group_name = str(group_label)
            key = f"{set_name}_{group_name}"
            cache_calc_label = f"{'peptide-torsions' if is_peptide_torsions else calc}{tag}"
            if is_peptide_torsions:
                # Unlike most analysis names, these fundamentally change the
                # output or geometry and must never reuse a cache accidentally.
                cache_calc_label += ".degrees" if degrees else ".radians"
                cache_calc_label += ".pbc" if pbc else ".whole"
                cache_calc_label += (
                    f".bondcut-{float(peptide_bond_cutoff_nm):g}nm"
                    if check_peptide_bonds
                    else ".no-bond-check"
                )
                cache_calc_label += f".{np.dtype(torsion_dtype).name}"
            if (
                is_densityprofile
                or is_residuedensity
                or (is_centerdist and distance_axes_name != "xyz")
            ):
                cache_calc_label = f"{cache_calc_label}.{distance_axes_name}"
            if uses_reference_geometry:
                cache_calc_label = f"{cache_calc_label}.{reference_image_mode_name}"
            if is_residuedensity:
                reference_atom = (
                    str(selection_atom_name).strip().upper()
                    if selection_atom_name is not None
                    else "ALL"
                )
                actual_site = (
                    ("CA" if selection_atom_name is None else reference_atom)
                    if residue_site is None
                    else str(residue_site).strip().upper()
                )
                # Preserve previous cache names when the site and reference
                # selections are the original identical atom (usually CA).
                if reference_atom != "ALL" and actual_site == reference_atom:
                    cache_calc_label += f".{reference_atom}.{mode}"
                else:
                    cache_calc_label += f".ref-{reference_atom}.site-{actual_site}.{mode}"
            if is_comtraj:
                cache_calc_label += ".relative" if remove_reference_motion else ".lab"
                cache_calc_label += ".whole" if not unwrap_within_frame else ".atomunwrap"
                cache_calc_label += f".{mode}"

            cache_file = results_dir / (
                f"{_filename_component(set_name)}."
                f"{_filename_component(group_name)}."
                f"{_filename_component(cache_calc_label)}.pkl"
            )

            if overwrite or not cache_file.exists():
                if not pdb_file.is_file():
                    raise FileNotFoundError(f"PDB file not found: {pdb_file}")
                if not dcd_file.is_file():
                    raise FileNotFoundError(f"DCD file not found: {dcd_file}")

                refsel_for_key = refsel
                if (is_distmsdmap or is_comtraj) and isinstance(refsel, Mapping):
                    if key not in refsel:
                        available = ", ".join(repr(str(item)) for item in refsel)
                        suffix = f" Available keys: {available}" if available else ""
                        raise KeyError(
                            f"refsel does not contain a selection for dataset {key!r}." f"{suffix}"
                        )
                    refsel_for_key = refsel[key]

                if is_peptide_torsions and all_torsions_for_set is not None:
                    value = _selected_peptide_torsions(all_torsions_for_set, pdb_file, selection)
                else:
                    value = calculate(
                        calc=calc,
                        pdb=pdb_file,
                        dcd=dcd_file,
                        selection="all" if is_peptide_torsions else selection,
                        selection2=sel2,
                        refsel=refsel_for_key,
                        pairsel=pairsel,
                        option=option,
                        include_hydrogens=include_hydrogens,
                        sidechain_only=sidechain_only,
                        workers=workers,
                        query_site_chunk=query_site_chunk,
                        stride=stride,
                        chunk=chunk,
                        frame_start=frame_start,
                        frame_stop=frame_stop,
                        time_start=time_start,
                        time_stop=time_stop,
                        delta_t=delta_t,
                        bins=bins,
                        value_range=value_range,
                        box_nm=box_nm,
                        mode=mode,
                        cutoff_nm=cutoff_nm,
                        pbc=pbc,
                        diffusion_mode=diffusion_mode,
                        distance_image=distance_image,
                        lags_frames=lags_frames,
                        max_lag_frames=max_lag_frames,
                        lag_stride=lag_stride,
                        origin_stride=origin_stride,
                        inter_targets_per_ref=inter_targets_per_ref,
                        random_seed=random_seed,
                        backend=backend,
                        n_jobs=n_jobs,
                        mp_start_method=mp_start_method,
                        min_intra_sequence_separation=min_intra_sequence_separation,
                        pair_block_size=pair_block_size,
                        origin_chunk=origin_chunk,
                        reference_chunk=reference_chunk,
                        map_work_dtype=map_work_dtype,
                        map_work_dir=map_work_dir,
                        exclude_reference=exclude_reference,
                        reference_label=reference_label,
                        distance_axes=distance_axes_name,
                        reference_image_mode=reference_image_mode_name,
                        selection_atom_name=selection_atom_name,
                        residue_site=residue_site,
                        remove_reference_motion=remove_reference_motion,
                        unwrap_within_frame=unwrap_within_frame,
                        degrees=degrees,
                        check_peptide_bonds=check_peptide_bonds,
                        peptide_bond_cutoff_nm=peptide_bond_cutoff_nm,
                        torsion_dtype=torsion_dtype,
                    )
                    if is_peptide_torsions:
                        all_torsions_for_set = value
                        value = _selected_peptide_torsions(
                            all_torsions_for_set, pdb_file, selection
                        )
                save_data(cache_file, value)
                data[key] = value
            else:
                data[key] = load_data(cache_file)

    return data


def _parallel_worker_count(value: int) -> int:
    """Resolve a notebook-level parallel worker count.

    ``-1`` means all logical CPUs; positive integers request that many workers.
    """
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("parallel_workers must be -1 or a positive integer")
    workers = int(value)
    if workers == -1:
        return int(os.cpu_count() or 1)
    if workers < 1:
        raise ValueError("parallel_workers must be -1 or a positive integer")
    return workers


def _parallel_backend_name(value: str) -> str:
    """Normalize thread/process names used by :func:`get_values`."""
    if not isinstance(value, str):
        raise TypeError("parallel_backend must be a string")
    name = value.strip().lower().replace("-", "_")
    aliases = {
        "thread": "thread",
        "threads": "thread",
        "threading": "thread",
        "process": "process",
        "processes": "process",
        "multiprocessing": "process",
        "mp": "process",
    }
    if name not in aliases:
        raise ValueError("parallel_backend must be 'thread' or 'process'")
    return aliases[name]


def get_values(
    sets: Sequence[str],
    segs: Mapping[str, Any],
    calc: str = "rgyr",
    *,
    overwrite: bool = False,
    base: Optional[PathLike] = None,
    tag: str = "",
    sel2: Any = None,
    refsel: Any = None,
    pairsel: Any = None,
    option: Optional[str] = None,
    include_hydrogens: bool = False,
    sidechain_only: bool = False,
    workers: int = 1,
    parallel_workers: int = 1,
    parallel_backend: str = "thread",
    query_site_chunk: int = 4096,
    pdb_name: str = "protein.pdb",
    dcd_name: str = "protein.dcd",
    results_subdir: str = "results",
    stride: int = 1,
    chunk: int = 200,
    frame_start: int = 0,
    frame_stop: Optional[int] = None,
    time_start: Any = None,
    time_stop: Any = None,
    delta_t: Any = None,
    bins: Union[int, Sequence[float], np.ndarray] = 60,
    value_range: Optional[tuple[float, float]] = None,
    box_nm: Optional[Sequence[float]] = None,
    mode: str = "cog",
    cutoff_nm: float = 0.8,
    pbc: bool = True,
    diffusion_mode: str = "both",
    distance_image: str = "hybrid",
    lags_frames: Optional[Sequence[int]] = None,
    max_lag_frames: int = 200,
    lag_stride: int = 1,
    origin_stride: int = 1,
    inter_targets_per_ref: Optional[int] = None,
    random_seed: int = 0,
    backend: str = "serial",
    n_jobs: int = 0,
    mp_start_method: Optional[str] = None,
    min_intra_sequence_separation: int = 2,
    pair_block_size: int = 8,
    origin_chunk: int = 16,
    reference_chunk: int = 2,
    map_work_dtype: str = "float32",
    map_work_dir: Optional[PathLike] = None,
    exclude_reference: bool = True,
    reference_label: str = "reference_rlp",
    distance_axes: str = "xyz",
    reference_image_mode: str = "as_is",
    selection_atom_name: Optional[str] = "CA",
    residue_site: Optional[str] = None,
    remove_reference_motion: bool = False,
    unwrap_within_frame: bool = False,
    degrees: bool = False,
    check_peptide_bonds: bool = True,
    peptide_bond_cutoff_nm: float = 0.22,
    torsion_dtype: Any = "float32",
) -> dict[str, Any]:
    """Calculate/load results, optionally parallelizing independent set/group jobs.

    ``calc="peptide-torsions"`` returns one mdsim PeptideTorsionResult per
    set/group with only the selected physical chains. Radians are the default.

    ``parallel_workers`` normally parallelizes independent set/group jobs.
    For ``peptide-torsions`` it instead parallelizes simulation sets, avoiding
    redundant DCD passes when ``segs`` contains several groups. The default
    of 1 is serial; ``parallel_workers=-1`` uses all logical CPUs.

    ``parallel_backend='thread'`` is the default because NumPy/SciPy trajectory
    kernels release the GIL and threads avoid copying large result objects between
    processes.  ``parallel_backend='process'`` is available for Python-heavy
    calculations and gives true process isolation, but every worker opens its own
    trajectory and all arguments/results must be pickleable.

    The existing ``workers`` argument is independent: it is passed down to
    calculations such as ``contacts``/``contactmap`` where SciPy itself can use
    multiple threads.  Avoid large values for both ``parallel_workers`` and
    ``workers`` simultaneously unless oversubscription is intentional.

    For ``calc`` equal to ``intrachain``, ``interchain``, or ``mincontact``,
    ``pairsel`` defines two sites. A site may contain one explicit atom or several
    explicit atoms joined with ``+``; the minimum cross-site atom distance is
    returned. For example, ``"39.CE2+CD2,69.SG+CB"`` compares two
    two-atom sites, while ``"39.CE2,69.SG"`` is the one-atom special case.
    Two bare residue numbers (for example ``"39,69"``) automatically compare all
    retained residue atoms. Residue mode uses heavy atoms by default; set
    ``include_hydrogens=True`` for all atoms and/or ``sidechain_only=True`` for
    the side-chain atom subset.
    """
    n_outer = _parallel_worker_count(parallel_workers)
    backend_name = _parallel_backend_name(parallel_backend)
    calc_key = str(calc).strip().lower().replace("-", "_")
    is_peptide_torsions = calc_key in {"peptide_torsions", "peptidetorsions", "torsion", "torsions"}

    # The mdsim torsion routine processes all atomistic chains per DCD.
    # With one set, run the groups together so the DCD is read just once.
    if n_outer == 1 or (len(sets) * len(segs)) <= 1 or (is_peptide_torsions and len(sets) <= 1):
        return _get_values_serial(
            sets,
            segs,
            calc=calc,
            overwrite=overwrite,
            base=base,
            tag=tag,
            sel2=sel2,
            refsel=refsel,
            pairsel=pairsel,
            option=option,
            include_hydrogens=include_hydrogens,
            sidechain_only=sidechain_only,
            workers=workers,
            query_site_chunk=query_site_chunk,
            pdb_name=pdb_name,
            dcd_name=dcd_name,
            results_subdir=results_subdir,
            stride=stride,
            chunk=chunk,
            frame_start=frame_start,
            frame_stop=frame_stop,
            time_start=time_start,
            time_stop=time_stop,
            delta_t=delta_t,
            bins=bins,
            value_range=value_range,
            box_nm=box_nm,
            mode=mode,
            cutoff_nm=cutoff_nm,
            pbc=pbc,
            diffusion_mode=diffusion_mode,
            distance_image=distance_image,
            lags_frames=lags_frames,
            max_lag_frames=max_lag_frames,
            lag_stride=lag_stride,
            origin_stride=origin_stride,
            inter_targets_per_ref=inter_targets_per_ref,
            random_seed=random_seed,
            backend=backend,
            n_jobs=n_jobs,
            mp_start_method=mp_start_method,
            min_intra_sequence_separation=min_intra_sequence_separation,
            pair_block_size=pair_block_size,
            origin_chunk=origin_chunk,
            reference_chunk=reference_chunk,
            map_work_dtype=map_work_dtype,
            map_work_dir=map_work_dir,
            exclude_reference=exclude_reference,
            reference_label=reference_label,
            distance_axes=distance_axes,
            reference_image_mode=reference_image_mode,
            selection_atom_name=selection_atom_name,
            residue_site=residue_site,
            remove_reference_motion=remove_reference_motion,
            unwrap_within_frame=unwrap_within_frame,
            degrees=degrees,
            check_peptide_bonds=check_peptide_bonds,
            peptide_bond_cutoff_nm=peptide_bond_cutoff_nm,
            torsion_dtype=torsion_dtype,
        )

    common_kwargs = dict(
        calc=calc,
        overwrite=overwrite,
        base=base,
        tag=tag,
        sel2=sel2,
        refsel=refsel,
        pairsel=pairsel,
        option=option,
        include_hydrogens=include_hydrogens,
        sidechain_only=sidechain_only,
        workers=workers,
        query_site_chunk=query_site_chunk,
        pdb_name=pdb_name,
        dcd_name=dcd_name,
        results_subdir=results_subdir,
        stride=stride,
        chunk=chunk,
        frame_start=frame_start,
        frame_stop=frame_stop,
        time_start=time_start,
        time_stop=time_stop,
        delta_t=delta_t,
        bins=bins,
        value_range=value_range,
        box_nm=box_nm,
        mode=mode,
        cutoff_nm=cutoff_nm,
        pbc=pbc,
        diffusion_mode=diffusion_mode,
        distance_image=distance_image,
        lags_frames=lags_frames,
        max_lag_frames=max_lag_frames,
        lag_stride=lag_stride,
        origin_stride=origin_stride,
        inter_targets_per_ref=inter_targets_per_ref,
        random_seed=random_seed,
        backend=backend,
        n_jobs=n_jobs,
        mp_start_method=mp_start_method,
        min_intra_sequence_separation=min_intra_sequence_separation,
        pair_block_size=pair_block_size,
        origin_chunk=origin_chunk,
        reference_chunk=reference_chunk,
        map_work_dtype=map_work_dtype,
        map_work_dir=map_work_dir,
        exclude_reference=exclude_reference,
        reference_label=reference_label,
        distance_axes=distance_axes,
        reference_image_mode=reference_image_mode,
        selection_atom_name=selection_atom_name,
        residue_site=residue_site,
        remove_reference_motion=remove_reference_motion,
        unwrap_within_frame=unwrap_within_frame,
        degrees=degrees,
        check_peptide_bonds=check_peptide_bonds,
        peptide_bond_cutoff_nm=peptide_bond_cutoff_nm,
        torsion_dtype=torsion_dtype,
    )

    if is_peptide_torsions:
        # Parallelize independent sets, rather than groups from the same DCD.
        # Each worker creates and caches all requested group results in one pass.
        Executor = ThreadPoolExecutor if backend_name == "thread" else ProcessPoolExecutor
        merged: dict[str, Any] = {}
        with Executor(max_workers=min(n_outer, len(sets))) as executor:
            future_to_set = {
                executor.submit(_get_values_serial, [str(set_label)], segs, **common_kwargs): str(
                    set_label
                )
                for set_label in sets
            }
            for future in as_completed(future_to_set):
                set_name = future_to_set[future]
                try:
                    merged.update(future.result())
                except Exception as exc:
                    raise RuntimeError(
                        f"parallel peptide-torsions calculation failed for set {set_name!r}"
                    ) from exc
        return {
            f"{set_label}_{group_label}": merged[f"{set_label}_{group_label}"]
            for set_label in map(str, sets)
            for group_label in map(str, segs)
        }

    tasks: list[tuple[str, str, Any]] = []
    expected_order: list[str] = []
    for set_label in sets:
        set_name = str(set_label)
        for group_label, selection in segs.items():
            group_name = str(group_label)
            key = f"{set_name}_{group_name}"
            expected_order.append(key)
            tasks.append((key, set_name, (group_label, selection)))

    Executor = ThreadPoolExecutor if backend_name == "thread" else ProcessPoolExecutor
    max_workers = min(n_outer, len(tasks))
    merged: dict[str, Any] = {}

    with Executor(max_workers=max_workers) as executor:
        future_to_key = {}
        for key, set_name, group_item in tasks:
            group_label, selection = group_item
            future = executor.submit(
                _get_values_serial,
                [set_name],
                {group_label: selection},
                **common_kwargs,
            )
            future_to_key[future] = key

        for future in as_completed(future_to_key):
            key = future_to_key[future]
            try:
                result = future.result()
            except Exception as exc:
                raise RuntimeError(f"parallel get_values calculation failed for {key!r}") from exc
            if key not in result:
                raise RuntimeError(
                    f"parallel get_values task for {key!r} returned keys {list(result)}"
                )
            merged[key] = result[key]

    return {key: merged[key] for key in expected_order}


def get_interchain_values(
    sets: Sequence[str],
    segs: Mapping[str, Any],
    references: Sequence[str],
    *,
    pairsel: Any,
    include_hydrogens: bool = False,
    sidechain_only: bool = False,
    overwrite: bool = False,
    base: Optional[PathLike] = None,
    tag: str = "",
    pdb_name: str = "protein.pdb",
    dcd_name: str = "protein.dcd",
    results_subdir: str = "results",
    stride: int = 1,
    chunk: int = 200,
    frame_start: int = 0,
    frame_stop: Optional[int] = None,
    box_nm: Optional[Sequence[float]] = None,
    pbc: bool = True,
    exclude_reference: bool = True,
) -> dict[str, dict[str, InterchainDistanceResult]]:
    """Batch the common all-reference interchain-distance notebook workflow.

    The return layout intentionally matches the manual pattern::

        out[p] = get_values(..., calc="interchain", refsel=p, tag=f"_{p}")

    In other words, the outer mapping is keyed by reference label and each value
    is a normal ``get_values``-style ``{set_group: InterchainDistanceResult}``
    mapping. Existing downstream RDF code can therefore consume the result
    without structural changes.

    For every set/group with missing caches, all requested references are
    calculated in *one* call to
    :func:`interchain_distances_all_references_from_dcd`, so the DCD is read once
    instead of once per reference chain. Individual reference results are still
    cached under the same filenames produced by the old loop with
    ``tag=f"{tag}_{reference}"``.

    ``pairsel`` uses the same generalized two-site syntax as :func:`get_values`.
    Multiple explicit atoms within either site are joined with ``+`` and the
    minimum cross-site atom distance is returned. Two bare residue numbers select
    the all-residue-atom mode. Heavy atoms are the default in residue mode;
    ``include_hydrogens`` and ``sidechain_only`` have the same meaning as in
    :func:`get_values`.
    """
    if not sets:
        raise ValueError("sets is empty")
    if not segs:
        raise ValueError("segs is empty")
    if isinstance(references, (str, bytes)) or not isinstance(references, Sequence):
        raise TypeError("references must be a sequence of reference-chain selections")

    reference_specs = [str(value) for value in references]
    if not reference_specs or any(not value.strip() for value in reference_specs):
        raise ValueError("references must contain at least one non-empty selection")
    if len(set(reference_specs)) != len(reference_specs):
        raise ValueError("references contains duplicate selections")

    root = Path(".") if base is None else Path(base)
    results_dir = root / results_subdir
    results_dir.mkdir(parents=True, exist_ok=True)

    output: dict[str, dict[str, InterchainDistanceResult]] = {
        reference: {} for reference in reference_specs
    }

    for set_label in sets:
        set_name = str(set_label)
        pdb_file = root / set_name / pdb_name
        dcd_file = root / set_name / dcd_name

        for group_label, selection in segs.items():
            group_name = str(group_label)
            key = f"{set_name}_{group_name}"

            cache_files = {
                reference: results_dir
                / (
                    f"{_filename_component(set_name)}."
                    f"{_filename_component(group_name)}."
                    f"{_filename_component(f'interchain{tag}_{reference}')}.pkl"
                )
                for reference in reference_specs
            }

            missing: list[str] = []
            for reference in reference_specs:
                cache_file = cache_files[reference]
                if not overwrite and cache_file.exists():
                    output[reference][key] = load_data(cache_file)
                else:
                    missing.append(reference)

            if not missing:
                continue

            if not pdb_file.is_file():
                raise FileNotFoundError(f"PDB file not found: {pdb_file}")
            if not dcd_file.is_file():
                raise FileNotFoundError(f"DCD file not found: {dcd_file}")

            batch = interchain_distances_all_references_from_dcd(
                pdb_file=pdb_file,
                dcd_files=[dcd_file],
                reference_chains=missing,
                target_chains=selection,
                selection=pairsel,
                exclude_reference=bool(exclude_reference),
                include_hydrogens=bool(include_hydrogens),
                sidechain_only=bool(sidechain_only),
                pbc=bool(pbc),
                box_nm=box_nm,
                stride=int(stride),
                chunk=int(chunk),
                frame_start=int(frame_start),
                frame_stop=frame_stop,
            )

            # A reference spec may be an alias for the physical chain label.  The
            # batch routine preserves selector order, so pair results by order
            # while retaining the caller's reference strings as the outer keys
            # and cache suffixes.
            batch_values = list(batch.values())
            if len(batch_values) != len(missing):
                raise ValueError(
                    "each entry in references must resolve to exactly one unique " "physical chain"
                )

            for reference, value in zip(missing, batch_values):
                save_data(cache_files[reference], value)
                output[reference][key] = value

    return output


# --- cached CA contact maps --------------------------------------------------


def _normalize_contact_filter_mode(mode: str) -> str:
    """Normalize chain-property filtering for contact-map reduction."""
    if not isinstance(mode, str):
        raise TypeError("filter_mode must be a string")
    normalized = mode.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "query": "query",
        "reference": "query",
        "ref": "query",
        "partner": "partner",
        "target": "partner",
        "both": "both",
        "query_and_partner": "both",
        "reference_and_target": "both",
    }
    if normalized not in aliases:
        raise ValueError("filter_mode must be 'query', 'partner', or 'both'")
    return aliases[normalized]


def _contact_chain_selection_for_key(
    selection: Any,
    *,
    key: str,
    name: str,
) -> Optional[tuple[str, ...]]:
    """Resolve a global or per-dataset chain-label selection."""
    if selection is None:
        return None
    value = selection.get(key) if isinstance(selection, Mapping) else selection
    if value is None:
        return None
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} for {key!r} must be a sequence of chain labels")
    labels = tuple(str(item) for item in value)
    if len(set(labels)) != len(labels):
        raise ValueError(f"{name} for {key!r} contains duplicate labels")
    return labels


def _contact_static_chain_mask(
    labels: Sequence[str],
    *,
    include: Optional[tuple[str, ...]],
    exclude: Optional[tuple[str, ...]],
    strict: bool,
    key: str,
    role: str,
) -> np.ndarray:
    """Return one static chain mask from exact labels."""
    labels_tuple = tuple(str(label) for label in labels)
    if include is not None and exclude is not None:
        raise ValueError(
            f"{key!r}: include and exclude selections are mutually exclusive for {role}"
        )
    label_to_index = {label: index for index, label in enumerate(labels_tuple)}

    if include is not None:
        if not include:
            raise ValueError(f"{key!r}: include selection for {role} is empty")
        missing = [label for label in include if label not in label_to_index]
        if missing and strict:
            raise ValueError(
                f"{key!r}: unknown {role} chain labels in include selection: {missing}; "
                f"available labels: {list(labels_tuple)}"
            )
        wanted = set(include)
        mask = np.asarray([label in wanted for label in labels_tuple], dtype=bool)
    elif exclude is not None:
        missing = [label for label in exclude if label not in label_to_index]
        if missing and strict:
            raise ValueError(
                f"{key!r}: unknown {role} chain labels in exclude selection: {missing}; "
                f"available labels: {list(labels_tuple)}"
            )
        unwanted = set(exclude)
        mask = np.asarray([label not in unwanted for label in labels_tuple], dtype=bool)
    else:
        mask = np.ones(len(labels_tuple), dtype=bool)

    if not np.any(mask):
        raise ValueError(f"{key!r}: chain selection leaves no {role} chains")
    return mask


def _contact_filter_array_for_labels(
    filter_result: Any,
    *,
    labels: Sequence[str],
    expected_frames: int,
    filter_prop: Optional[str],
    key: str,
    role: str,
) -> np.ndarray:
    """Retrieve and align a filter property to ``(n_frames, n_chains)``.

    Most trajectory results in this module store chain-resolved properties as
    ``(n_chains, n_frames)``. Contact-map masks use the opposite orientation,
    ``(n_frames, n_chains)``. This helper accepts either orientation, aligns
    chains by ``chain_labels`` when available, and always returns
    ``(expected_frames, len(labels))``.
    """
    requested_labels = tuple(str(label) for label in labels)
    preferred_fields = (
        "contacts_per_chain",
        "rg_per_chain_nm",
        "distance_per_chain_nm",
        "n_chains_above_threshold",
    )
    fields = (
        [filter_prop]
        if filter_prop is not None
        else list(preferred_fields) + _candidate_filter_fields(filter_result)
    )

    source_labels: Optional[tuple[str, ...]] = None
    if _result_has_field(filter_result, "chain_labels"):
        source_labels = tuple(str(label) for label in _result_field(filter_result, "chain_labels"))
        if len(set(source_labels)) != len(source_labels):
            raise ValueError(f"{key!r}: filter chain_labels contains duplicates")

    matches: list[tuple[str, np.ndarray]] = []
    seen: set[str] = set()
    for field in fields:
        if field is None or field in seen or not _result_has_field(filter_result, field):
            continue
        seen.add(field)
        try:
            candidate = np.asarray(_result_field(filter_result, field), dtype=float)
        except (TypeError, ValueError):
            continue
        if candidate.ndim != 2:
            continue

        aligned: Optional[np.ndarray] = None

        if source_labels is not None:
            n_source = len(source_labels)
            source_index = {label: index for index, label in enumerate(source_labels)}
            if any(label not in source_index for label in requested_labels):
                continue
            selected_indices = np.asarray(
                [source_index[label] for label in requested_labels],
                dtype=np.int64,
            )

            # Standard convention used by contacts/Rg/distances:
            # (n_chains, n_frames) -> align rows, then transpose.
            if candidate.shape == (n_source, int(expected_frames)):
                aligned = candidate[selected_indices, :].T

            # Also accept already frame-major arrays:
            # (n_frames, n_chains) -> align columns directly.
            elif candidate.shape == (int(expected_frames), n_source):
                aligned = candidate[:, selected_indices]

        else:
            # Without labels we can only align by positional row/column order.
            n_requested = len(requested_labels)
            if candidate.shape == (n_requested, int(expected_frames)):
                aligned = candidate.T
            elif candidate.shape == (int(expected_frames), n_requested):
                aligned = candidate

        if aligned is None:
            continue
        if aligned.shape != (int(expected_frames), len(requested_labels)):
            continue

        matches.append((str(field), np.asarray(aligned, dtype=float)))

    if not matches:
        raise ValueError(
            f"{key!r}: no filter property can be aligned to the {role} chain labels "
            f"with {expected_frames} frames; pass filter_prop explicitly and ensure "
            "the filter result has compatible chain_labels"
        )
    if filter_prop is not None or len(matches) == 1:
        return matches[0][1]

    preferred = [item for item in matches if item[0] in preferred_fields]
    if len(preferred) == 1:
        return preferred[0][1]
    names = ", ".join(repr(name) for name, _ in matches)
    raise ValueError(
        f"{key!r}: multiple filter properties can be aligned for {role}: {names}; "
        "pass filter_prop explicitly"
    )


def calculate_contact_maps(
    data_by_set: Mapping[str, CAContactTrajectoryResult],
    *,
    cutoff_nm: Optional[float] = None,
    min_intra_sequence_separation: int = 1,
    inter_normalization: str = "per_reference_chain",
    frame_start: Optional[int] = None,
    frame_stop: Optional[int] = None,
    time_start: Any = None,
    time_stop: Any = None,
    delta_t: Any = None,
    filter_by: Optional[Mapping[str, Any]] = None,
    filter_prop: Optional[str] = None,
    filter_min: Optional[float] = None,
    filter_max: Optional[float] = None,
    filter_mode: str = "both",
    include_chains: Any = None,
    exclude_chains: Any = None,
    include_query_chains: Any = None,
    exclude_query_chains: Any = None,
    include_partner_chains: Any = None,
    exclude_partner_chains: Any = None,
    strict_chain_selection: bool = True,
    average_over_sets: bool = False,
    group_by: Optional[Mapping[str, str]] = None,
) -> dict[str, ResidueContactMapResult]:
    """Reduce cached per-frame CA contacts into filtered residue contact maps.

    The expensive trajectory-reading stage is performed by
    ``get_values(..., calc='contactmap')``.  This function works entirely from
    that cached sparse result.

    Time selection follows the other notebook helpers: frame stop and time stop
    are exclusive, and numeric time values are in nanoseconds.  ``filter_by``
    supplies a chain-by-frame property mapping keyed like ``data_by_set``.
    ``filter_mode`` applies the property limits to query chains, partner chains,
    or both.  For homotypic maps, ``include_chains``/``exclude_chains`` apply to
    both chain roles; the role-specific arguments override that shorthand.

    ``inter_normalization='per_reference_chain'`` reports the mean number of
    interchain CA contacts per reference chain and selected frame.
    ``'per_chain_pair'`` instead reports contact probability per ordered chain
    pair and selected frame.
    """
    if not data_by_set:
        raise ValueError("data_by_set is empty")

    filter_min, filter_max = _validate_filter_options(
        filter_by,
        filter_min=filter_min,
        filter_max=filter_max,
        filter_prop=filter_prop,
    )
    filter_mode_name = _normalize_contact_filter_mode(filter_mode)

    individual: dict[str, ResidueContactMapResult] = {}
    for key in sorted(data_by_set, key=_dataset_sort_key):
        raw = data_by_set[key]
        if not isinstance(raw, CAContactTrajectoryResult):
            required = (
                "frame_offsets",
                "query_chain_labels",
                "partner_chain_labels",
                "distance_nm",
            )
            if not all(hasattr(raw, field) for field in required):
                raise TypeError(
                    f"{key!r}: expected CAContactTrajectoryResult, got {type(raw).__name__}"
                )

        start, stop = _resolve_frame_slice(
            raw.n_frames,
            frame_start=frame_start,
            frame_stop=frame_stop,
            time_start=time_start,
            time_stop=time_stop,
            delta_t=delta_t,
        )
        frame_mask = np.zeros(raw.n_frames, dtype=bool)
        frame_mask[start:stop] = True

        common_include = _contact_chain_selection_for_key(
            include_chains, key=key, name="include_chains"
        )
        common_exclude = _contact_chain_selection_for_key(
            exclude_chains, key=key, name="exclude_chains"
        )
        query_include = _contact_chain_selection_for_key(
            include_query_chains, key=key, name="include_query_chains"
        )
        query_exclude = _contact_chain_selection_for_key(
            exclude_query_chains, key=key, name="exclude_query_chains"
        )
        partner_include = _contact_chain_selection_for_key(
            include_partner_chains, key=key, name="include_partner_chains"
        )
        partner_exclude = _contact_chain_selection_for_key(
            exclude_partner_chains, key=key, name="exclude_partner_chains"
        )

        if raw.symmetric_homotypic:
            if query_include is None:
                query_include = common_include
            if query_exclude is None:
                query_exclude = common_exclude
            if partner_include is None:
                partner_include = common_include
            if partner_exclude is None:
                partner_exclude = common_exclude

        query_static = _contact_static_chain_mask(
            raw.query_chain_labels,
            include=query_include,
            exclude=query_exclude,
            strict=bool(strict_chain_selection),
            key=key,
            role="query",
        )
        partner_static = _contact_static_chain_mask(
            raw.partner_chain_labels,
            include=partner_include,
            exclude=partner_exclude,
            strict=bool(strict_chain_selection),
            key=key,
            role="partner",
        )

        query_mask = np.broadcast_to(
            query_static.reshape(1, -1),
            (raw.n_frames, raw.n_query_chains),
        ).copy()
        partner_mask = np.broadcast_to(
            partner_static.reshape(1, -1),
            (raw.n_frames, raw.n_partner_chains),
        ).copy()
        query_mask &= frame_mask[:, None]
        partner_mask &= frame_mask[:, None]

        if filter_by is not None:
            if key not in filter_by:
                raise KeyError(f"filter_by does not contain a matching entry for {key!r}")
            filter_result = filter_by[key]

            if filter_mode_name in {"query", "both"}:
                query_filter = _contact_filter_array_for_labels(
                    filter_result,
                    labels=raw.query_chain_labels,
                    expected_frames=raw.n_frames,
                    filter_prop=filter_prop,
                    key=key,
                    role="query",
                )
                query_keep = _filter_threshold_mask(
                    query_filter,
                    filter_min=filter_min,
                    filter_max=filter_max,
                )
                query_mask &= query_keep

            if filter_mode_name in {"partner", "both"}:
                partner_filter = _contact_filter_array_for_labels(
                    filter_result,
                    labels=raw.partner_chain_labels,
                    expected_frames=raw.n_frames,
                    filter_prop=filter_prop,
                    key=key,
                    role="partner",
                )
                partner_keep = _filter_threshold_mask(
                    partner_filter,
                    filter_min=filter_min,
                    filter_max=filter_max,
                )
                partner_mask &= partner_keep

        individual[key] = residue_contact_map_from_trajectory(
            raw,
            cutoff_nm=cutoff_nm,
            frame_mask=frame_mask,
            query_chain_mask=query_mask,
            partner_chain_mask=partner_mask,
            min_intra_sequence_separation=min_intra_sequence_separation,
            inter_normalization=inter_normalization,
        )

    if not average_over_sets:
        return individual

    output: dict[str, ResidueContactMapResult] = {}
    groups = _ordered_set_groups(list(individual), group_by=group_by)
    for group, member_keys in groups.items():
        output[group] = average_residue_contact_maps(
            [individual[key] for key in member_keys],
            set_labels=member_keys,
        )
    return output


def plot_contact_map(
    result: ResidueContactMapResult,
    *,
    component: str = "combined",
    flip_triangles: bool = False,
    cmap: Any = "viridis",
    color_scale: str = "linear",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    colorbar: bool = True,
    colorbar_label: Optional[str] = None,
    annotate_triangles: bool = True,
    tick_step: Optional[int] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    figsize: tuple[float, float] = (7.0, 6.0),
    ax: Optional[plt.Axes] = None,
    save_path: Optional[PathLike] = None,
    save_dpi: Optional[float] = None,
    save_bbox_inches: Optional[str] = "tight",
    save_kwargs: Optional[Mapping[str, Any]] = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a combined, intrachain, or interchain residue contact map.

    By default, the combined homotypic map places intrachain contacts in the
    lower triangle and interchain contacts in the upper triangle. Set
    ``flip_triangles=True`` to swap them (interchain lower, intrachain upper).
    This option applies to ``component='combined'`` only; it does not change
    the input arrays. For heterotypic use, ``component='inter'`` plots the
    rectangular query-by-partner map.

    Axis labels use 1-based residue indices. Residue 1 is always shown; with
    ``tick_step=None`` subsequent labels are multiples of 5 for sequences with
    50 residues or fewer and multiples of 10 for longer sequences. Supplying
    ``tick_step`` overrides the automatic spacing.
    """
    component_name = str(component).strip().lower().replace("-", "_")
    aliases = {
        "combined": "combined",
        "both": "combined",
        "intra": "intra",
        "intrachain": "intra",
        "inter": "inter",
        "interchain": "inter",
    }
    if component_name not in aliases:
        raise ValueError("component must be 'combined', 'intra', or 'inter'")
    component_name = aliases[component_name]

    if not isinstance(flip_triangles, (bool, np.bool_)):
        raise TypeError("flip_triangles must be a boolean")

    if component_name == "combined":
        if result.combined is None:
            raise ValueError(
                "combined triangular plotting requires a square homotypic map; "
                "use component='inter' for a heterotypic map"
            )
        values = np.asarray(result.combined, dtype=float)
        if flip_triangles:
            values = values.T  # reflect the two triangles across the diagonal
        x_residues = result.partner_residue_numbers
        y_residues = result.query_residue_numbers
    elif component_name == "intra":
        values = np.asarray(result.intra, dtype=float)
        x_residues = result.partner_residue_numbers
        y_residues = result.query_residue_numbers
    else:
        values = np.asarray(result.inter, dtype=float)
        x_residues = result.partner_residue_numbers
        y_residues = result.query_residue_numbers

    scale = str(color_scale).strip().lower()
    if scale not in {"linear", "log"}:
        raise ValueError("color_scale must be 'linear' or 'log'")

    masked = np.ma.masked_invalid(values)
    norm = None
    if scale == "log":
        from matplotlib.colors import LogNorm

        masked = np.ma.masked_less_equal(masked, 0.0)
        positive = values[np.isfinite(values) & (values > 0.0)]
        if positive.size == 0:
            raise ValueError("the selected contact map contains no positive values for log color")
        lower = float(np.min(positive)) if vmin is None else float(vmin)
        upper = float(np.max(positive)) if vmax is None else float(vmax)
        if lower <= 0.0 or upper <= lower:
            raise ValueError("log color limits require 0 < vmin < vmax")
        norm = LogNorm(vmin=lower, vmax=upper)

    if ax is None:
        figure, ax = plt.subplots(figsize=figsize)
    else:
        figure = ax.figure

    cmap_object = plt.get_cmap(cmap).copy()
    cmap_object.set_bad("white")
    image_kwargs: dict[str, Any] = {
        "origin": "upper",
        "interpolation": "nearest",
        "aspect": "equal" if values.shape[0] == values.shape[1] else "auto",
        "cmap": cmap_object,
    }
    if norm is not None:
        image_kwargs["norm"] = norm
    else:
        image_kwargs["vmin"] = vmin
        image_kwargs["vmax"] = vmax

    image = ax.imshow(masked, **image_kwargs)

    def tick_positions_and_labels(n_values: int) -> tuple[np.ndarray, list[str]]:
        """Return 0-based pixel positions with 1-based residue-index labels.

        The first residue is always labeled ``1``. Thereafter, automatic labels
        are placed at multiples of 5 for sequences of 50 residues or fewer and
        at multiples of 10 for longer sequences. ``tick_step`` overrides this
        automatic choice while retaining the same 1-based labeling convention.
        """
        if n_values <= 0:
            return np.asarray([], dtype=np.int64), []

        if tick_step is not None:
            if isinstance(tick_step, (bool, np.bool_)):
                raise TypeError("tick_step must be an integer >= 1 or None")
            try:
                step_float = float(tick_step)
            except (TypeError, ValueError) as exc:
                raise TypeError("tick_step must be an integer >= 1 or None") from exc
            if not math.isfinite(step_float) or not step_float.is_integer() or step_float < 1:
                raise ValueError("tick_step must be an integer >= 1 or None")
            step = int(step_float)
        else:
            step = 5 if n_values <= 50 else 10

        labels = [1]
        labels.extend(range(step, n_values + 1, step))
        labels = list(dict.fromkeys(labels))

        # imshow pixel 0 corresponds to residue index 1, so residue label r is
        # located at pixel position r-1.
        positions = np.asarray([label - 1 for label in labels], dtype=np.int64)
        return positions, [str(label) for label in labels]

    x_ticks, x_tick_labels = tick_positions_and_labels(len(x_residues))
    y_ticks, y_tick_labels = tick_positions_and_labels(len(y_residues))
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_tick_labels)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_tick_labels)

    ax.set_xlabel(xlabel or "partner residue index")
    ax.set_ylabel(ylabel or "query residue index")
    if title is None:
        if component_name == "combined":
            title = (
                "CA contact map: interchain (lower), intrachain (upper)"
                if flip_triangles
                else "CA contact map: intrachain (lower), interchain (upper)"
            )
        elif component_name == "intra":
            title = "intrachain CA contact map"
        else:
            title = "interchain CA contact map"
    ax.set_title(title)

    if component_name == "combined" and annotate_triangles:
        ax.text(
            0.04,
            0.08,
            "interchain" if flip_triangles else "intrachain",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            color="white",
            bbox={"facecolor": "black", "alpha": 0.45, "edgecolor": "none", "pad": 2.0},
        )
        ax.text(
            0.96,
            0.92,
            "intrachain" if flip_triangles else "interchain",
            transform=ax.transAxes,
            ha="right",
            va="top",
            color="white",
            bbox={"facecolor": "black", "alpha": 0.45, "edgecolor": "none", "pad": 2.0},
        )

    if colorbar:
        cbar = figure.colorbar(image, ax=ax)
        if colorbar_label is None:
            colorbar_label = result.units
        cbar.set_label(colorbar_label)

    _save_figure_if_requested(
        figure,
        save_path,
        save_dpi=save_dpi,
        save_bbox_inches=save_bbox_inches,
        save_kwargs=save_kwargs,
    )
    return figure, ax


def count_chains_above_threshold(
    result,
    threshold,
    *,
    inclusive=False,
    output_field="n_chains_above_threshold",
):
    """
    Count, for every frame, how many chains have contact counts above a threshold.

    Returns an object compatible with plot_timeseries().
    """
    contacts = np.asarray(result.contacts_per_chain)

    if contacts.ndim != 2:
        raise ValueError("contacts_per_chain must have shape (n_chains, n_frames)")

    if inclusive:
        above = contacts >= threshold
    else:
        above = contacts > threshold

    # Shape required by plot_timeseries: (1, n_frames)
    counts = np.sum(above, axis=0, dtype=np.int64)[None, :]

    return SimpleNamespace(**{output_field: counts})


# --- quenching-curve averaging, fitting, and plotting ------------------------


@dataclass(frozen=True)
class QuenchingAverageResult:
    """One averaged reactive-quenching or brightness-autocorrelation curve.

    ``survival_mean`` is calculated over chains for a set-level result or over
    independent simulation sets for an overall result. ``survival_std`` and
    ``survival_stderr`` describe variation over those contributors.
    """

    lag_time_ps: np.ndarray
    survival_mean: np.ndarray
    survival_std: np.ndarray
    survival_stderr: np.ndarray
    n_contributors: np.ndarray
    observation_counts: np.ndarray
    contributor_labels: tuple[str, ...]
    aggregation: str

    @property
    def time_ps(self) -> np.ndarray:
        return self.lag_time_ps

    @property
    def values(self) -> np.ndarray:
        return self.survival_mean

    @property
    def curve_mean(self) -> np.ndarray:
        """Semantic alias for survival or autocorrelation data."""
        return self.survival_mean

    @property
    def curve_std(self) -> np.ndarray:
        """Semantic alias for :attr:`survival_std`."""
        return self.survival_std

    @property
    def curve_stderr(self) -> np.ndarray:
        """Semantic alias for :attr:`survival_stderr`."""
        return self.survival_stderr

    @property
    def series(self) -> np.ndarray:
        """Columns: time_ps, mean, SEM, SD, contributors, observations."""
        return np.column_stack(
            (
                self.lag_time_ps,
                self.survival_mean,
                self.survival_stderr,
                self.survival_std,
                self.n_contributors.astype(float),
                self.observation_counts,
            )
        )


@dataclass(frozen=True)
class QuenchingStatisticsResult:
    """Set-level and overall averages of reactive-quenching or brightness-autocorrelation curves."""

    per_set: dict[str, QuenchingAverageResult]
    overall: dict[str, QuenchingAverageResult]
    group_members: dict[str, tuple[str, ...]]
    min_sets: int
    within_set_weighting: str


@dataclass(frozen=True)
class QuenchingFitResult:
    """Single-exponential fit of a reactive or brightness-correlation decay.

    For a reactive-quenching correlation, use ``k_obs_per_s`` as the simulated
    Trp-quenching observed rate.  For a brightness autocorrelation, use
    ``k_relax_per_s`` for the equilibrium brightness/contact relaxation rate.
    Neither is the microscopic electron-transfer rate ``q(r)``.
    """

    tau0_ps: Optional[float]
    k0_per_s: float
    k_q_per_s: float
    k_obs_per_s: float
    tau_obs_s: float
    amplitude: float
    offset: float
    covariance: np.ndarray
    parameter_stderr: np.ndarray
    fit_time_ps: np.ndarray
    fitted_survival: np.ndarray
    n_points: int
    include_intrinsic_decay: bool
    enforce_zero_offset: bool
    sigma_source: Optional[str]

    @property
    def k0(self) -> float:
        return self.k0_per_s

    @property
    def k_q(self) -> float:
        return self.k_q_per_s

    @property
    def k_obs(self) -> float:
        """Compatibility alias; equals k_q for a quenching-only fit."""
        return self.k_obs_per_s

    @property
    def tau_q_s(self) -> float:
        """Quenching lifetime 1/k_q in seconds."""
        if self.k_q_per_s <= 0.0:
            return float("inf")
        return 1.0 / self.k_q_per_s

    @property
    def A(self) -> float:
        return self.amplitude

    @property
    def C(self) -> float:
        return self.offset

    @property
    def cov(self) -> np.ndarray:
        return self.covariance

    @property
    def k_relax_per_s(self) -> float:
        """Brightness-autocorrelation relaxation rate represented by the fit."""
        return self.k_obs_per_s

    @property
    def tau_relax_s(self) -> float:
        """Brightness-autocorrelation relaxation time represented by the fit."""
        return self.tau_obs_s

    @property
    def k_reactive_obs_per_s(self) -> float:
        """Observed rate from a fitted reactive-quenching correlation."""
        return self.k_obs_per_s

    @property
    def tau_reactive_obs_s(self) -> float:
        """Observed lifetime from a fitted reactive-quenching correlation."""
        return self.tau_obs_s

    @property
    def fitted_curve(self) -> np.ndarray:
        """Semantic alias for :attr:`fitted_survival`."""
        return self.fitted_survival


def _flatten_quenching_curve_results(
    data: Any,
    *,
    prefix: str = "",
) -> dict[str, Any]:
    """Flatten arbitrary mappings whose leaves are quenching-curve results."""
    is_curve = (
        _result_has_field(data, "lag_time_ps")
        and _result_has_field(data, "survival_per_chain")
        and _result_has_field(data, "counts_per_chain")
    )
    if is_curve:
        return {prefix or "data": data}

    if not isinstance(data, Mapping):
        raise TypeError(
            "data must be a QuenchingCurveResult or a mapping containing " "quenching-curve results"
        )
    if not data:
        raise ValueError("data is empty")

    flattened: dict[str, Any] = {}
    for raw_key, value in data.items():
        key = str(raw_key)
        path = key if not prefix else f"{prefix}/{key}"
        child = _flatten_quenching_curve_results(value, prefix=path)
        duplicate = set(flattened).intersection(child)
        if duplicate:
            raise ValueError(f"duplicate flattened quenching keys: {sorted(duplicate)}")
        flattened.update(child)
    return flattened


def _normalize_quenching_weighting(weighting: str) -> str:
    if not isinstance(weighting, str):
        raise TypeError("within_set_weighting must be a string")
    normalized = weighting.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "counts": "counts",
        "count": "counts",
        "observations": "counts",
        "time_origins": "counts",
        "equal": "equal",
        "chains": "equal",
        "equal_chains": "equal",
    }
    if normalized not in aliases:
        raise ValueError("within_set_weighting must be 'counts' or 'equal'")
    return aliases[normalized]


def _weighted_curve_statistics(
    values: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Column-wise weighted mean, sample SD, SEM, N, and total weight."""
    y = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if y.shape != w.shape or y.ndim != 2:
        raise ValueError("values and weights must share shape (n_contributors, n_bins)")

    valid = np.isfinite(y) & np.isfinite(w) & (w > 0.0)
    w_use = np.where(valid, w, 0.0)
    y_use = np.where(valid, y, 0.0)
    total_weight = np.sum(w_use, axis=0)
    n_contributors = np.sum(valid, axis=0, dtype=np.int64)

    mean = np.full(y.shape[1], np.nan, dtype=np.float64)
    has_data = total_weight > 0.0
    mean[has_data] = (
        np.sum(w_use[:, has_data] * y_use[:, has_data], axis=0) / total_weight[has_data]
    )

    std = np.zeros(y.shape[1], dtype=np.float64)
    stderr = np.zeros(y.shape[1], dtype=np.float64)

    # Kish effective sample size gives the ordinary sample formulas when all
    # contributors have equal weights, while avoiding false precision when one
    # chain dominates through many more time-origin observations.
    sum_w2 = np.sum(w_use * w_use, axis=0)
    effective_n = np.zeros(y.shape[1], dtype=np.float64)
    positive = sum_w2 > 0.0
    effective_n[positive] = total_weight[positive] ** 2 / sum_w2[positive]

    for column in np.flatnonzero(n_contributors >= 2):
        deviations = y[:, column] - mean[column]
        numerator = float(
            np.sum(w_use[:, column] * np.where(valid[:, column], deviations * deviations, 0.0))
        )
        denominator = float(total_weight[column])
        neff = float(effective_n[column])
        if denominator <= 0.0 or neff <= 1.0:
            continue
        population_variance = numerator / denominator
        sample_variance = population_variance * neff / (neff - 1.0)
        std[column] = math.sqrt(max(sample_variance, 0.0))
        stderr[column] = std[column] / math.sqrt(neff)

    return mean, std, stderr, n_contributors, total_weight


def _chain_selection_for_key(
    selection: Any,
    *,
    key: str,
    name: str,
) -> Optional[tuple[str, ...]]:
    """Resolve a global or per-set chain-label selection.

    ``selection`` may be one sequence of labels applied to every set, or a
    mapping from flattened dataset key to a sequence of labels. A string is
    rejected because it would otherwise be interpreted as a sequence of
    characters.
    """
    if selection is None:
        return None

    value = selection.get(key) if isinstance(selection, Mapping) else selection
    if value is None:
        return None
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{name} for {key!r} must be a sequence of chain labels, not a string")
    if not isinstance(value, Sequence):
        raise TypeError(f"{name} for {key!r} must be a sequence of chain labels")

    labels = tuple(str(item) for item in value)
    if len(set(labels)) != len(labels):
        raise ValueError(f"{name} for {key!r} contains duplicate chain labels")
    return labels


def _select_quenching_chain_rows(
    survival: np.ndarray,
    counts: np.ndarray,
    chain_labels: tuple[str, ...],
    *,
    include_chains: Optional[tuple[str, ...]],
    exclude_chains: Optional[tuple[str, ...]],
    strict: bool,
    label: str,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Select quenching rows by exact chain label before aggregation."""
    if include_chains is not None and exclude_chains is not None:
        raise ValueError("include_chains and exclude_chains are mutually exclusive")

    label_to_index = {chain_label: index for index, chain_label in enumerate(chain_labels)}

    if include_chains is not None:
        if not include_chains:
            raise ValueError(f"{label!r}: include_chains is empty")
        missing = [name for name in include_chains if name not in label_to_index]
        if missing and strict:
            raise ValueError(
                f"{label!r}: include_chains contains labels not present in the "
                f"quenching result: {missing}. Available labels: {list(chain_labels)}"
            )
        requested = set(include_chains)
        indices = [
            index for index, chain_label in enumerate(chain_labels) if chain_label in requested
        ]
    elif exclude_chains is not None:
        missing = [name for name in exclude_chains if name not in label_to_index]
        if missing and strict:
            raise ValueError(
                f"{label!r}: exclude_chains contains labels not present in the "
                f"quenching result: {missing}. Available labels: {list(chain_labels)}"
            )
        excluded = set(exclude_chains)
        indices = [
            index for index, chain_label in enumerate(chain_labels) if chain_label not in excluded
        ]
    else:
        indices = list(range(len(chain_labels)))

    if not indices:
        raise ValueError(f"{label!r}: chain selection leaves no chains to average")

    index_array = np.asarray(indices, dtype=np.int64)
    selected_labels = tuple(chain_labels[index] for index in indices)
    return survival[index_array, :], counts[index_array, :], selected_labels


def _average_one_quenching_set(
    result: Any,
    *,
    weighting: str,
    label: str,
    include_chains: Optional[tuple[str, ...]] = None,
    exclude_chains: Optional[tuple[str, ...]] = None,
    strict_chain_selection: bool = True,
) -> QuenchingAverageResult:
    time = np.asarray(_result_field(result, "lag_time_ps"), dtype=np.float64)
    survival = np.asarray(_result_field(result, "survival_per_chain"), dtype=np.float64)
    counts = np.asarray(_result_field(result, "counts_per_chain"), dtype=np.float64)

    if time.ndim != 1:
        raise ValueError(f"{label!r}: lag_time_ps must be one-dimensional")
    if survival.ndim != 2:
        raise ValueError(f"{label!r}: survival_per_chain must have shape (n_chains, n_bins)")
    if counts.shape != survival.shape:
        raise ValueError(
            f"{label!r}: counts_per_chain has shape {counts.shape}, expected {survival.shape}"
        )
    if survival.shape[1] != time.size:
        raise ValueError(
            f"{label!r}: survival data have {survival.shape[1]} bins, "
            f"but lag_time_ps has {time.size}"
        )
    if not np.all(np.isfinite(time)) or not np.all(np.diff(time) > 0.0):
        raise ValueError(f"{label!r}: lag_time_ps must be finite and strictly increasing")

    n_chains = int(survival.shape[0])
    if n_chains < 1:
        raise ValueError(f"{label!r}: no chain curves are present")

    if _result_has_field(result, "chain_labels"):
        chain_labels = tuple(str(value) for value in _result_field(result, "chain_labels"))
        if len(chain_labels) != n_chains:
            raise ValueError(
                f"{label!r}: chain_labels contains {len(chain_labels)} labels, "
                f"expected {n_chains}"
            )
        if len(set(chain_labels)) != len(chain_labels):
            raise ValueError(f"{label!r}: chain_labels contains duplicate labels")
    else:
        chain_labels = tuple(f"chain {index}" for index in range(n_chains))

    survival, counts, chain_labels = _select_quenching_chain_rows(
        survival,
        counts,
        chain_labels,
        include_chains=include_chains,
        exclude_chains=exclude_chains,
        strict=bool(strict_chain_selection),
        label=label,
    )

    if weighting == "counts":
        weights = counts
    else:
        weights = np.where(
            np.isfinite(survival) & np.isfinite(counts) & (counts > 0.0),
            1.0,
            0.0,
        )

    mean, std, stderr, n_contributors, _ = _weighted_curve_statistics(
        survival,
        weights,
    )
    observation_counts = np.sum(
        np.where(
            np.isfinite(survival) & np.isfinite(counts) & (counts > 0.0),
            counts,
            0.0,
        ),
        axis=0,
    )
    keep = np.isfinite(mean)
    if not np.any(keep):
        raise ValueError(f"{label!r}: no finite populated quenching bins")

    return QuenchingAverageResult(
        lag_time_ps=time[keep].copy(),
        survival_mean=mean[keep],
        survival_std=std[keep],
        survival_stderr=stderr[keep],
        n_contributors=n_contributors[keep],
        observation_counts=observation_counts[keep],
        contributor_labels=chain_labels,
        aggregation=("chains_count_weighted" if weighting == "counts" else "chains_equal_weight"),
    )


def _align_quenching_curves(
    curves: Sequence[QuenchingAverageResult],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Align set means and observation counts on the union of lag times."""
    if not curves:
        raise ValueError("no quenching curves were supplied")

    union_time = np.unique(
        np.concatenate([np.asarray(curve.lag_time_ps, dtype=np.float64) for curve in curves])
    )
    values = np.full((len(curves), union_time.size), np.nan, dtype=np.float64)
    observations = np.zeros((len(curves), union_time.size), dtype=np.float64)

    for row, curve in enumerate(curves):
        times = np.asarray(curve.lag_time_ps, dtype=np.float64)
        indices = np.searchsorted(union_time, times)
        if np.any(indices >= union_time.size) or not np.allclose(
            union_time[indices], times, rtol=1.0e-12, atol=1.0e-12
        ):
            raise RuntimeError("failed to align quenching lag-time grids")
        values[row, indices] = curve.survival_mean
        observations[row, indices] = curve.observation_counts

    return union_time, values, observations


def calculate_quenching_statistics(
    data: Any,
    *,
    within_set_weighting: str = "counts",
    min_sets: int = 2,
    group_by: Optional[Mapping[str, str]] = None,
    include_chains: Any = None,
    exclude_chains: Any = None,
    strict_chain_selection: bool = True,
) -> QuenchingStatisticsResult:
    """Average per-chain reactive-quenching or brightness-autocorrelation curves within sets and across sets.

    The leaves of ``data`` must be :class:`QuenchingCurveResult`-compatible
    objects. :class:`FluorescenceAutocorrelationResult` is compatible through
    its ``survival_per_chain`` alias. Arbitrarily nested mappings are flattened using ``/``-separated
    path labels. For the usual flat ``<set>_<group>`` mapping, overall curves
    are grouped by the suffix exactly like :func:`calculate_averages`.

    Chain curves are pooled within each simulation set using their time-origin
    counts by default. Independent simulation sets are always averaged with
    equal set weight. Both SD and SEM are retained in the returned curves.

    ``include_chains`` and ``exclude_chains`` select rows by exact values in
    each input result's ``chain_labels`` and are mutually exclusive. Each may
    be either one sequence applied to every set or a mapping from flattened
    dataset key to a sequence. Missing labels raise by default; set
    ``strict_chain_selection=False`` to ignore labels absent from a set. The
    selected labels are recorded in each per-set result's
    ``contributor_labels``.
    """
    weighting = _normalize_quenching_weighting(within_set_weighting)
    if include_chains is not None and exclude_chains is not None:
        raise ValueError("include_chains and exclude_chains are mutually exclusive")
    if not isinstance(strict_chain_selection, (bool, np.bool_)):
        raise TypeError("strict_chain_selection must be a boolean")

    if isinstance(min_sets, (bool, np.bool_)):
        raise TypeError("min_sets must be an integer >= 1")
    try:
        min_sets_value = int(min_sets)
    except (TypeError, ValueError) as exc:
        raise TypeError("min_sets must be an integer >= 1") from exc
    if min_sets_value < 1 or float(min_sets_value) != float(min_sets):
        raise ValueError("min_sets must be an integer >= 1")

    leaves = _flatten_quenching_curve_results(data)
    keys = sorted(leaves, key=_dataset_sort_key)

    for name, specification in (
        ("include_chains", include_chains),
        ("exclude_chains", exclude_chains),
    ):
        if isinstance(specification, Mapping):
            unknown = [str(key) for key in specification if str(key) not in leaves]
            if unknown:
                raise KeyError(
                    f"{name} contains dataset keys not present in data: {unknown}. "
                    f"Available keys: {keys}"
                )

    per_set: dict[str, QuenchingAverageResult] = {}
    for key in keys:
        include_for_key = _chain_selection_for_key(include_chains, key=key, name="include_chains")
        exclude_for_key = _chain_selection_for_key(exclude_chains, key=key, name="exclude_chains")
        per_set[key] = _average_one_quenching_set(
            leaves[key],
            weighting=weighting,
            label=key,
            include_chains=include_for_key,
            exclude_chains=exclude_for_key,
            strict_chain_selection=bool(strict_chain_selection),
        )

    groups = _ordered_set_groups(keys, group_by=group_by)
    overall: dict[str, QuenchingAverageResult] = {}
    group_members: dict[str, tuple[str, ...]] = {}

    for group, member_keys_raw in groups.items():
        member_keys = tuple(str(key) for key in member_keys_raw)
        curves = [per_set[key] for key in member_keys]
        time, set_values, set_observations = _align_quenching_curves(curves)
        valid = np.isfinite(set_values)
        n_contributors = np.sum(valid, axis=0, dtype=np.int64)
        keep = n_contributors >= min_sets_value
        if not np.any(keep):
            continue

        mean = np.full(time.size, np.nan, dtype=np.float64)
        std = np.zeros(time.size, dtype=np.float64)
        stderr = np.zeros(time.size, dtype=np.float64)

        mean[keep] = np.nanmean(set_values[:, keep], axis=0)
        for column in np.flatnonzero(keep):
            values = set_values[valid[:, column], column]
            if values.size >= 2:
                std[column] = float(np.std(values, ddof=1))
                stderr[column] = std[column] / math.sqrt(float(values.size))

        observation_counts = np.sum(
            np.where(valid, set_observations, 0.0),
            axis=0,
        )
        overall[group] = QuenchingAverageResult(
            lag_time_ps=time[keep],
            survival_mean=mean[keep],
            survival_std=std[keep],
            survival_stderr=stderr[keep],
            n_contributors=n_contributors[keep],
            observation_counts=observation_counts[keep],
            contributor_labels=member_keys,
            aggregation="sets_equal_weight",
        )
        group_members[group] = member_keys

    if not overall:
        raise ValueError(
            "no lag-time bin has the required number of contributing sets; "
            "reduce min_sets or inspect the input curves"
        )

    return QuenchingStatisticsResult(
        per_set=per_set,
        overall=overall,
        group_members=group_members,
        min_sets=min_sets_value,
        within_set_weighting=weighting,
    )


def _quenching_fit_arrays(
    curve: Any,
    *,
    uncertainty: Optional[str],
) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[str]]:
    if isinstance(curve, QuenchingAverageResult):
        time = np.asarray(curve.lag_time_ps, dtype=np.float64)
        values = np.asarray(curve.survival_mean, dtype=np.float64)
        if uncertainty is None:
            sigma = None
            source = None
        else:
            normalized = str(uncertainty).strip().lower()
            if normalized in {"sem", "stderr", "standard_error"}:
                sigma = np.asarray(curve.survival_stderr, dtype=np.float64)
                source = "stderr"
            elif normalized in {"sd", "std", "standard_deviation"}:
                sigma = np.asarray(curve.survival_std, dtype=np.float64)
                source = "std"
            else:
                raise ValueError("uncertainty must be 'stderr', 'std', or None")
        return time, values, sigma, source

    array = np.asarray(curve, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] < 2:
        raise ValueError(
            "curve must be a QuenchingAverageResult or an array with at least "
            "columns time_ps and mean"
        )
    time = array[:, 0]
    values = array[:, 1]
    if uncertainty is None:
        return time, values, None, None
    if array.shape[1] < 3:
        raise ValueError("array input needs a third uncertainty column")
    return time, values, array[:, 2], "column_2"


def fit_quenching_decay(
    curve: Any,
    tau0_ps: Any = None,
    *,
    include_intrinsic_decay: bool = False,
    uncertainty: Optional[str] = "stderr",
    use_uncertainty_as_sigma: bool = True,
    enforce_zero_offset: bool = True,
    fit_range_ps: Optional[tuple[float, float]] = None,
    initial_k_q_per_s: float = 1.0e7,
    absolute_sigma: bool = True,
    maxfev: int = 20000,
) -> QuenchingFitResult:
    """Fit an averaged quenching curve without plotting.

    By default, ``include_intrinsic_decay=False`` because the simulated
    survival curve contains only distance-dependent quenching. The fitted model
    is therefore ``A*exp[-k_q*t] + C`` and ``tau0_ps`` is not required. In this
    mode ``k0_per_s`` is zero and ``k_obs_per_s`` is retained as a compatibility
    alias equal to ``k_q_per_s``.

    Set ``include_intrinsic_decay=True`` only when fitting data that already
    includes intrinsic fluorescence decay. The model then becomes
    ``A*exp[-(k0+k_q)t] + C``, where ``k0=1/tau0`` is fixed by ``tau0_ps``.
    With ``enforce_zero_offset=True``, ``C`` is fixed to zero.
    """
    try:
        from scipy.optimize import curve_fit
    except Exception as exc:
        raise ImportError("SciPy is required for quenching-curve fitting") from exc

    if not isinstance(include_intrinsic_decay, (bool, np.bool_)):
        raise TypeError("include_intrinsic_decay must be a boolean")

    if bool(include_intrinsic_decay):
        if tau0_ps is None:
            raise ValueError("tau0_ps is required when include_intrinsic_decay=True")
        tau0_value: Optional[float] = _time_value_to_ps(
            tau0_ps,
            name="tau0_ps",
        )
        if tau0_value <= 0.0:
            raise ValueError("tau0_ps must be > 0")
        k0 = 1.0 / (tau0_value * 1.0e-12)
    else:
        if tau0_ps is None:
            tau0_value = None
        else:
            tau0_value = _time_value_to_ps(tau0_ps, name="tau0_ps")
            if tau0_value <= 0.0:
                raise ValueError("tau0_ps must be > 0 when supplied")
        k0 = 0.0

    initial_kq = float(initial_k_q_per_s)
    if not math.isfinite(initial_kq) or initial_kq < 0.0:
        raise ValueError("initial_k_q_per_s must be finite and >= 0")
    if int(maxfev) < 1:
        raise ValueError("maxfev must be >= 1")

    requested_uncertainty = uncertainty if use_uncertainty_as_sigma else None
    time_ps, values, sigma, sigma_source = _quenching_fit_arrays(
        curve,
        uncertainty=requested_uncertainty,
    )
    mask = np.isfinite(time_ps) & np.isfinite(values) & (time_ps > 0.0)

    if fit_range_ps is not None:
        if len(fit_range_ps) != 2:
            raise ValueError("fit_range_ps must contain (minimum, maximum)")
        fit_min = float(fit_range_ps[0])
        fit_max = float(fit_range_ps[1])
        if not math.isfinite(fit_min) or not math.isfinite(fit_max) or fit_max <= fit_min:
            raise ValueError("fit_range_ps must be a finite increasing pair")
        mask &= (time_ps >= fit_min) & (time_ps <= fit_max)

    if sigma is not None:
        mask &= np.isfinite(sigma) & (sigma > 0.0)

    t_fit_ps = time_ps[mask]
    y_fit_data = values[mask]
    sigma_fit = None if sigma is None else sigma[mask]
    n_parameters = 2 if enforce_zero_offset else 3
    if t_fit_ps.size < n_parameters + 1:
        raise ValueError(
            f"fit requires at least {n_parameters + 1} valid points; " f"found {t_fit_ps.size}"
        )

    t_fit_s = t_fit_ps * 1.0e-12
    amplitude_guess = max(float(y_fit_data[0]), np.finfo(float).tiny)

    if enforce_zero_offset:

        def model(t_s: np.ndarray, k_q: float, amplitude: float) -> np.ndarray:
            return amplitude * np.exp(-(k0 + k_q) * t_s)

        p0 = (initial_kq, amplitude_guess)
        bounds = ([0.0, 0.0], [np.inf, np.inf])
    else:

        def model(
            t_s: np.ndarray,
            k_q: float,
            amplitude: float,
            offset: float,
        ) -> np.ndarray:
            return amplitude * np.exp(-(k0 + k_q) * t_s) + offset

        p0 = (initial_kq, amplitude_guess, 0.0)
        bounds = ([0.0, 0.0, 0.0], [np.inf, np.inf, np.inf])

    popt, covariance = curve_fit(
        model,
        t_fit_s,
        y_fit_data,
        p0=p0,
        sigma=sigma_fit,
        absolute_sigma=bool(absolute_sigma and sigma_fit is not None),
        bounds=bounds,
        maxfev=int(maxfev),
    )

    if enforce_zero_offset:
        k_q, amplitude = (float(popt[0]), float(popt[1]))
        offset = 0.0
    else:
        k_q, amplitude, offset = (float(popt[0]), float(popt[1]), float(popt[2]))

    k_obs = k0 + k_q
    fitted = amplitude * np.exp(-k_obs * t_fit_s) + offset
    parameter_stderr = np.sqrt(np.maximum(np.diag(covariance), 0.0))

    return QuenchingFitResult(
        tau0_ps=None if tau0_value is None else float(tau0_value),
        k0_per_s=float(k0),
        k_q_per_s=float(k_q),
        k_obs_per_s=float(k_obs),
        tau_obs_s=float(1.0 / k_obs),
        amplitude=float(amplitude),
        offset=float(offset),
        covariance=np.asarray(covariance, dtype=np.float64),
        parameter_stderr=np.asarray(parameter_stderr, dtype=np.float64),
        fit_time_ps=t_fit_ps.copy(),
        fitted_survival=np.asarray(fitted, dtype=np.float64),
        n_points=int(t_fit_ps.size),
        include_intrinsic_decay=bool(include_intrinsic_decay),
        enforce_zero_offset=bool(enforce_zero_offset),
        sigma_source=sigma_source,
    )


def _time_scale_from_ps(time_unit: str) -> tuple[float, str]:
    normalized = str(time_unit).strip().lower().replace("μ", "u").replace("µ", "u")
    scales = {
        "ps": (1.0, "ps"),
        "ns": (1.0e-3, "ns"),
        "us": (1.0e-6, "µs"),
        "ms": (1.0e-9, "ms"),
        "s": (1.0e-12, "s"),
    }
    if normalized not in scales:
        raise ValueError("time_unit must be one of 'ps', 'ns', 'us', 'ms', or 's'")
    return scales[normalized]


def _time_to_ps_scale(time_unit: str) -> float:
    normalized = str(time_unit).strip().lower().replace("μ", "u").replace("µ", "u")
    scales = {"ps": 1.0, "ns": 1.0e3, "us": 1.0e6, "ms": 1.0e9, "s": 1.0e12}
    if normalized not in scales:
        raise ValueError("external time_unit must be 'ps', 'ns', 'us', 'ms', or 's'")
    return scales[normalized]


def plot_quenching_statistics(
    statistics: Union[QuenchingStatisticsResult, QuenchingAverageResult],
    *,
    group: Optional[str] = None,
    fit_result: Optional[QuenchingFitResult] = None,
    tau0_ps: Any = None,
    apply_intrinsic_decay: bool = True,
    show_sets: bool = True,
    show_overall: bool = True,
    uncertainty: Optional[str] = "stderr",
    colors: Optional[Union[Mapping[str, Any], Sequence[Any]]] = None,
    labels: Optional[Union[Mapping[str, str], Sequence[str]]] = None,
    overall_label: str = "overall average",
    overall_color: Any = "black",
    fit_label: str = "fit",
    fit_color: Optional[Any] = None,
    set_linewidth: float = 1.5,
    overall_linewidth: float = 2.5,
    fit_linewidth: float = 2.0,
    fit_linestyle: str = "--",
    set_alpha: float = 1.0,
    uncertainty_alpha: float = 0.20,
    external_series: Optional[Union[Mapping[str, Any], Sequence[Mapping[str, Any]]]] = None,
    time_unit: str = "s",
    xscale: str = "log",
    yscale: str = "linear",
    xlabel: Optional[str] = None,
    ylabel: str = "survival probability",
    title: Optional[str] = None,
    xlim: Optional[tuple[float, float]] = None,
    ylim: Optional[tuple[float, float]] = None,
    ax: Optional[plt.Axes] = None,
    figsize: tuple[float, float] = (6.0, 4.0),
    legend: bool = True,
    grid: bool = True,
    save_path: Optional[PathLike] = None,
    save_dpi: Optional[float] = None,
    save_bbox_inches: Optional[str] = "tight",
    save_kwargs: Optional[Mapping[str, Any]] = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot set-level and overall quenching curves without performing a fit.

    Pass a separately calculated :class:`QuenchingFitResult` through
    ``fit_result`` to overlay it.

    The statistics and the recommended fit represent quenching-only survival
    from the simulations. When ``tau0_ps`` is supplied and
    ``apply_intrinsic_decay=True`` (the default), every simulated curve is
    converted to total fluorescence survival as::

        S_total(t) = S_quenching(t) * exp(-t/tau0)

    The transformation is applied to all set curves, the overall curve, its SD
    or SEM band, and a quenching-only fitted curve. External series are left
    unchanged because they are presumed to already represent the quantity the
    user intends to compare. If ``tau0_ps`` is omitted, no intrinsic decay is
    applied.
    """
    if not isinstance(apply_intrinsic_decay, (bool, np.bool_)):
        raise TypeError("apply_intrinsic_decay must be a boolean")
    if float(set_linewidth) < 0.0 or float(overall_linewidth) < 0.0 or float(fit_linewidth) < 0.0:
        raise ValueError("line widths must be >= 0")
    if not 0.0 <= float(set_alpha) <= 1.0:
        raise ValueError("set_alpha must be between 0 and 1")
    if not 0.0 <= float(uncertainty_alpha) <= 1.0:
        raise ValueError("uncertainty_alpha must be between 0 and 1")

    output_scale, output_time_label = _time_scale_from_ps(time_unit)

    if tau0_ps is None or not bool(apply_intrinsic_decay):
        tau0_value_ps: Optional[float] = None
    else:
        tau0_value_ps = _time_value_to_ps(tau0_ps, name="tau0_ps")
        if tau0_value_ps <= 0.0:
            raise ValueError("tau0_ps must be > 0")

    def intrinsic_factor(time_ps: np.ndarray) -> np.ndarray:
        time_array = np.asarray(time_ps, dtype=np.float64)
        if tau0_value_ps is None:
            return np.ones_like(time_array, dtype=np.float64)
        return np.exp(-time_array / float(tau0_value_ps))

    if isinstance(statistics, QuenchingAverageResult):
        if group is not None:
            raise ValueError("group is only used with QuenchingStatisticsResult")
        overall_curve = statistics
        set_items: list[tuple[str, QuenchingAverageResult]] = []
        selected_group = None
    elif isinstance(statistics, QuenchingStatisticsResult):
        if group is None:
            if len(statistics.overall) != 1:
                raise ValueError(
                    "group must be supplied when statistics contains multiple overall groups: "
                    f"{list(statistics.overall)}"
                )
            selected_group = next(iter(statistics.overall))
        else:
            selected_group = str(group)
            if selected_group not in statistics.overall:
                raise KeyError(
                    f"unknown quenching group {selected_group!r}; "
                    f"available groups: {list(statistics.overall)}"
                )
        overall_curve = statistics.overall[selected_group]
        member_keys = statistics.group_members[selected_group]
        set_items = [(key, statistics.per_set[key]) for key in member_keys]
    else:
        raise TypeError("statistics must be QuenchingStatisticsResult or QuenchingAverageResult")

    if ax is None:
        figure, ax = plt.subplots(figsize=figsize)
    else:
        figure = ax.figure

    def style_value(
        specification: Optional[Union[Mapping[str, Any], Sequence[Any]]],
        key: str,
        index: int,
        default: Any,
    ) -> Any:
        if specification is None:
            return default
        if isinstance(specification, Mapping):
            return specification.get(key, default)
        if (
            isinstance(specification, Sequence) or isinstance(specification, np.ndarray)
        ) and not isinstance(specification, (str, bytes)):
            return specification[index] if index < len(specification) else default
        raise TypeError("colors and labels must be mappings, sequences, or None")

    if show_sets:
        for index, (key, curve) in enumerate(set_items):
            time_ps = np.asarray(curve.lag_time_ps, dtype=np.float64)
            time_values = time_ps * output_scale
            plotted_survival = np.asarray(curve.survival_mean, dtype=np.float64) * intrinsic_factor(
                time_ps
            )
            mask = np.isfinite(time_values) & np.isfinite(plotted_survival)
            if str(xscale).lower() == "log":
                mask &= time_values > 0.0
            if not np.any(mask):
                continue
            kwargs: dict[str, Any] = {
                "linewidth": float(set_linewidth),
                "alpha": float(set_alpha),
                "label": style_value(labels, key, index, key),
            }
            color = style_value(colors, key, index, None)
            if color is not None:
                kwargs["color"] = color
            ax.plot(time_values[mask], plotted_survival[mask], **kwargs)

    if show_overall:
        overall_time_ps = np.asarray(overall_curve.lag_time_ps, dtype=np.float64)
        time_values = overall_time_ps * output_scale
        overall_factor = intrinsic_factor(overall_time_ps)
        plotted_overall = np.asarray(overall_curve.survival_mean, dtype=np.float64) * overall_factor
        mask = np.isfinite(time_values) & np.isfinite(plotted_overall)
        if str(xscale).lower() == "log":
            mask &= time_values > 0.0
        ax.plot(
            time_values[mask],
            plotted_overall[mask],
            color=overall_color,
            linewidth=float(overall_linewidth),
            label=overall_label,
            zorder=10,
        )

        if uncertainty is not None:
            normalized = str(uncertainty).strip().lower()
            if normalized in {"sem", "stderr", "standard_error"}:
                band = overall_curve.survival_stderr
                band_label = "±SEM"
            elif normalized in {"sd", "std", "standard_deviation"}:
                band = overall_curve.survival_std
                band_label = "±SD across sets"
            else:
                raise ValueError("uncertainty must be 'stderr', 'std', or None")
            plotted_band = np.asarray(band, dtype=np.float64) * overall_factor
            band_mask = mask & np.isfinite(plotted_band)
            if np.any(band_mask):
                ax.fill_between(
                    time_values[band_mask],
                    plotted_overall[band_mask] - plotted_band[band_mask],
                    plotted_overall[band_mask] + plotted_band[band_mask],
                    color=overall_color,
                    alpha=float(uncertainty_alpha),
                    linewidth=0.0,
                    label=band_label,
                )

    if fit_result is not None:
        fit_time_ps = np.asarray(fit_result.fit_time_ps, dtype=np.float64)
        fit_time = fit_time_ps * output_scale

        # Reconstruct exactly the model that was fitted. For the recommended
        # quenching-only fit, k_obs == k_q. A deterministic intrinsic survival
        # factor can then be applied for comparison with fluorescence data.
        fit_values = float(fit_result.amplitude) * np.exp(
            -float(fit_result.k_obs_per_s) * fit_time_ps * 1.0e-12
        ) + float(fit_result.offset)
        if tau0_value_ps is not None:
            if bool(fit_result.include_intrinsic_decay):
                raise ValueError(
                    "fit_result already includes intrinsic decay; do not also "
                    "supply tau0_ps to plot_quenching_statistics"
                )
            fit_values = fit_values * intrinsic_factor(fit_time_ps)
        mask = np.isfinite(fit_time) & np.isfinite(fit_values)
        if str(xscale).lower() == "log":
            mask &= fit_time > 0.0
        fit_kwargs: dict[str, Any] = {
            "linestyle": fit_linestyle,
            "linewidth": float(fit_linewidth),
            "label": fit_label,
            "zorder": 11,
        }
        if fit_color is not None:
            fit_kwargs["color"] = fit_color
        ax.plot(fit_time[mask], fit_values[mask], **fit_kwargs)

    if external_series is not None:
        datasets = (
            [external_series] if isinstance(external_series, Mapping) else list(external_series)
        )
        for dataset_index, dataset in enumerate(datasets):
            if "t" not in dataset or "series" not in dataset:
                raise KeyError("each external dataset requires 't' and 'series'")
            external_time = np.asarray(dataset["t"], dtype=np.float64)
            source_unit = str(dataset.get("time_unit", "s"))
            external_x = external_time * _time_to_ps_scale(source_unit) * output_scale
            series_values = dataset["series"]
            if isinstance(series_values, np.ndarray) and series_values.ndim == 1:
                series_list = [series_values]
            else:
                series_list = list(series_values)
            base_label = dataset.get("label", "external")
            for series_index, values in enumerate(series_list):
                y = np.asarray(values, dtype=np.float64)
                if y.shape != external_x.shape:
                    raise ValueError("external time and series arrays must have the same shape")
                mask = np.isfinite(external_x) & np.isfinite(y)
                if str(xscale).lower() == "log":
                    mask &= external_x > 0.0
                label = (
                    str(base_label) if len(series_list) == 1 else f"{base_label} {series_index + 1}"
                )
                kwargs = {
                    "linewidth": float(dataset.get("linewidth", 1.0)),
                    "linestyle": dataset.get("linestyle", "-"),
                    "label": label,
                }
                if dataset.get("color") is not None:
                    kwargs["color"] = dataset["color"]
                ax.plot(external_x[mask], y[mask], **kwargs)

    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_xlabel(xlabel or f"time ({output_time_label})")
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    elif selected_group is not None:
        ax.set_title(str(selected_group))
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if grid:
        ax.grid(True)
    if legend:
        handles, legend_labels = ax.get_legend_handles_labels()
        if any(label and not str(label).startswith("_") for label in legend_labels):
            ax.legend()

    _save_figure_if_requested(
        figure,
        save_path,
        save_dpi=save_dpi,
        save_bbox_inches=save_bbox_inches,
        save_kwargs=save_kwargs,
    )
    return figure, ax


def load_quenching_time_series_csv(
    path: PathLike,
    *,
    label: Optional[str] = None,
    color: Optional[Any] = None,
    time_unit: str = "s",
    delimiter: str = ",",
    comments: str = "#",
) -> dict[str, Any]:
    """Load one time column and one or more external series for plotting."""
    _time_to_ps_scale(time_unit)  # validate now rather than during plotting
    array = np.loadtxt(path, delimiter=delimiter, comments=comments)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[1] < 2:
        raise ValueError(f"{path!r} must contain time plus at least one series")
    return {
        "t": array[:, 0].astype(np.float64),
        "series": [array[:, index].astype(np.float64) for index in range(1, array.shape[1])],
        "label": label,
        "color": color,
        "time_unit": time_unit,
    }


def load_multiple_quenching_time_series_csv(
    paths: Sequence[PathLike],
    *,
    labels: Optional[Sequence[Optional[str]]] = None,
    colors: Optional[Sequence[Any]] = None,
    time_unit: str = "s",
    delimiter: str = ",",
    comments: str = "#",
) -> list[dict[str, Any]]:
    """Load several external quenching time-series CSV files."""
    path_list = [paths] if isinstance(paths, (str, Path)) else list(paths)
    label_list = [None] * len(path_list) if labels is None else list(labels)
    color_list = [None] * len(path_list) if colors is None else list(colors)
    if len(label_list) < len(path_list):
        label_list.extend([None] * (len(path_list) - len(label_list)))
    if len(color_list) < len(path_list):
        color_list.extend([None] * (len(path_list) - len(color_list)))
    return [
        load_quenching_time_series_csv(
            path,
            label=label_list[index],
            color=color_list[index],
            time_unit=time_unit,
            delimiter=delimiter,
            comments=comments,
        )
        for index, path in enumerate(path_list)
    ]


# --- all-residue distance-dependent diffusion maps ----------------------------


def fit_dist_diffusion_maps(
    data_by_set: Mapping[str, DistMSDMapResult],
    *,
    fit_tmin_ns: float,
    fit_tmax_ns: float,
    dims: int = 1,
    min_sets: int = 1,
    average_over_sets: bool = True,
    group_by: Optional[Mapping[str, str]] = None,
) -> dict[str, DistDiffusionMapFitResult]:
    """Fit all-residue MSD maps and optionally average D maps over simulation sets.

    ``data_by_set`` is normally returned by ``get_values(..., calc='distmsdmap')``.
    Each simulation set is fitted independently. With ``average_over_sets=True``,
    keys are grouped using the same ``<set>_<group>`` convention as the other
    notebook analysis helpers and the final D values are equal-weight means across
    independent sets. ``min_sets`` specifies the minimum number of finite set-level
    fits required for each map cell.
    """
    if not data_by_set:
        raise ValueError("data_by_set is empty")

    keys = sorted(data_by_set, key=_dataset_sort_key)
    if not average_over_sets:
        return {
            str(key): fit_dist_msd_map_linear(
                data_by_set[key],
                fit_tmin_ns=fit_tmin_ns,
                fit_tmax_ns=fit_tmax_ns,
                dims=dims,
                min_sets=1,
            )
            for key in keys
        }

    output: dict[str, DistDiffusionMapFitResult] = {}
    groups = _ordered_set_groups(keys, group_by=group_by)
    for group, member_keys in groups.items():
        subset = {str(key): data_by_set[key] for key in member_keys}
        output[str(group)] = fit_dist_msd_map_linear(
            subset,
            fit_tmin_ns=fit_tmin_ns,
            fit_tmax_ns=fit_tmax_ns,
            dims=dims,
            min_sets=min_sets,
        )
    return output


def plot_dist_diffusion_map(
    result: DistDiffusionMapFitResult,
    *,
    component: str = "combined",
    flip_triangles: bool = False,
    cmap: Any = "viridis",
    color_scale: str = "linear",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    kinetic_scale: Optional[float] = 1.0,
    colorbar: bool = True,
    colorbar_label: Optional[str] = None,
    annotate_triangles: bool = True,
    tick_step: Optional[int] = None,
    xlabel: str = "partner residue",
    ylabel: str = "query residue",
    title: Optional[str] = None,
    figsize: tuple[float, float] = (7.0, 6.0),
    ax: Optional[plt.Axes] = None,
    save_path: Optional[PathLike] = None,
    save_dpi: Optional[float] = None,
    save_bbox_inches: Optional[str] = "tight",
    save_kwargs: Optional[Mapping[str, Any]] = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot fitted all-residue D values like a triangular contact map.

    ``component='combined'`` places intrachain D in the lower triangle and
    interchain D in the upper triangle including the diagonal. Set
    ``flip_triangles=True`` to swap their positions (interchain lower,
    intrachain upper), while retaining the interchain diagonal. ``'intra'``
    and ``'inter'`` show their respective sparse triangles separately and are
    not affected by ``flip_triangles``.
    """
    name = str(component).strip().lower().replace("-", "_")
    aliases = {
        "combined": "combined",
        "both": "combined",
        "intra": "intra",
        "intrachain": "intra",
        "inter": "inter",
        "interchain": "inter",
    }
    if name not in aliases:
        raise ValueError("component must be 'combined', 'intra', or 'inter'")
    name = aliases[name]

    if not isinstance(flip_triangles, (bool, np.bool_)):
        raise TypeError("flip_triangles must be a boolean")

    if name == "combined":
        values = np.asarray(result.combined_d_nm2_per_ns, dtype=float)
        if flip_triangles:
            values = values.T  # reflect the two triangles across the diagonal
    elif name == "intra":
        values = np.asarray(result.intra_d_nm2_per_ns, dtype=float)
    else:
        values = np.asarray(result.inter_d_nm2_per_ns, dtype=float)

    values = values / kinetic_scale

    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("diffusion map must be a square two-dimensional array")

    scale = str(color_scale).strip().lower()
    if scale not in {"linear", "log"}:
        raise ValueError("color_scale must be 'linear' or 'log'")

    masked = np.ma.masked_invalid(values)
    norm = None
    if scale == "log":
        from matplotlib.colors import LogNorm

        masked = np.ma.masked_less_equal(masked, 0.0)
        positive = values[np.isfinite(values) & (values > 0.0)]
        if positive.size == 0:
            raise ValueError("the selected diffusion map contains no positive values")
        lower = float(np.min(positive)) if vmin is None else float(vmin)
        upper = float(np.max(positive)) if vmax is None else float(vmax)
        if lower <= 0.0 or upper <= lower:
            raise ValueError("log color limits require 0 < vmin < vmax")
        norm = LogNorm(vmin=lower, vmax=upper)

    if ax is None:
        figure, ax = plt.subplots(figsize=figsize)
    else:
        figure = ax.figure

    cmap_object = plt.get_cmap(cmap).copy()
    cmap_object.set_bad("white")
    kwargs: dict[str, Any] = {
        "origin": "upper",
        "interpolation": "nearest",
        "aspect": "equal",
        "cmap": cmap_object,
    }
    if norm is None:
        kwargs["vmin"] = vmin
        kwargs["vmax"] = vmax
    else:
        kwargs["norm"] = norm
    image = ax.imshow(masked, **kwargs)

    residues = np.asarray(result.residue_numbers, dtype=np.int64)
    n_res = int(residues.size)
    if n_res != values.shape[0]:
        raise ValueError("residue-number metadata does not match diffusion-map size")

    if tick_step is None:
        step = 5 if n_res <= 50 else 10
    else:
        if isinstance(tick_step, (bool, np.bool_)):
            raise TypeError("tick_step must be an integer >=1 or None")
        step_float = float(tick_step)
        if not math.isfinite(step_float) or not step_float.is_integer() or step_float < 1:
            raise ValueError("tick_step must be an integer >=1 or None")
        step = int(step_float)

    positions = [0]
    positions.extend(range(step - 1, n_res, step))
    positions = list(dict.fromkeys(int(x) for x in positions if 0 <= int(x) < n_res))
    labels = [str(int(residues[p])) for p in positions]
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if title is None:
        if result.distance_min_nm is None and result.distance_max_nm is None:
            window = "all initial distances"
        else:
            lower = "-inf" if result.distance_min_nm is None else f"{result.distance_min_nm:g}"
            upper = "inf" if result.distance_max_nm is None else f"{result.distance_max_nm:g}"
            window = f"r0={lower}-{upper} nm"
        if name == "combined":
            title = f"D map ({result.atom_name}; {window})"
        elif name == "intra":
            title = f"intrachain D map ({result.atom_name}; {window})"
        else:
            title = f"interchain D map ({result.atom_name}; {window})"
    ax.set_title(title)

    if name == "combined" and annotate_triangles:
        ax.text(
            0.04,
            0.08,
            "interchain" if flip_triangles else "intrachain",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            color="white",
            bbox={"facecolor": "black", "alpha": 0.45, "edgecolor": "none", "pad": 2.0},
        )
        ax.text(
            0.96,
            0.92,
            "intrachain" if flip_triangles else "interchain",
            transform=ax.transAxes,
            ha="right",
            va="top",
            color="white",
            bbox={"facecolor": "black", "alpha": 0.45, "edgecolor": "none", "pad": 2.0},
        )

    if colorbar:
        cbar = figure.colorbar(image, ax=ax)
        cbar.set_label(colorbar_label or result.units)

    _save_figure_if_requested(
        figure,
        save_path,
        save_dpi=save_dpi,
        save_bbox_inches=save_bbox_inches,
        save_kwargs=save_kwargs,
    )
    return figure, ax


def plot_contact_diffusion_triangles(
    diffusion_result: DistDiffusionMapFitResult,
    contact_result: ResidueContactMapResult,
    *,
    diffusion_component: str = "combined",
    contact_component: str = "combined",
    diffusion_cmap: Any = "viridis",
    contact_cmap: Any = "magma",
    diffusion_color_scale: str = "linear",
    contact_color_scale: str = "linear",
    diffusion_vmin: Optional[float] = None,
    diffusion_vmax: Optional[float] = None,
    contact_vmin: Optional[float] = None,
    contact_vmax: Optional[float] = None,
    kinetic_scale: Optional[float] = 1.0,
    diffusion_colorbar: bool = True,
    contact_colorbar: bool = True,
    diffusion_colorbar_label: Optional[str] = None,
    contact_colorbar_label: Optional[str] = None,
    annotate_triangles: bool = True,
    tick_step: Optional[int] = None,
    xlabel: str = "residue",
    ylabel: str = "residue",
    title: Optional[str] = None,
    figsize: tuple[float, float] = (8.0, 7.0),
    ax: Optional[plt.Axes] = None,
    save_path: Optional[PathLike] = None,
    save_dpi: Optional[float] = None,
    save_bbox_inches: Optional[str] = "tight",
    save_kwargs: Optional[Mapping[str, Any]] = None,
) -> tuple[plt.Figure, plt.Axes, Any, Any]:
    """Plot diffusion in the lower triangle and contacts in the upper triangle.

    Parameters
    ----------
    diffusion_result
        Result object produced by ``fit_dist_diffusion_maps``.
    contact_result
        Result object produced by ``calculate_contact_maps`` or compatible code.
    diffusion_component, contact_component
        Component names to extract before combining. Each can be ``'combined'``,
        ``'intra'``, or ``'inter'``. The selected arrays must be square and of
        matching size.

    Returns
    -------
    figure, ax, diffusion_cbar, contact_cbar
        The Matplotlib figure/axes and the left/right colorbars. A colorbar
        entry is ``None`` if that colorbar was disabled.
    """
    from matplotlib.colors import LogNorm, Normalize
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    def _normalize_component(value: Any, *, kind: str) -> str:
        if not isinstance(value, str):
            raise TypeError(f"{kind}_component must be 'combined', 'intra', or 'inter'")
        name = value.strip().lower().replace("-", "_")
        aliases = {
            "combined": "combined",
            "both": "combined",
            "intra": "intra",
            "intrachain": "intra",
            "inter": "inter",
            "interchain": "inter",
        }
        if name not in aliases:
            raise ValueError(f"{kind}_component must be 'combined', 'intra', or 'inter'")
        return aliases[name]

    def _extract_diffusion_values(
        result: DistDiffusionMapFitResult, component: str
    ) -> tuple[np.ndarray, np.ndarray, str]:
        name = _normalize_component(component, kind="diffusion")
        if name == "combined":
            values = np.asarray(result.combined_d_nm2_per_ns, dtype=float)
        elif name == "intra":
            values = np.asarray(result.intra_d_nm2_per_ns, dtype=float)
        else:
            values = np.asarray(result.inter_d_nm2_per_ns, dtype=float)
        residues = np.asarray(result.residue_numbers, dtype=np.int64)
        units = getattr(result, "units", "D (nm$^2$/ns)")
        return values, residues, units

    def _extract_contact_values(
        result: ResidueContactMapResult, component: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
        name = _normalize_component(component, kind="contact")
        if name == "combined":
            if result.combined is None:
                raise ValueError(
                    "contact_component='combined' requires a square homotypic contact map"
                )
            values = np.asarray(result.combined, dtype=float)
        elif name == "intra":
            values = np.asarray(result.intra, dtype=float)
        else:
            values = np.asarray(result.inter, dtype=float)
        qres = np.asarray(result.query_residue_numbers, dtype=np.int64)
        pres = np.asarray(result.partner_residue_numbers, dtype=np.int64)
        units = getattr(result, "units", "contact probability")
        return values, qres, pres, units

    def _build_norm(
        values_1d: np.ndarray,
        *,
        scale: str,
        vmin: Optional[float],
        vmax: Optional[float],
        label: str,
    ):
        scale_name = str(scale).strip().lower()
        if scale_name not in {"linear", "log"}:
            raise ValueError(f"{label}_color_scale must be 'linear' or 'log'")
        finite = np.asarray(values_1d, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            raise ValueError(f"{label} triangle contains no finite values")
        if scale_name == "log":
            positive = finite[finite > 0.0]
            if positive.size == 0:
                raise ValueError(
                    f"{label} triangle contains no positive values for log color scale"
                )
            lower = float(np.min(positive)) if vmin is None else float(vmin)
            upper = float(np.max(positive)) if vmax is None else float(vmax)
            if lower <= 0.0 or upper <= lower:
                raise ValueError(f"{label} log color limits require 0 < vmin < vmax")
            return LogNorm(vmin=lower, vmax=upper), scale_name
        lower = None if vmin is None else float(vmin)
        upper = None if vmax is None else float(vmax)
        if lower is None:
            lower = float(np.min(finite))
        if upper is None:
            upper = float(np.max(finite))
        if upper < lower:
            raise ValueError(f"{label} requires vmax >= vmin")
        return Normalize(vmin=lower, vmax=upper), scale_name

    diff_values, diff_residues, diff_units = _extract_diffusion_values(
        diffusion_result, diffusion_component
    )
    contact_values, contact_qres, contact_pres, contact_units = _extract_contact_values(
        contact_result, contact_component
    )

    if diff_values.ndim != 2 or diff_values.shape[0] != diff_values.shape[1]:
        raise ValueError("selected diffusion component must be a square two-dimensional array")
    if contact_values.ndim != 2 or contact_values.shape[0] != contact_values.shape[1]:
        raise ValueError("selected contact component must be a square two-dimensional array")
    if diff_values.shape != contact_values.shape:
        raise ValueError("selected diffusion and contact arrays must have the same square shape")

    diff_values = diff_values / kinetic_scale

    n_res = int(diff_residues.size)
    if n_res != diff_values.shape[0]:
        raise ValueError("diffusion residue-number metadata does not match map size")
    if contact_qres.size != n_res or contact_pres.size != n_res:
        raise ValueError("contact residue-number metadata does not match map size")
    if not np.array_equal(contact_qres, contact_pres):
        raise ValueError(
            "selected contact component must correspond to a square homotypic residue map"
        )
    if not np.array_equal(diff_residues, contact_qres):
        raise ValueError(
            "diffusion and contact residue-number arrays must match to form a shared plot"
        )

    lower_rows, lower_cols = np.tril_indices(n_res, k=-1)
    upper_rows, upper_cols = np.triu_indices(n_res, k=0)

    diff_triangle_values = diff_values[lower_rows, lower_cols]
    contact_triangle_values = contact_values[upper_rows, upper_cols]

    diffusion_norm, diffusion_scale_name = _build_norm(
        diff_triangle_values,
        scale=diffusion_color_scale,
        vmin=diffusion_vmin,
        vmax=diffusion_vmax,
        label="diffusion",
    )
    contact_norm, contact_scale_name = _build_norm(
        contact_triangle_values,
        scale=contact_color_scale,
        vmin=contact_vmin,
        vmax=contact_vmax,
        label="contact",
    )

    diff_display = np.full_like(diff_values, np.nan, dtype=float)
    contact_display = np.full_like(contact_values, np.nan, dtype=float)
    diff_display[lower_rows, lower_cols] = diff_triangle_values
    contact_display[upper_rows, upper_cols] = contact_triangle_values

    if diffusion_scale_name == "log":
        diff_masked = np.ma.masked_invalid(diff_display)
        diff_masked = np.ma.masked_less_equal(diff_masked, 0.0)
    else:
        diff_masked = np.ma.masked_invalid(diff_display)
    if contact_scale_name == "log":
        contact_masked = np.ma.masked_invalid(contact_display)
        contact_masked = np.ma.masked_less_equal(contact_masked, 0.0)
    else:
        contact_masked = np.ma.masked_invalid(contact_display)

    if ax is None:
        figure, ax = plt.subplots(figsize=figsize)
    else:
        figure = ax.figure

    diffusion_cmap_object = plt.get_cmap(diffusion_cmap).copy()
    diffusion_cmap_object.set_bad((1.0, 1.0, 1.0, 0.0))
    contact_cmap_object = plt.get_cmap(contact_cmap).copy()
    contact_cmap_object.set_bad((1.0, 1.0, 1.0, 0.0))

    diff_image = ax.imshow(
        diff_masked,
        origin="upper",
        interpolation="nearest",
        aspect="equal",
        cmap=diffusion_cmap_object,
        norm=diffusion_norm,
    )
    contact_image = ax.imshow(
        contact_masked,
        origin="upper",
        interpolation="nearest",
        aspect="equal",
        cmap=contact_cmap_object,
        norm=contact_norm,
    )

    if tick_step is None:
        step = 5 if n_res <= 50 else 10
    else:
        if isinstance(tick_step, (bool, np.bool_)):
            raise TypeError("tick_step must be an integer >= 1 or None")
        step_float = float(tick_step)
        if not math.isfinite(step_float) or not step_float.is_integer() or step_float < 1:
            raise ValueError("tick_step must be an integer >= 1 or None")
        step = int(step_float)

    positions = [0]
    positions.extend(range(step - 1, n_res, step))
    positions = list(dict.fromkeys(int(x) for x in positions if 0 <= int(x) < n_res))
    labels = [str(int(diff_residues[p])) for p in positions]
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if title is None:
        title = "diffusion map (lower) and contact map (upper)"
    ax.set_title(title)

    if annotate_triangles:
        ax.text(
            0.04,
            0.08,
            "diffusion",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            color="white",
            bbox={"facecolor": "black", "alpha": 0.45, "edgecolor": "none", "pad": 2.0},
        )
        ax.text(
            0.96,
            0.92,
            "contacts",
            transform=ax.transAxes,
            ha="right",
            va="top",
            color="white",
            bbox={"facecolor": "black", "alpha": 0.45, "edgecolor": "none", "pad": 2.0},
        )

    divider = make_axes_locatable(ax)
    diffusion_cbar_obj = None
    contact_cbar_obj = None

    if diffusion_colorbar:
        left_cax = divider.append_axes("left", size="4.5%", pad=0.50)
        diffusion_cbar_obj = figure.colorbar(diff_image, cax=left_cax)
        diffusion_cbar_obj.set_label(diffusion_colorbar_label or diff_units)
        diffusion_cbar_obj.ax.yaxis.set_ticks_position("left")
        diffusion_cbar_obj.ax.yaxis.set_label_position("left")

    if contact_colorbar:
        right_cax = divider.append_axes("right", size="4.5%", pad=0.12)
        contact_cbar_obj = figure.colorbar(contact_image, cax=right_cax)
        contact_cbar_obj.set_label(contact_colorbar_label or contact_units)

    _save_figure_if_requested(
        figure,
        save_path,
        save_dpi=save_dpi,
        save_bbox_inches=save_bbox_inches,
        save_kwargs=save_kwargs,
    )
    return figure, ax, diffusion_cbar_obj, contact_cbar_obj


# --- Notebook-facing translational diffusion analysis ---------------------------


def calculate_translational_msd(
    trajectories: Any,
    *,
    max_lag_frames: Optional[int] = 200,
    n_jobs: int = 1,
    block_chains: int = 16,
) -> Any:
    """MSDs for a cached ``get_values(..., calc='comtraj')`` result or mapping."""
    if isinstance(trajectories, Mapping) and not hasattr(trajectories, "position_per_chain_nm"):
        return {
            key: calculate_translational_msd(
                value, max_lag_frames=max_lag_frames, n_jobs=n_jobs, block_chains=block_chains
            )
            for key, value in trajectories.items()
        }
    return translational_msd_from_trajectories(
        trajectories,
        max_lag_frames=max_lag_frames,
        n_jobs=n_jobs,
        block_chains=block_chains,
    )


def calculate_radial_translational_msd(
    trajectories: Any,
    *,
    bins: Union[int, Sequence[float]] = 10,
    value_range: Optional[tuple[float, float]] = (0, 30),
    max_lag_frames: int = 100,
    origin_stride: int = 1,
    min_origins_per_chain: int = 5,
) -> Any:
    """Initial-radius-conditioned MSDs, preserving dataset mapping keys."""
    if isinstance(trajectories, Mapping) and not hasattr(trajectories, "position_per_chain_nm"):
        return {
            key: calculate_radial_translational_msd(
                value,
                bins=bins,
                value_range=value_range,
                max_lag_frames=max_lag_frames,
                origin_stride=origin_stride,
                min_origins_per_chain=min_origins_per_chain,
            )
            for key, value in trajectories.items()
        }
    return radial_translational_msd_from_trajectories(
        trajectories,
        bins=bins,
        value_range=value_range,
        max_lag_frames=max_lag_frames,
        origin_stride=origin_stride,
        min_origins_per_chain=min_origins_per_chain,
    )


@dataclass(frozen=True)
class TranslationalDiffusionSummary:
    mean: float
    sem: float
    n_sets: int
    per_set_d_nm2_per_ns: np.ndarray
    corrected: bool


def summarize_translational_diffusion(
    fits: Mapping[str, TranslationalDiffusionFitResult],
    *,
    corrected: bool = False,
) -> TranslationalDiffusionSummary:
    """Equal-replicate-set mean and SEM from fits, not pooled-chain SEM."""
    if not fits:
        raise ValueError("fits is empty")
    field = "d_corrected_nm2_per_ns" if corrected else "d_nm2_per_ns"
    numbers = [getattr(fit, field) for fit in fits.values()]
    if any(v is None for v in numbers):
        raise ValueError("corrected=True requires Yeh--Hummer corrections for all sets")
    values = np.asarray(numbers, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("all set diffusion values must be finite")
    sem = float(np.std(values, ddof=1) / np.sqrt(values.size)) if values.size > 1 else float("nan")
    return TranslationalDiffusionSummary(
        mean=float(np.mean(values)),
        sem=sem,
        n_sets=len(values),
        per_set_d_nm2_per_ns=values,
        corrected=bool(corrected),
    )


# --- residue-resolved radial density summaries / maps ------------------------


@dataclass(frozen=True)
class ResidueDensityMapResult:
    """Residue-by-radius number-density map; rows follow residue_numbers."""

    density_nm3: np.ndarray  # (n_residues, n_bins)
    stderr_nm3: np.ndarray  # across independent sets; NaN for one set
    edges_nm: np.ndarray
    r_nm: np.ndarray
    residue_numbers: np.ndarray
    residue_names: tuple[str, ...]
    normalization: str  # 'total' or 'per_chain'
    n_frames: int  # summed over sets if averaging
    n_selected_chain_frames: int
    n_sets: int = 1
    set_labels: tuple[str, ...] = ()
    per_set_density_nm3: Optional[np.ndarray] = None
    units: str = "nm^-3"
    residue_site: str = "CA"  # sampled site: CA, COG, COM, ...


def _residue_density_single_map(
    result: ResidueDensityProfileResult,
    *,
    label: str,
    include_chains: Optional[tuple[str, ...]],
    exclude_chains: Optional[tuple[str, ...]],
    frame_start: Optional[int],
    frame_stop: Optional[int],
    time_start: Any,
    time_stop: Any,
    delta_t: Any,
    filter_result: Any,
    filter_prop: str,
    filter_min: Optional[float],
    filter_max: Optional[float],
    normalization: str,
) -> ResidueDensityMapResult:
    indices = np.asarray(result.bin_index_per_frame)
    if indices.shape != (result.n_frames, result.n_chains, result.n_residues):
        raise ValueError(f"{label}: invalid cached bin_index_per_frame shape")
    labels = tuple(result.chain_labels)
    if len(set(labels)) != len(labels):
        raise ValueError(f"{label}: duplicate cached chain labels")
    if include_chains is not None and exclude_chains is not None:
        raise ValueError("include_chains and exclude_chains are mutually exclusive")
    if include_chains is not None:
        missing = set(include_chains) - set(labels)
        if missing:
            raise ValueError(f"{label}: unknown included chains: {sorted(missing)}")
        static_chains = np.asarray([s in include_chains for s in labels], dtype=bool)
    elif exclude_chains is not None:
        missing = set(exclude_chains) - set(labels)
        if missing:
            raise ValueError(f"{label}: unknown excluded chains: {sorted(missing)}")
        static_chains = np.asarray([s not in exclude_chains for s in labels], dtype=bool)
    else:
        static_chains = np.ones(result.n_chains, dtype=bool)
    if not np.any(static_chains):
        raise ValueError(f"{label}: no query chains remain")

    uses_frames = frame_start is not None or frame_stop is not None
    uses_time = time_start is not None or time_stop is not None
    if uses_frames and uses_time:
        raise ValueError("specify either frame_start/stop or time_start/stop")
    selected_frames = np.ones(result.n_frames, dtype=bool)
    if uses_frames:
        start, stop = _resolve_frame_slice(
            result.n_frames,
            frame_start=frame_start,
            frame_stop=frame_stop,
        )
        selected_frames[:] = False
        selected_frames[start:stop] = True
    elif uses_time:
        if delta_t is None:
            raise ValueError("delta_t is required for time filtering")
        dt = _delta_t_to_ns(delta_t)
        times = np.asarray(result.source_frame_indices, dtype=float) * dt
        if time_start is not None:
            selected_frames &= times >= _time_value_to_ns(time_start, name="time_start")
        if time_stop is not None:
            selected_frames &= times < _time_value_to_ns(time_stop, name="time_stop")
    if not np.any(selected_frames):
        raise ValueError(f"{label}: no frames selected")

    # Each row of the filter is a physical chain; align by explicit chain label.
    chain_frame_keep = np.broadcast_to(
        static_chains[None, :], (result.n_frames, result.n_chains)
    ).copy()
    if filter_result is not None:
        values = np.asarray(_result_field(filter_result, filter_prop), dtype=float)
        if values.ndim != 2:
            raise ValueError(f"{label}: filter property must be (chains, frames)")
        filter_labels = tuple(str(x) for x in _result_field(filter_result, "chain_labels"))
        if len(filter_labels) != values.shape[0] or len(set(filter_labels)) != len(filter_labels):
            raise ValueError(f"{label}: invalid filter chain_labels")
        lookup = {s: i for i, s in enumerate(filter_labels)}
        missing = set(labels) - set(lookup)
        if missing:
            raise ValueError(f"{label}: filter missing chains {sorted(missing)}")
        values = values[[lookup[s] for s in labels], :]
        if values.shape[1] != result.n_frames:
            source = np.asarray(result.source_frame_indices, dtype=np.int64)
            if source.shape != (result.n_frames,) or np.any(source < 0):
                raise ValueError(f"{label}: invalid source frame indices")
            if source.size != np.unique(source).size:
                raise ValueError(
                    f"{label}: source frames repeat across multiple DCDs; "
                    "supply a filter already aligned to the cached frames"
                )
            if values.shape[1] <= int(source.max()):
                raise ValueError(f"{label}: filter does not cover the cached source frames")
            values = values[:, source]
        keep = np.isfinite(values)
        if filter_min is not None:
            keep &= values >= filter_min
        if filter_max is not None:
            keep &= values <= filter_max
        chain_frame_keep &= keep.T

    keep = chain_frame_keep[selected_frames, :]
    binned = indices[selected_frames, :, :]
    chain_frames = int(np.sum(keep))
    if chain_frames == 0:
        raise ValueError(f"{label}: no chain-frame observations satisfy the filter")
    counts = np.zeros((result.n_residues, result.n_bins), dtype=np.int64)
    for ri in range(result.n_residues):
        res_bins = binned[:, :, ri]
        valid = keep & (res_bins >= 0) & (res_bins < result.n_bins)
        counts[ri, :] = np.bincount(res_bins[valid], minlength=result.n_bins)
    denom = int(np.sum(selected_frames)) if normalization == "total" else chain_frames
    densities = counts / (denom * np.asarray(result.shell_volumes_nm3)[None, :])
    return ResidueDensityMapResult(
        density_nm3=densities,
        stderr_nm3=np.full_like(densities, np.nan),
        edges_nm=np.asarray(result.edges_nm).copy(),
        r_nm=np.asarray(result.r_nm).copy(),
        residue_numbers=np.asarray(result.residue_numbers).copy(),
        residue_names=tuple(result.residue_names),
        normalization=normalization,
        n_frames=int(np.sum(selected_frames)),
        n_selected_chain_frames=chain_frames,
        set_labels=(str(label),),
        per_set_density_nm3=densities[None, :, :],
        residue_site=str(result.atom_name),
    )


def calculate_residue_density_maps(
    data: Union[ResidueDensityProfileResult, Mapping[str, ResidueDensityProfileResult]],
    *,
    include_chains: Any = None,
    exclude_chains: Any = None,
    filter_by: Optional[Mapping[str, Any]] = None,
    filter_prop: str = "distance_per_chain_nm",
    filter_min: Optional[float] = None,
    filter_max: Optional[float] = None,
    normalization: str = "total",
    average_over_sets: bool = False,
    group_by: Optional[Mapping[str, str]] = None,
    frame_start: Optional[int] = None,
    frame_stop: Optional[int] = None,
    time_start: Any = None,
    time_stop: Any = None,
    delta_t: Any = None,
) -> Union[ResidueDensityMapResult, dict[str, ResidueDensityMapResult]]:
    """Rebuild residue x radius densities from compact cached bin indices.

    By default ``normalization='total'`` means selected residue sites per nm^3
    (total among all selected chains; zero contribution from excluded chains).
    ``normalization='per_chain'`` instead divides by the number of qualifying
    chain-frame observations; the integral per residue is <= 1.

    ``filter_by`` uses the same per-chain distance mapping as other notebook
    analyses: e.g. the result of ``get_values(..., calc='centerdist')``. The
    filter is applied *frame by frame* to each chain, and never changes the
    condensate center that was used to build the cache.

    ``average_over_sets=True`` assigns equal statistical weight to independent
    simulation sets, with stderr across set-level maps (NaN with only one set).
    Sets are matched by residue *sequence position* when their residue counts
    agree. Residue names need not match; inconsistent names are reported as an
    empty string in the aggregate map. If numbering differs across sets, the
    aggregate's residue_numbers use 1..N positional indices. Radial bin edges
    and residue_site must still match exactly.
    """
    norm = str(normalization).strip().lower()
    if norm not in {"total", "per_chain"}:
        raise ValueError("normalization must be 'total' or 'per_chain'")
    if include_chains is not None and exclude_chains is not None:
        raise ValueError("include_chains and exclude_chains are mutually exclusive")
    _validate_filter_options(
        filter_by,
        filter_min=filter_min,
        filter_max=filter_max,
        filter_prop=filter_prop if filter_by is not None else None,
    )
    if not isinstance(data, Mapping):
        if average_over_sets:
            raise ValueError("average_over_sets needs a mapping of sets")
        if filter_by is not None:
            raise ValueError("for single-result input, supply data and filter_by as keyed mappings")
        if isinstance(include_chains, Mapping) or isinstance(exclude_chains, Mapping):
            raise ValueError("single-result chain selections must be label sequences")
        return _residue_density_single_map(
            data,
            label="data",
            include_chains=_diffusion_chain_selection(
                include_chains, key="data", name="include_chains"
            ),
            exclude_chains=_diffusion_chain_selection(
                exclude_chains, key="data", name="exclude_chains"
            ),
            frame_start=frame_start,
            frame_stop=frame_stop,
            time_start=time_start,
            time_stop=time_stop,
            delta_t=delta_t,
            filter_result=None,
            filter_prop=filter_prop,
            filter_min=filter_min,
            filter_max=filter_max,
            normalization=norm,
        )
    if not data:
        raise ValueError("data mapping is empty")
    maps: dict[str, ResidueDensityMapResult] = {}
    for key, result in data.items():
        if filter_by is not None and key not in filter_by:
            raise KeyError(f"filter_by missing dataset {key!r}")
        maps[key] = _residue_density_single_map(
            result,
            label=key,
            include_chains=_diffusion_chain_selection(
                include_chains, key=key, name="include_chains"
            ),
            exclude_chains=_diffusion_chain_selection(
                exclude_chains, key=key, name="exclude_chains"
            ),
            frame_start=frame_start,
            frame_stop=frame_stop,
            time_start=time_start,
            time_stop=time_stop,
            delta_t=delta_t,
            filter_result=None if filter_by is None else filter_by[key],
            filter_prop=filter_prop,
            filter_min=filter_min,
            filter_max=filter_max,
            normalization=norm,
        )
    if not average_over_sets:
        return maps
    groups = _ordered_set_groups(list(maps), group_by=group_by)
    averaged: dict[str, ResidueDensityMapResult] = {}
    for name, keys in groups.items():
        first = maps[keys[0]]
        n_residues = int(len(first.residue_numbers))
        for key in keys[1:]:
            other = maps[key]
            if len(other.residue_numbers) != n_residues:
                raise ValueError(
                    f"{name}: set {key!r} has {len(other.residue_numbers)} residues, "
                    f"expected {n_residues}; sequence-position averaging requires equal lengths"
                )
            if not np.array_equal(first.edges_nm, other.edges_nm):
                raise ValueError(f"{name}: set {key!r} has different radial bin edges")
            if first.residue_site != other.residue_site:
                raise ValueError(
                    f"{name}: set {key!r} uses residue_site={other.residue_site!r}, "
                    f"expected {first.residue_site!r}"
                )
        # Align by ordinal residue position, not amino-acid identity. Preserve
        # actual residue numbers only when identical across every simulation set.
        # Inconsistent residue names are intentionally blank rather than taken
        # from the first set and incorrectly presented as a consensus sequence.
        matching_numbers = all(
            np.array_equal(first.residue_numbers, maps[key].residue_numbers) for key in keys[1:]
        )
        residue_numbers = (
            first.residue_numbers.copy()
            if matching_numbers
            else np.arange(1, n_residues + 1, dtype=np.int64)
        )
        residue_names = tuple(
            (
                first.residue_names[ri]
                if all(maps[key].residue_names[ri] == first.residue_names[ri] for key in keys[1:])
                else ""
            )
            for ri in range(n_residues)
        )
        values = np.stack([maps[key].density_nm3 for key in keys], axis=0)
        mean = np.mean(values, axis=0)
        sem = (
            np.std(values, axis=0, ddof=1) / np.sqrt(len(keys))
            if len(keys) > 1
            else np.full_like(mean, np.nan)
        )
        averaged[name] = ResidueDensityMapResult(
            density_nm3=mean,
            stderr_nm3=sem,
            edges_nm=first.edges_nm.copy(),
            r_nm=first.r_nm.copy(),
            residue_numbers=residue_numbers,
            residue_names=residue_names,
            normalization=norm,
            n_frames=sum(maps[k].n_frames for k in keys),
            n_selected_chain_frames=sum(maps[k].n_selected_chain_frames for k in keys),
            n_sets=len(keys),
            set_labels=tuple(str(k) for k in keys),
            per_set_density_nm3=values,
            residue_site=first.residue_site,
        )
    return averaged


def plot_residue_density_map(
    data: ResidueDensityMapResult,
    *,
    figsize: tuple[float, float] = (9, 6),
    cmap: Any = "viridis",
    vmin: Optional[float] = 0.0,
    vmax: Optional[float] = None,
    colorbar_label: Optional[str] = None,
    yticks_every: int = 10,
    xticks: Optional[Sequence[float]] = None,
    xlim: Optional[tuple[float, float]] = None,
    title: Optional[str] = None,
    save_path: Optional[PathLike] = None,
    dpi: int = 200,
) -> tuple[Any, Any]:
    """Plot residue density vs. radial distance, with residue 1 at top."""
    z = np.asarray(data.density_nm3, dtype=float)
    if z.shape != (len(data.residue_numbers), len(data.edges_nm) - 1):
        raise ValueError("density matrix and axes have inconsistent shapes")
    if yticks_every < 1:
        raise ValueError("yticks_every must be >= 1")
    fig, ax = plt.subplots(figsize=figsize)
    row_edges = np.arange(len(data.residue_numbers) + 1, dtype=float) + 0.5
    mesh = ax.pcolormesh(
        data.edges_nm,
        row_edges,
        z,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        shading="flat",
        rasterized=True,
    )
    tick_indices = np.arange(0, len(data.residue_numbers), yticks_every)
    if len(data.residue_numbers) and tick_indices[-1] != len(data.residue_numbers) - 1:
        tick_indices = np.append(tick_indices, len(data.residue_numbers) - 1)
    ax.set_yticks(tick_indices + 1, labels=[str(data.residue_numbers[i]) for i in tick_indices])
    ax.set_ylim(row_edges[-1], row_edges[0])  # first residue at the top
    ax.set_xlabel("Distance from condensate center (nm)")
    ax.set_ylabel("Residue number")
    if xticks is not None:
        ax.set_xticks(xticks)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if title is not None:
        ax.set_title(title)
    label = colorbar_label or (
        "Residue number density (nm$^{-3}$)"
        if data.normalization == "total"
        else "Per-chain residue density (nm$^{-3}$)"
    )
    fig.colorbar(mesh, ax=ax, label=label)
    fig.tight_layout()
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return fig, ax


# BEGIN CONDENSATE PMF EXTENSION
# Copy this block into condensate_analysis.py, or run install_condensate_pmfs.py.


@dataclass
class PMFResult:
    """Jacobian-corrected PMF on the bins of a HistogramResult.

    `values` is in units `units`, and is NaN where the source histogram
    has zero probability. `radial_density` stores the unnormalized p(r)/J(r)
    in nm^-3; its normalization is immaterial for relative PMFs.
    `jacobian='spherical'` treats the supplied distance like a conventional
    radial coordinate. This is an approximation for minimum-site distances.
    """

    values: np.ndarray
    edges: np.ndarray
    radial_density: np.ndarray
    temperature_K: float
    units: str
    jacobian: str
    reference: str
    reference_range: Optional[tuple[float, float]] = None
    n_samples: int = 0
    n_sets: int = 1
    set_labels: tuple[str, ...] = ()

    @property
    def centers(self) -> np.ndarray:
        return 0.5 * (self.edges[:-1] + self.edges[1:])

    @property
    def widths(self) -> np.ndarray:
        return np.diff(self.edges)

    @property
    def pmf(self) -> np.ndarray:
        return self.values


def calculate_pmf(
    histogram: HistogramResult,
    *,
    temperature_K: float = 298.15,
    jacobian: str = "spherical",
    reference: str = "minimum",
    reference_range: Optional[tuple[float, float]] = None,
    units: str = "kJ/mol",
) -> PMFResult:
    """Convert one binned distance distribution into a relative PMF.

    For spherical correction, divide each bin probability mass by its EXACT
    shell volume (4*pi/3)*(r_outer**3 - r_inner**3), then compute -RT*log.
    `jacobian='none'` computes a free-energy profile in the scalar coordinate.

    reference='minimum': set the minimum *finite* PMF to zero.
    reference='last':    set the last occupied bin to zero.
    reference='range':   set the mean PMF within reference_range=(lo,hi) to zero.

    Source histograms may be probability-density histograms or raw-count
    histograms; the relative PMF is identical for both. Empty bins produce NaN
    rather than infinite / arbitrarily capped free energies.

    This assumes an unbiased equilibrium histogram. Do NOT apply directly to
    umbrella-biased trajectories without reweighting (WHAM/MBAR).
    """
    edges = np.asarray(histogram.edges, dtype=float)
    heights = np.asarray(histogram.values, dtype=float)
    if edges.ndim != 1 or heights.ndim != 1 or len(edges) != len(heights) + 1:
        raise ValueError("histogram.values and histogram.edges have incompatible shapes")
    if heights.size == 0 or not np.all(np.isfinite(edges)) or not np.all(np.diff(edges) > 0):
        raise ValueError("histogram must have nonempty, strictly increasing finite bin edges")
    if np.any(~np.isfinite(heights)) or np.any(heights < 0):
        raise ValueError("histogram heights must be finite and nonnegative")
    if not np.any(heights > 0):
        raise ValueError("histogram contains no occupied bins")

    T = float(temperature_K)
    if not np.isfinite(T) or T <= 0:
        raise ValueError("temperature_K must be finite and > 0")
    unit_name = str(units).strip().lower().replace(" ", "")
    if unit_name == "kj/mol":
        R = 0.00831446261815324
        unit_label = "kJ/mol"
    elif unit_name == "kcal/mol":
        R = 0.00198720425864083
        unit_label = "kcal/mol"
    else:
        raise ValueError("units must be 'kJ/mol' or 'kcal/mol'")

    jac = str(jacobian).strip().lower()
    if jac not in {"spherical", "none"}:
        raise ValueError("jacobian must be 'spherical' or 'none'")
    if jac == "spherical" and edges[0] < 0:
        raise ValueError("spherical radial distances must have nonnegative edges")

    widths = np.diff(edges)
    # Convert the histogram to a per-bin probability mass; normalization to
    # total mass only adds a constant that is removed by PMF referencing.
    masses = heights * widths if bool(histogram.density) else heights.copy()
    masses = masses / np.sum(masses)

    if jac == "spherical":
        shell_volumes = (4.0 * np.pi / 3.0) * (edges[1:] ** 3 - edges[:-1] ** 3)
        radial_density = masses / shell_volumes
    else:
        radial_density = masses / widths

    pmf = np.full(heights.shape, np.nan, dtype=float)
    occupied = radial_density > 0
    pmf[occupied] = -(R * T) * np.log(radial_density[occupied])

    ref = str(reference).strip().lower()
    centers = (edges[:-1] + edges[1:]) / 2
    if ref in {"minimum", "min"}:
        zero = float(np.nanmin(pmf))
        ref = "minimum"
    elif ref == "last":
        zero = float(pmf[np.flatnonzero(occupied)[-1]])
    elif ref == "range":
        if reference_range is None or len(reference_range) != 2:
            raise ValueError("reference='range' requires reference_range=(rmin, rmax)")
        lo, hi = map(float, reference_range)
        if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
            raise ValueError("reference_range must contain finite ascending limits")
        mask = occupied & (centers >= lo) & (centers <= hi)
        if not np.any(mask):
            raise ValueError("reference_range contains no occupied bin centers")
        zero = float(np.mean(pmf[mask]))
    else:
        raise ValueError("reference must be 'minimum', 'last', or 'range'")
    pmf[occupied] -= zero

    return PMFResult(
        values=pmf,
        edges=edges.copy(),
        radial_density=radial_density,
        temperature_K=T,
        units=unit_label,
        jacobian=jac,
        reference=ref,
        reference_range=tuple(map(float, reference_range)) if ref == "range" else None,
        n_samples=int(getattr(histogram, "n_samples", 0)),
        n_sets=int(getattr(histogram, "n_sets", 1)),
        set_labels=tuple(getattr(histogram, "set_labels", ())),
    )


def calculate_pmfs(
    histograms: Mapping[str, HistogramResult],
    *,
    temperature_K: float = 298.15,
    jacobian: str = "spherical",
    reference: str = "minimum",
    reference_range: Optional[tuple[float, float]] = None,
    units: str = "kJ/mol",
) -> dict[str, PMFResult]:
    """Convert a mapping from calculate_histograms() into PMFs, same keys."""
    if not histograms:
        raise ValueError("histograms is empty")
    return {
        key: calculate_pmf(
            hist,
            temperature_K=temperature_K,
            jacobian=jacobian,
            reference=reference,
            reference_range=reference_range,
            units=units,
        )
        for key, hist in histograms.items()
    }


def plot_pmfs(
    pmfs: Union[Mapping[str, PMFResult], PMFResult],
    *,
    colors: Optional[Union[Mapping[str, Any], Sequence[Any]]] = None,
    labels: Optional[Union[Mapping[str, str], Sequence[str]]] = None,
    linewidth: float = 2.0,
    smooth: bool = False,
    smooth_sigma: float = 1.5,
    title: Optional[str] = None,
    xlabel: str = "Distance (nm)",
    ylabel: Optional[str] = None,
    xlim: Optional[tuple[float, float]] = None,
    ylim: Optional[tuple[float, float]] = None,
    figsize: tuple[float, float] = (6.0, 4.0),
    ax: Optional[plt.Axes] = None,
    legend: bool = True,
    save_path: Optional[PathLike] = None,
    save_dpi: Optional[float] = None,
    save_bbox_inches: Optional[str] = "tight",
    save_kwargs: Optional[Mapping[str, Any]] = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot PMFs as lines through occupied bin centers, with optional smoothing.

    Unsampled bins (NaNs) break the line; no zero baseline, stair steps, fill,
    interpolation over gaps, or extension beyond the occupied bin centers.
    With smooth=True, Gaussian smoothing and shape-preserving PCHIP interpolation
    are applied *separately* to each uninterrupted stretch of occupied bins.
    Smoothing affects the plotted line only, never the PMFResult values.
    """
    if isinstance(pmfs, PMFResult):
        pmfs = {"PMF": pmfs}
    if not pmfs:
        raise ValueError("pmfs is empty")
    if not np.isfinite(float(linewidth)) or linewidth <= 0:
        raise ValueError("linewidth must be finite and > 0")
    if not np.isfinite(float(smooth_sigma)) or smooth_sigma <= 0:
        raise ValueError("smooth_sigma must be finite and > 0")

    if smooth:
        from scipy.interpolate import PchipInterpolator
        from scipy.ndimage import gaussian_filter1d

    def style_value(spec, key, i, default=None):
        if spec is None:
            return default
        if isinstance(spec, Mapping):
            return spec.get(key, default)
        if isinstance(spec, (Sequence, np.ndarray)) and not isinstance(spec, (str, bytes)):
            return spec[i] if i < len(spec) else default
        raise TypeError("colors and labels must be mappings, sequences, or None")

    if ax is None:
        figure, ax = plt.subplots(figsize=figsize)
    else:
        figure = ax.figure

    units_found = {result.units for result in pmfs.values()}
    if ylabel is None:
        if len(units_found) != 1:
            raise ValueError(
                "PMFs with different units require an explicit ylabel (and conversion)"
            )
        ylabel = f"PMF ({next(iter(units_found))})"

    for i, (key, result) in enumerate(pmfs.items()):
        centers = np.asarray(result.centers, dtype=float)
        y = np.asarray(result.values, dtype=float)
        edges = np.asarray(result.edges, dtype=float)
        if (
            y.ndim != 1
            or edges.shape != (len(y) + 1,)
            or not np.all(np.isfinite(edges))
            or not np.all(np.diff(edges) > 0)
        ):
            raise ValueError(f"{key!r}: incompatible PMF values and edges")
        y = np.where(np.isfinite(y), y, np.nan)
        label = style_value(labels, key, i, str(key))
        color = style_value(colors, key, i)

        if not smooth:
            # Matplotlib renders NaNs as disconnections, not as zeros.
            (line,) = ax.plot(centers, y, linewidth=float(linewidth), color=color, label=label)
            # Show an isolated occupied bin as a point (it cannot form a line).
            finite = np.isfinite(y)
            singletons = finite & ~np.r_[False, finite[:-1]] & ~np.r_[finite[1:], False]
            if np.any(singletons):
                ax.plot(
                    centers[singletons],
                    y[singletons],
                    linestyle="None",
                    marker="o",
                    markersize=3,
                    color=line.get_color(),
                    label="_nolegend_",
                )
            continue

        # Smooth separately within finite runs, never across a NaN gap.
        valid_idx = np.flatnonzero(np.isfinite(y))
        if valid_idx.size == 0:
            continue
        starts = np.r_[0, np.flatnonzero(np.diff(valid_idx) > 1) + 1]
        ends = np.r_[starts[1:], valid_idx.size]
        first_segment = True
        curve_color = color  # None lets Matplotlib choose once for this dataset
        for start, end in zip(starts, ends):
            idx = valid_idx[start:end]
            seg_x, seg_y = centers[idx], y[idx]
            if len(idx) == 1:
                (line,) = ax.plot(
                    seg_x,
                    seg_y,
                    linestyle="None",
                    marker="o",
                    markersize=3,
                    color=curve_color,
                    label=label if first_segment else "_nolegend_",
                )
            else:
                smooth_y = gaussian_filter1d(seg_y, sigma=float(smooth_sigma), mode="reflect")
                curve_x = np.linspace(seg_x[0], seg_x[-1], max(100, 10 * len(idx)))
                curve_y = PchipInterpolator(seg_x, smooth_y)(curve_x)
                (line,) = ax.plot(
                    curve_x,
                    curve_y,
                    linewidth=float(linewidth),
                    color=curve_color,
                    label=label if first_segment else "_nolegend_",
                )
            curve_color = line.get_color()
            first_segment = False

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if legend:
        handles, labs = ax.get_legend_handles_labels()
        if any(lab and not str(lab).startswith("_") for lab in labs):
            ax.legend()

    _save_figure_if_requested(
        figure,
        save_path,
        save_dpi=save_dpi,
        save_bbox_inches=save_bbox_inches,
        save_kwargs=save_kwargs,
    )
    return figure, ax


def plot_pmf(
    pmf: Union[Mapping[str, PMFResult], PMFResult], **kwargs: Any
) -> tuple[plt.Figure, plt.Axes]:
    """Singular spelling of plot_pmfs(); accepts one PMF or a PMF mapping."""
    return plot_pmfs(pmf, **kwargs)


# END CONDENSATE PMF EXTENSION


def _nonnegative(value, name):
    value = float(value)
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


def _curve_arrays(curve):
    """Extract physical time (ps) and survival from an averaged MD curve."""
    if hasattr(curve, "lag_time_ps") and hasattr(curve, "survival_mean"):
        t = np.asarray(curve.lag_time_ps, dtype=float)
        y = np.asarray(curve.survival_mean, dtype=float)
    else:
        a = np.asarray(curve, dtype=float)
        if a.ndim != 2 or a.shape[1] < 2:
            raise ValueError("Supply a QuenchingAverageResult or [time_ps, survival] array")
        t, y = a[:, 0], a[:, 1]

    if t.ndim != 1 or y.shape != t.shape or t.size < 2:
        raise ValueError("Time and survival must be matching 1D arrays of length >= 2")
    if not np.all(np.isfinite(t)) or np.any(t < 0) or np.any(np.diff(t) <= 0):
        raise ValueError("Time must be finite, nonnegative, and strictly increasing")
    if not np.all(np.isfinite(y)):
        raise ValueError("Survival contains missing or nonfinite values")

    return t.copy(), y.copy()


def make_sample_survival(cond_curve, *, f_cond, k_dil_total, k0_cond):
    """
    Combine MD condensate survival with an exponential dilute contribution.

    cond_curve:
        QuenchingAverageResult, or array with columns [time_ps, survival].
        Must contain combined intra+inter quenching over ALL selected chains,
        but NO intrinsic decay. Time must already be kinetically corrected.
    f_cond:
        Condensate signal fraction, between 0 and 1.
    k_dil_total:
        Dilute rate INCLUDING intrinsic decay, s^-1.
    k0_cond:
        Intrinsic decay rate in the condensate, s^-1.

    No time rescaling, first-point normalization, or extrapolation is applied.
    """
    t_ps, S_cond_q = _curve_arrays(cond_curve)

    f_cond = float(f_cond)
    if not np.isfinite(f_cond) or not 0 <= f_cond <= 1:
        raise ValueError("f_cond must lie in [0, 1]")

    k_dil_total = _nonnegative(k_dil_total, "k_dil_total")
    k0_cond = _nonnegative(k0_cond, "k0_cond")

    if np.any((S_cond_q < -1e-10) | (S_cond_q > 1 + 1e-10)):
        raise ValueError("Input survival must be normalized to [0, 1]")

    t_s = t_ps * 1e-12

    S_dil_total = np.exp(-k_dil_total * t_s)
    S_cond_total = np.exp(-k0_cond * t_s) * S_cond_q

    dilute_component = (1 - f_cond) * S_dil_total
    condensate_component = f_cond * S_cond_total
    S_sample = dilute_component + condensate_component

    return dict(
        lag_time_ps=t_ps,
        survival_mean=S_sample,
        curve=np.column_stack((t_ps, S_sample)),
        S_cond_quenching=S_cond_q,
        S_cond_total=S_cond_total,
        S_dil_total=S_dil_total,
        dilute_component=dilute_component,
        condensate_component=condensate_component,
        f_cond=f_cond,
        k_dil_total=k_dil_total,
        k0_cond=k0_cond,
    )


def fit_sample_survival(
    sample,
    *,
    fit_range_ps=None,
    fit_amplitude=True,
    enforce_zero_offset=True,
    initial_k_cond_total=1e5,
    maxfev=20000,
):
    """
    Fit:
        A * [(1-f)*exp(-kd*t) + f*exp(-kc*t)] + C

    Only one kinetic rate, kc, is adjustable; f and kd are fixed.
    A is adjustable unless fit_amplitude=False.
    C is zero unless enforce_zero_offset=False; otherwise C >= 0.

    Returns kc and kq = kc - k0_cond, both in s^-1.
    Fits are unweighted. Covariance describes the numerical fit,
    not experimental uncertainty.
    """
    t_ps, y = _curve_arrays(sample["curve"])

    f = float(sample["f_cond"])
    if not np.isfinite(f) or not 0 < f <= 1:
        raise ValueError("Require 0 < f_cond <= 1 to fit the condensate rate")

    kd = _nonnegative(sample["k_dil_total"], "k_dil_total")
    k0 = _nonnegative(sample["k0_cond"], "k0_cond")
    initial = _nonnegative(initial_k_cond_total, "initial_k_cond_total")
    if initial == 0:
        initial = 1.0

    mask = t_ps > 0

    if fit_range_ps is not None:
        limits = np.asarray(fit_range_ps, dtype=float)
        if (
            limits.shape != (2,)
            or not np.all(np.isfinite(limits))
            or not 0 <= limits[0] < limits[1]
        ):
            raise ValueError("fit_range_ps must be a finite increasing (min, max) pair")
        if limits[0] < t_ps[0] or limits[1] > t_ps[-1]:
            raise ValueError(
                "Fit window exceeds available MD lags; " "no extrapolation is performed"
            )
        mask &= (t_ps >= limits[0]) & (t_ps <= limits[1])

    tfit, yfit = t_ps[mask], y[mask]
    npar = 1 + int(bool(fit_amplitude)) + int(not enforce_zero_offset)

    if tfit.size < npar + 1:
        raise ValueError("Too few points in the fitting window")

    # Fit time in microseconds and rate in inverse microseconds.
    def model(t_us, kc_us, *extra):
        i = 0
        A = extra[i] if fit_amplitude else 1.0
        i += int(bool(fit_amplitude))
        C = extra[i] if not enforce_zero_offset else 0.0

        return A * ((1 - f) * np.exp(-kd * 1e-6 * t_us) + f * np.exp(-kc_us * t_us)) + C

    p0 = [initial * 1e-6]
    if fit_amplitude:
        p0.append(1.0)
    if not enforce_zero_offset:
        p0.append(0.0)

    popt, pcov = curve_fit(
        model,
        tfit * 1e-6,
        yfit,
        p0=p0,
        bounds=(np.zeros(npar), np.full(npar, np.inf)),
        maxfev=int(maxfev),
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-12,
    )

    fitted = model(tfit * 1e-6, *popt)

    # Convert fitted rate and covariance back to s^-1.
    scale = np.ones(npar)
    scale[0] = 1e6
    covariance = pcov * scale[:, None] * scale[None, :]

    kc = float(popt[0] * 1e6)
    index = 1
    amplitude = float(popt[index]) if fit_amplitude else 1.0
    index += int(bool(fit_amplitude))
    offset = float(popt[index]) if not enforce_zero_offset else 0.0

    return SimpleNamespace(
        k_cond_total_per_s=kc,
        k_obs_per_s=kc,
        k_q_per_s=kc - k0,  # Intrinsic decay already subtracted
        k0_per_s=k0,
        amplitude=amplitude,
        offset=offset,
        fit_time_ps=tfit,
        fitted_survival=fitted,
        residuals=yfit - fitted,
        rmse=float(np.sqrt(np.mean((yfit - fitted) ** 2))),
        covariance=covariance,
        parameter_stderr=np.sqrt(np.maximum(np.diag(covariance), 0)),
        n_points=tfit.size,
        f_cond=f,
        k_dil_total_per_s=kd,
    )


def kr(
    h,
    *,
    rmax=1.2,
    smooth_sigma=None,
    q0val=4.2e9,
    betaval=40,
    aval=0.525,
    epsval=1e-4,
    show=False,
    report=False,
):
    edges = np.asarray(h.edges, dtype=float)
    centers = np.asarray(h.centers, dtype=float)
    P = np.asarray(h.values, dtype=float)

    if not h.density or not np.isclose(edges[0], 0.0):
        raise ValueError("Need a density histogram starting at r=0")
    if not 0 < rmax <= edges[-1]:
        raise ValueError("rmax must be inside the histogram range")

    if smooth_sigma is not None:
        # Gaussian smoothing (same as plot_histograms)
        P_smooth = gaussian_filter1d(P, sigma=smooth_sigma, mode="reflect")
    else:
        P_smooth = P

    # Shape-preserving cubic interpolation
    # Extend constant end-bin values to histogram boundaries
    x = np.r_[edges[0], centers, edges[-1]]
    y = np.r_[P_smooth[0], P_smooth, P_smooth[-1]]
    P_interp = PchipInterpolator(x, y, extrapolate=False)

    # Normalize the smoothed PDF over the full histogram range
    norm = P_interp.integrate(edges[0], edges[-1])

    # Fine grid for numerical integration
    r = np.linspace(0, rmax, 4001)
    Pr = P_interp(r) / norm

    # Distance-dependent quenching rate
    qr = q0val * np.exp(-betaval * epsval * np.logaddexp(0.0, (r - aval) / epsval))
    # qr = q0val * np.exp(-betaval * np.maximum(r - aval, 0.0))

    # Integrate q(r) P(r)
    integrand = qr * Pr
    k_avg = simpson(integrand, x=r)

    # Probability of finding r <= rmax
    p_close = P_interp.integrate(aval, rmax) / norm

    # Optional: conditional mean quenching rate for r <= rmax
    # print(f"Conditional mean q = {k_avg/p_close:.5e} s^-1")
    if show:
        # Plot PDF and integrand
        fig, axes = plt.subplots(1, 2, figsize=(10, 3))

        r_full = np.linspace(edges[0], edges[-1], 4001)

        axes[0].stairs(P, edges, alpha=0.4, label="histogram")
        axes[0].plot(r_full, P_interp(r_full) / norm, lw=2, label="smoothed")
        axes[0].set_xlim(0, 2)
        axes[0].set_xlabel("r (nm)")
        axes[0].set_ylabel("P(r) (nm$^{-1}$)")
        axes[0].legend()

        axes[1].plot(r, integrand)
        axes[1].set_xlabel("r (nm)")
        axes[1].set_ylabel("q(r) P(r) (s$^{-1}$ nm$^{-1}$)")
        axes[1].set_xlim(0, rmax)

        plt.tight_layout()
        plt.show()

    if report:
        print(f"P(r < {rmax:g} nm) = {p_close:.6f}")
        print(f"Integral q(r)P(r) = {k_avg:.5e} s^-1")

    return k_avg


def kr_diffusive(
    h,
    Dcoord_cm2_s,
    V_nm3,
    *,
    rmax=1.2,
    smooth_sigma=None,
    q0val=4.2e9,
    betaval=40,
    aval=0.525,
    epsval=1e-4,
    show=False,
    report=False,
):
    """
    Reaction-limited and diffusion-influenced quenching rates
    from an equilibrium minimum-distance histogram.

    Distances: nm
    q0val: s^-1
    betaval: nm^-1
    epsval: nm
    Dcoord_cm2_s: effective diffusion along r, cm^2/s
    V_nm3: simulation box volume, nm^3

    Returns:
        kR, kq       first-order rates, s^-1
        kR2, kq2     bimolecular rates, M^-1 s^-1
    """

    edges = np.asarray(h.edges, dtype=float)
    centers = np.asarray(h.centers, dtype=float)
    P = np.asarray(h.values, dtype=float)

    if not h.density or not np.isclose(edges[0], 0.0):
        raise ValueError("Need density histogram starting at r=0")
    if not 0 < rmax <= edges[-1]:
        raise ValueError("rmax must be inside histogram range")
    if np.any(P < 0) or len(P) != len(edges) - 1:
        raise ValueError("Invalid histogram")
    if Dcoord_cm2_s <= 0 or V_nm3 <= 0:
        raise ValueError("D and V must be positive")
    if epsval is not None and epsval <= 0:
        raise ValueError("epsval must be positive")

    # Same smoothing as your kr function
    if smooth_sigma is not None:
        P_smooth = gaussian_filter1d(P, sigma=smooth_sigma, mode="reflect")
    else:
        P_smooth = P

    # Same PCHIP interpolation
    x = np.r_[edges[0], centers, edges[-1]]
    y = np.r_[P_smooth[0], P_smooth, P_smooth[-1]]

    P_interp = PchipInterpolator(x, y, extrapolate=False)
    P_antideriv = P_interp.antiderivative()

    norm = float(P_antideriv(edges[-1]) - P_antideriv(edges[0]))
    if norm <= 0:
        raise ValueError("Histogram normalization is zero")

    def qfunc(r):
        if epsval is None:
            sep = np.maximum(r - aval, 0.0)
        else:
            sep = epsval * np.logaddexp(0.0, (r - aval) / epsval)
        return q0val * np.exp(-betaval * sep)

    # Equilibrium probability mass in each histogram bin
    mass = np.maximum((P_antideriv(edges[1:]) - P_antideriv(edges[:-1])) / norm, 0.0)

    # Integrate P(r)*q(r) within every bin.
    # Quenching is neglected beyond rmax, but the complete
    # distance distribution remains in the diffusion model.
    nodes, weights = leggauss(8)

    lo = edges[:-1]
    hi = np.minimum(edges[1:], rmax)
    widths = np.maximum(hi - lo, 0.0)

    points = lo[:, None] + widths[:, None] * (nodes[None, :] + 1.0) / 2.0

    Ppoints = P_interp(points) / norm
    qpoints = qfunc(points)

    sink = widths / 2.0 * np.sum(weights[None, :] * Ppoints * qpoints, axis=1)

    # Remove only negligible empty tails.
    # Interior gaps indicate insufficiently sampled P(r).
    threshold = np.max(mass) * 1e-12
    valid = np.flatnonzero(mass > threshold)

    if valid.size == 0:
        raise ValueError("No nonzero probability bins")

    first = int(valid[0])
    last = int(valid[-1]) + 1

    if np.any(mass[first:last] <= threshold):
        raise ValueError(
            "P(r) has internal zero-probability gaps; " "increase sampling or smoothing"
        )

    retained_mass = np.sum(mass[first:last])
    lost_rate = np.sum(sink) - np.sum(sink[first:last])

    if retained_mass < 1.0 - 1e-6 or lost_rate > 1e-6 * np.sum(sink):
        raise ValueError("Removing low-probability bins changes the result")

    m = mass[first:last] / retained_mass
    K = sink[first:last] / retained_mass
    rc = centers[first:last]

    # Reaction-limited rate: equilibrium average
    kR = float(np.sum(K))

    # Diffusive conductance between neighboring bins.
    # Resistance integral ensures detailed balance
    # with the measured equilibrium P(r).
    delta = np.diff(rc)
    midpoint = 0.5 * (rc[:-1] + rc[1:])

    quadrature_points = midpoint[:, None] + 0.5 * delta[:, None] * nodes[None, :]

    pdf_links = P_interp(quadrature_points) / norm

    if np.any(pdf_links <= 0):
        raise ValueError("P(r) vanishes between populated bins")

    resistance = delta / 2.0 * np.sum(weights[None, :] / pdf_links, axis=1)

    D_nm2_s = Dcoord_cm2_s * 1e14
    coupling = D_nm2_s / resistance

    # Symmetric diffusion-reaction operator.
    # Its smallest eigenvalue is the long-time decay rate.
    diagonal = K / m

    diagonal[:-1] += coupling / m[:-1]
    diagonal[1:] += coupling / m[1:]

    offdiagonal = -coupling / np.sqrt(m[:-1] * m[1:])

    kq = float(
        eigh_tridiagonal(
            diagonal,
            offdiagonal,
            eigvals_only=True,
            select="i",
            select_range=(0, 0),
            check_finite=False,
        )[0]
    )

    # Convert first-order to bimolecular rates
    factor = 0.602214076 * V_nm3
    kR2 = factor * kR
    kq2 = factor * kq

    if show:
        fig, axes = plt.subplots(1, 2, figsize=(10, 3))

        r_full = np.linspace(edges[0], edges[-1], 4001)

        axes[0].stairs(P, edges, alpha=0.4, label="histogram")
        axes[0].plot(r_full, P_interp(r_full) / norm, lw=2, label="smoothed")
        axes[0].set_xlim(0, 2)
        axes[0].set_xlabel("r (nm)")
        axes[0].set_ylabel("P(r) (nm$^{-1}$)")
        axes[0].legend()

        r_plot = np.linspace(0, rmax, 4001)
        integrand = qfunc(r_plot) * P_interp(r_plot) / norm

        axes[1].plot(r_plot, integrand)
        axes[1].set_xlabel("r (nm)")
        axes[1].set_ylabel("q(r) P(r) (s$^{-1}$ nm$^{-1}$)")
        axes[1].set_xlim(0, rmax)

        plt.tight_layout()
        plt.show()

    p_close = P_interp.integrate(0, rmax) / norm

    if report:
        print(f"P(r < {rmax:g} nm) = {p_close:.6f}")
        print(f"kR = {kR:.5e} s^-1")
        print(f"kq = {kq:.5e} s^-1")
        print(f"kq/kR = {kq/kR:.5f}")
        print(f"kR bimolecular = {kR2:.5e} M^-1 s^-1")
        print(f"kq bimolecular = {kq2:.5e} M^-1 s^-1")

    return kR, kq, kR2, kq2


def fill_internal_gaps(
    P,
    edges,
    max_gap_bins=3,
    max_tail_fraction=1e-3,
):
    """
    Fill short internal gaps and remove isolated,
    negligibly populated tails beyond large gaps.

    Leading and trailing zero regions remain zero.
    """
    P = np.asarray(P, dtype=float).copy()
    edges = np.asarray(edges, dtype=float)
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)

    original_mass = np.sum(P * widths)
    if original_mass <= 0:
        raise ValueError("Empty probability distribution")

    # First remove negligible tails separated by large gaps
    while True:
        positive = np.flatnonzero(P > 0)
        if len(positive) < 2:
            break

        large_gap_found = False

        for left, right in zip(positive[:-1], positive[1:]):
            gap_size = right - left - 1

            if gap_size <= max_gap_bins:
                continue

            large_gap_found = True

            right_mass = np.sum(P[right:] * widths[right:]) / original_mass

            left_mass = np.sum(P[: left + 1] * widths[: left + 1]) / original_mass

            if right_mass <= max_tail_fraction:
                # print(
                #    f"Removing upper tail after gap "
                #    f"{centers[left]:.3f}–"
                #    f"{centers[right]:.3f} nm "
                #    f"({gap_size} empty bins, "
                #    f"probability={right_mass:.3e})"
                # )
                P[right:] = 0.0
                break

            elif left_mass <= max_tail_fraction:
                # print(
                #    f"Removing lower tail before gap "
                #    f"({gap_size} empty bins, "
                #    f"probability={left_mass:.3e})"
                # )
                P[: left + 1] = 0.0
                break

            else:
                raise ValueError(
                    f"Significant internal gap of "
                    f"{gap_size} bins near "
                    f"{centers[left]:.3f} nm"
                )

        if not large_gap_found:
            break

    # Fill remaining short internal gaps
    positive = np.flatnonzero(P > 0)

    if len(positive) >= 2:
        first, last = positive[0], positive[-1]

        missing = np.flatnonzero(P[first : last + 1] == 0) + first

        if len(missing):
            P[missing] = np.interp(centers[missing], centers[positive], P[positive])
            # print(f"Interpolated {len(missing)} internal bins")

    # removed_fraction = (original_mass - np.sum(P * widths)) / original_mass
    # print(f"Total probability removed: {removed_fraction:.3e}")

    return P


def estimate_D_from_kDplus(
    h,
    *,
    kDplus=1.5e6,
    q0val=4.2e9,
    betaval=40,
    aval=0.525,
    epsval=1e-4,
    smooth_sigma=None,
    r_lower=0.0,
    r_upper=8,
    ngrid=20001,
    show=False,
    report=False,
    output_path=None,
    kR_input=None,
):
    """
    Estimate constant D using Lapidus et al. (2002), Eq. 9.

    Parameters
    ----------
    h : HistogramResult
        Complete equilibrium distance distribution.
        h.values must be a probability density.
    kDplus : float
        Diffusion-limited rate, s^-1.
    q0val : float
        Intrinsic contact quenching rate, s^-1.
    betaval : float
        Distance-decay exponent, nm^-1.
    aval : float
        Contact distance, nm.
    epsval : float or None
        Smooth-capping length, nm.
    smooth_sigma : float or None
        Gaussian histogram smoothing, in bins.
    r_lower, r_upper : float
        Numerical limits of the OUTER Eq. 9 integral.
        Neither affects normalization or kR.
    ngrid : int
        Number of points on full distance grid.
    show : bool
        Plot P(r) and the Eq. 9 integrand.
    kR_input : float or None
        Optional direct replacement for kR in BOTH the inner
        (q(r)-kR) factor and the kR**2 denominator of Eq. 9.
        q0val and q(r) are NOT changed. If inconsistent with
        integral(q(r)*P(r) dr), Eq. 9 no longer has its usual
        equilibrium interpretation and may be singular at an
        empty lower-distance tail.

    Returns
    -------
    dict
        D in nm^2/ns, A^2/ns, cm^2/s, and diagnostics.
    """

    edges = np.asarray(h.edges, dtype=float)
    centers = np.asarray(h.centers, dtype=float)
    P_raw = np.asarray(h.values, dtype=float)

    if not h.density or not np.isclose(edges[0], 0.0):
        raise ValueError("Need a density histogram starting at r=0")

    if len(P_raw) != len(edges) - 1 or np.any(P_raw < 0) or not np.all(np.isfinite(P_raw)):
        raise ValueError("Invalid histogram")

    if kDplus <= 0 or q0val <= 0 or betaval < 0:
        raise ValueError("Invalid rate parameters")
    if kR_input is not None and (not np.isfinite(kR_input) or kR_input <= 0):
        raise ValueError("kR_input must be a positive finite rate")

    if epsval is not None and epsval <= 0:
        raise ValueError("epsval must be positive")

    lo = float(r_lower)
    hi = edges[-1] if r_upper is None else float(r_upper)

    if not edges[0] <= lo < hi <= edges[-1]:
        raise ValueError("Invalid integration limits")

    # --------------------------------------------------
    # 1. Smooth and interpolate the full P(r)
    # --------------------------------------------------

    P_filled = fill_internal_gaps(P_raw, edges, max_gap_bins=3, max_tail_fraction=1e-3)

    # Smooth only over the retained populated support
    P_smooth = np.zeros_like(P_filled)

    positive = np.flatnonzero(P_filled > 0)
    first, last = positive[0], positive[-1]

    if smooth_sigma is not None:
        P_smooth[first : last + 1] = gaussian_filter1d(
            P_filled[first : last + 1], sigma=smooth_sigma, mode="reflect"
        )
    else:
        P_smooth = P_filled.copy()

    x = np.r_[edges[0], centers, edges[-1]]
    y = np.r_[P_smooth[0], P_smooth, P_smooth[-1]]

    P_interp = PchipInterpolator(x, y, extrapolate=False)

    # Full distance grid, independent of r_upper
    r = np.linspace(edges[0], edges[-1], ngrid)
    Pr = np.maximum(P_interp(r), 0.0)

    # Normalize over the entire histogram range
    norm = trapezoid(Pr, x=r)

    if norm <= 0:
        raise ValueError("Zero probability normalization")

    Pr /= norm

    # --------------------------------------------------
    # 2. Distance-dependent quenching
    # --------------------------------------------------

    if epsval is None:
        separation = np.maximum(r - aval, 0.0)
    else:
        separation = epsval * np.logaddexp(0.0, (r - aval) / epsval)

    # Factor out q0 for numerical conditioning.
    # When kR is overridden without changing q0, D *does*
    # depend on the specified value of q0.
    qshape = np.exp(-betaval * separation)

    if np.ptp(qshape) <= 1e-12 * np.max(qshape):
        raise ValueError("q(r) is nearly constant; Eq. 9 cannot determine D")

    qmean = trapezoid(qshape * Pr, x=r)

    if qmean <= 0:
        raise ValueError("Mean quenching propensity is zero")

    kR_calculated = q0val * qmean
    kR = kR_calculated if kR_input is None else float(kR_input)

    # q(r) stays q0val*qshape regardless of override.
    # This is the dimensionless version of the supplied kR.
    qref = kR / q0val
    inconsistent = not np.isclose(kR, kR_calculated, rtol=1e-8, atol=0.0)

    # --------------------------------------------------
    # 3. Cumulative integral in Eq. 9
    # --------------------------------------------------

    delta = (qshape - qref) * Pr

    # Equation 9 explicitly requires the UPPER-tail integral:
    # C(r) = integral_r^L [qshape(y)-kR/q0val] P(y) dy.
    C_right = -cumulative_trapezoid(delta[::-1], x=r[::-1], initial=0.0)[::-1]

    if inconsistent:
        # With external kR different from <q>, the integral of
        # delta is nonzero: C_left and C_right are NOT equivalent.
        C = C_right
        warnings.warn(
            "kR_input differs from integral(q*P). Using the "
            "upper-limit cumulative integral of Eq. 9 without "
            "rescaling q0. The resulting D is not an equilibrium "
            "SSS diffusion coefficient and may depend strongly "
            "on the lower-distance cutoff and numerical grid.",
            RuntimeWarning,
            stacklevel=2,
        )
    else:
        # For mathematically consistent kR, the two directions
        # agree and switching direction suppresses roundoff.
        C_left = -cumulative_trapezoid(delta, x=r, initial=0.0)
        cdf = cumulative_trapezoid(Pr, x=r, initial=0.0)
        C = np.where(cdf <= 0.5, C_left, C_right)

    # --------------------------------------------------
    # 4. Handle zero-probability boundaries
    # --------------------------------------------------

    positive = np.flatnonzero(Pr > 0.0)

    if positive.size == 0:
        raise ValueError("P(r) is zero everywhere")

    first = int(positive[0])
    last = int(positive[-1])

    # A zero-probability gap inside the sampled support
    # is not equivalent to an empty boundary region.
    if np.any(Pr[first : last + 1] == 0.0):
        raise ValueError(
            "P(r) has an INTERNAL zero-density gap. "
            "Increase sampling, adjust binning, or fit a "
            "physically justified continuous distribution."
        )

    # In the consistent case both P and C vanish beyond the
    # populated support. With inconsistent kR, C can remain
    # NONZERO below the lower support, where P=0. The exact
    # Eq. 9 integral is then singular; we leave C untouched
    # for diagnostics and set the undefined integrand to zero
    # outside support (a cutoff-dependent numerical convention).
    if inconsistent and first > 0 and lo < r[first]:
        warnings.warn(
            "P(r)=0 below its populated support, while the "
            "upper-tail cumulative integral is nonzero there. "
            "The formal Eq. 9 integral diverges; this code "
            "omits empty bins and returns a grid-dependent "
            "numerical estimate. Set r_lower inside the "
            "populated region to specify a finite cutoff.",
            RuntimeWarning,
            stacklevel=2,
        )

    # --------------------------------------------------
    # 5. Eq. 9 outer integrand
    # --------------------------------------------------

    integrand_full = np.zeros_like(Pr)

    integrand_full[first : last + 1] = C[first : last + 1] ** 2 / Pr[first : last + 1] / qref**2

    if not np.all(np.isfinite(integrand_full)):
        raise ValueError("Nonfinite Eq. 9 integrand. " "Check low-probability tails.")

    def outer_integral(lower, upper):
        rr = np.r_[lower, r[(r > lower) & (r < upper)], upper]

        yy = np.interp(rr, r, integrand_full)

        integral = trapezoid(yy, x=rr)

        return float(integral), rr, yy

    # Selected range
    integral, r_outer, integrand = outer_integral(lo, hi)

    # Complete distribution for convergence test
    I_full, _, _ = outer_integral(edges[0], edges[-1])

    if I_full <= 0:
        raise ValueError("Zero Eq. 9 integral; cannot estimate D")

    fraction = integral / I_full

    # --------------------------------------------------
    # 6. Calculate diffusion coefficient
    # --------------------------------------------------

    # From Eq. 9:
    # 1/kDplus = I/D
    # Therefore D = kDplus * I

    D_nm2_s = kDplus * integral

    D_nm2_ns = D_nm2_s * 1e-9
    D_A2_ns = D_nm2_s * 1e-7
    D_cm2_s = D_nm2_s * 1e-14

    # --------------------------------------------------
    # 7. Optional plots
    # --------------------------------------------------

    if show:

        fig, axes = plt.subplots(1, 2, figsize=(11, 3.7))

        axes[0].stairs(P_raw, edges, alpha=0.35, label="histogram")

        axes[0].plot(r, Pr, lw=2, label="smoothed P(r)")

        axes[0].axvline(hi, ls="--", color="gray", label="upper cutoff")

        axes[0].set_xlabel("r (nm)")
        axes[0].set_ylabel("P(r) (nm$^{-1}$)")
        axes[0].legend()

        axes[1].plot(r, integrand_full, label="Full Eq. 9 integrand")

        axes[1].axvspan(lo, hi, alpha=0.15, color="gray", label="Integrated range")

        axes[1].set_xlabel("r (nm)")
        axes[1].set_ylabel("SSS integrand (nm)")
        axes[1].legend()

        plt.tight_layout()
        if output_path is not None:
            plt.savefig(output_path)
        plt.show()

    # --------------------------------------------------
    # 8. Report results
    # --------------------------------------------------

    if report:
        print(f"kR (from P,q):     {kR_calculated:.5e} s^-1")
        print(f"kR (used Eq. 9):   {kR:.5e} s^-1")
        if inconsistent:
            print("NOTE: kR override without q0 rescaling; Eq. 9 inconsistent")
        print(f"kD+:               {kDplus:.5e} s^-1")
        # print(f"I (selected):      {I:.5e} nm^2")
        # print(f"I (full):          {I_full:.5e} nm^2")
        # print(f"Fraction retained: {fraction:.6f}")
        print(f"D:                 {D_nm2_ns:.5e} nm^2/ns")

    return {
        "D_nm2_ns": D_nm2_ns,
        "D_A2_ns": D_A2_ns,
        "D_cm2_s": D_cm2_s,
        "kR_s": kR,
        "kR_calculated_s": kR_calculated,
        "kR_used_s": kR,
        "kR_overridden": kR_input is not None,
        "kR_inconsistent": inconsistent,
        "kDplus_s": kDplus,
        "I_nm2": integral,
        "I_full_nm2": I_full,
        "fraction_retained": fraction,
        "r": r,
        "P": Pr,
        "q_per_s": q0val * qshape,
        "q0_used_s": q0val,
        "C_dimensionless": C,
        "r_outer": r_outer,
        "integrand": integrand,
    }


def as_ns(x):
    return float(x.value_in_unit(nanoseconds)) if hasattr(x, "value_in_unit") else float(x)


def overall_D_from_binned(msd, series, refs, dt_ns, start_ns, fit_min, fit_max):
    """Overall D and SEM across selected reference chains."""
    first = int(np.ceil(start_ns / dt_ns - 1e-10))
    nframes = msd.n_frames
    t = msd.t_ns
    fit_window = (t >= fit_min) & (t <= fit_max)

    D_refs = []

    for ref in refs:
        idx = msd.chain_labels.index(ref)
        x = np.atleast_2d(np.asarray(series[ref], dtype=float))

        if x.shape[1] != nframes:
            x = x[:, first : first + nframes]
        if x.shape[1] != nframes:
            raise ValueError(f"Time series for {ref} does not align with MSD")

        pooled_msd = np.full(len(t), np.nan)

        for k, lag in enumerate(msd.lags_frames):
            lag = int(lag)
            origins = np.arange(0, nframes - lag, msd.origin_stride)

            r0 = x[:, origins]
            r1 = x[:, origins + lag]
            valid = np.isfinite(r0) & np.isfinite(r1)

            counts, _ = np.histogram(r0[valid], bins=msd.r_edges_nm)
            y = msd.msd_rep_nm2[idx, :, k]
            use = (counts > 0) & np.isfinite(y)

            if np.any(use):
                pooled_msd[k] = np.average(y[use], weights=counts[use])

        valid = fit_window & np.isfinite(pooled_msd)
        if np.sum(valid) >= 2:
            slope = np.polyfit(t[valid], pooled_msd[valid], 1)[0]
            D_refs.append(slope / 2.0)

    D_refs = np.asarray(D_refs)
    if not len(D_refs):
        return np.nan, np.nan, 0

    mean = np.mean(D_refs)
    sem = np.std(D_refs, ddof=1) / np.sqrt(len(D_refs)) if len(D_refs) > 1 else np.nan

    return mean, sem, len(D_refs)
