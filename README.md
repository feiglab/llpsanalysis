# LLPS analysis

This repository installs two separate Python packages from one source distribution:

- `llpsanalysis` — the original analysis routines.
- `condensate` — notebook-facing condensate trajectory analysis.

## Install

Requires Python 3.9 or later. Python-version-specific dependencies are
selected automatically by pip (see `pyproject.toml`). In the repository root:

```bash
python -m pip install -e .
```

Both packages share the dependencies and version specified in `pyproject.toml`.

## Notebook imports

```python
from llpsanalysis.llpsanalysis import *
from condensate.condensate_analysis import *
```

Alternatively, import specific functions from `condensate.condensate_analysis`.
No changes to the existing `llpsanalysis/__init__.py` are required.
The new `src/condensate/__init__.py` makes it discoverable as a separate package.
Remove old notebook `sys.path` edits for the local source file and restart the kernel.

## Existing caches

`condensate_analysis.py` supports portable cache data and trusted legacy pickle files.
Compatibility aliases cover saved class module paths of `condensate_analysis` and
`llpsanalysis.condensate_analysis`. Only load pickle files you trust.

## Quick check

```bash
python -c "from condensate.condensate_analysis import get_values, calculate_histograms; print('Import OK')"
python -m pytest -q
```

The import also requires a suitable version of `mdsim` with the analysis symbols used
by `condensate_analysis.py`. Ensure the published `mdsim` GitHub revision is current.

## Python 3.9 compatibility

Python 3.9 installations use NumPy 1.x, SciPy 1.13.x, Matplotlib 3.9.x,
MDTraj 1.10.x, and OpenMM 8.1.x. These ranges are declared using PEP 508
environment markers, so newer Python versions can use newer package releases.
The `mdsim` GitHub dependency itself requires `numpy<2.0` on Python 3.9.

For a clean 3.9 verification, activate a Python 3.9 environment and run:

```bash
python -m pip install -e .
python -c "import llpsanalysis, condensate; print('Both packages imported')"
python -m pytest -q
```

The source was syntax-parsed with Python 3.9 grammar, but its full runtime
behavior has not been tested under a Python 3.9 interpreter in this build.
