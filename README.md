# sarChangeDetection
Contains various change detectors for SAR imagery

## Installation

### From source (development)

This project is [uv](https://docs.astral.sh/uv/)-managed.

```powershell
uv sync
```

This creates `.venv/`, installs the package (editable) plus `numpy` and `numba`, and makes `sarChangeDetection` importable.

### As a dependency from git

```powershell
pip install git+https://github.com/yapitsmejs/sarChangeDetection.git
# or with uv:
uv add "sar-change-detection @ git+https://github.com/yapitsmejs/sarChangeDetection.git"
```

This installs `numpy` and `numba` (the only declared runtime dependencies). The package then runs on its **NumPy fallback** by default.

## Optional: GPU acceleration with CuPy

CuPy is **not** a declared dependency, because the correct wheel depends on the host's NVIDIA GPU and CUDA Toolkit version and cannot be chosen statically. Install it manually for your platform.

**With a CUDA Toolkit installed** (uses system CUDA, no bundled libs):

| CUDA major | Wheel |
|---|---|
| 11 | `cupy-cuda11x` |
| 12 | `cupy-cuda12x` |
| 13 | `cupy-cuda13x` |

```powershell
pip install cupy-cuda12x   # match your CUDA Toolkit major version (11 / 12 / 13)
```

**With an NVIDIA GPU but no CUDA Toolkit** (bundles CUDA libraries via PyPI):

```powershell
pip install "cupy-cuda12x[ctk]"   # use the major your driver supports; default 12
```

Determine your CUDA major from `nvcc --version` (Toolkit) or the "CUDA Version" line of `nvidia-smi` (driver-supported runtime). Only one cupy distribution may be installed at a time — uninstall any existing `cupy` / `cupy-cuda*` first.

When developing from a clone, the bundled `scripts/install_cupy.py` automates this detection and installation:

```powershell
uv run python scripts/install_cupy.py
```

(That script is not shipped in the wheel and is therefore unavailable to git-installed consumers — use the manual `pip install` steps above.)