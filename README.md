# sarChangeDetection
Contains various change detectors for SAR imagery

## Installation

### From source (development)

This project is [uv](https://docs.astral.sh/uv/)-managed.

```powershell
uv sync
```

This creates `.venv/`, installs the package (editable) plus its runtime dependencies (`numpy`, `numba`, `scipy`) and the dev dependency `matplotlib` (for plotting during development), and makes `sarChangeDetection` importable.

### As a dependency from git

```powershell
pip install git+https://github.com/yapitsmejs/sarChangeDetection.git
# or with uv:
uv add "sar-change-detection @ git+https://github.com/yapitsmejs/sarChangeDetection.git"
```

This installs `numpy`, `numba`, and `scipy` (the declared runtime dependencies). The package then runs on its **NumPy/SciPy fallback** by default.

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

## Usage

### Novák multi-polarization GLRT — `getTestStatistics`

Per-pixel generalized likelihood ratio test statistic for equality of the two local (windowed) polarimetric covariance matrices of a co-registered image pair (Conradsen et al. 2003 / Novák formulation).

```python
import numpy as np
from sarChangeDetection.novakMultiChannelCD import getTestStatistics

# img1, img2: co-registered (H, W, C) channel-last SAR images, same shape.
# Real or complex; NumPy or CuPy arrays.
stat = getTestStatistics(img1, img2, kernel_size=(3, 3))
# stat: (H, W) float64. Q in [0, 1]: Q = 1 => no change, Q -> 0 => strong change.
# Border pixels whose window does not fully overlap the image are NaN.
```

- **Channel-last** `(H, W, C)`; both images must share the same shape.
- **Odd positive kernel** `(kh, kw)` (enforced); looks `n = kh*kw`, needs `kh*kw >= C` for a non-singular covariance.
- **GPU auto-acceleration**: when a CUDA device is present, NumPy inputs are moved to the GPU and returned as a NumPy array (numpy in → numpy out); CuPy inputs stay on the GPU. CPU-only hosts use the NumPy/SciPy path. Enable the GPU via the CuPy section above.
- **Numerics**: inputs are promoted to float64 / complex128 before the covariance determinants. Float32 overflows for multi-channel large-amplitude SAR (`det ~ amp**(2C)`, then squared → `inf/inf = NaN`), so without this promotion multi-channel float32 data returns all-NaN while single-channel works.