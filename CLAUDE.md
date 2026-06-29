# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

SAR (synthetic-aperture radar) change-detection library. Currently one detector subpackage: the Novák multi-polarization GLRT in `novakMultiChannelCD`, exposing `getTestStatistics` (per-pixel map over a co-registered image pair) and `getSingleTestStatistic` (the same statistic once, over two `(n, C)` sample populations). The package is `sarChangeDetection` (import name), distributed as `sar-change-detection`. Channel-last `(H, W, C)` arrays throughout the image path.

## Commands

This project is [uv](https://docs.astral.sh/uv/)-managed.

```powershell
uv sync                              # create .venv, editable install + runtime deps (numpy, numba, scipy); dev dep matplotlib
uv run python -m sarChangeDetection.novakMultiChannelCD.getTestStatistics   # self-checks: cupy-vs-NumPy getTestStatistics (skips with "no GPU" on CPU hosts) + getSingleTestStatistic-vs-getTestStatistics single-pixel (runs on every host)
uv run python scripts/install_cupy.py                                        # detect GPU/CUDA and install the matching CuPy wheel (dev only; not shipped in the wheel)
```

There is no test framework, linter, or formatter configured. The only built-in verification is the `__main__` self-check above — run it after touching the GLRT code.

## Architecture

### Dual CPU/GPU backend via array-library dispatch

The core design is **one code path parameterized by the array module**. `_local_sample_covariances` and `_glrt_statistic` take `xp` (numpy or cupy) and a `convolve2d` callable, so the same code runs on both backends. CuPy is **deliberately not a declared dependency** (the correct wheel depends on host GPU/CUDA and can't be picked statically). Instead:

- `cupy` is imported eagerly at module load under `try/except`; `_HAVE_CUPY_GPU = cp.cuda.runtime.getDeviceCount() > 0` decides at import time (CPU-only hosts fall back to NumPy, never hard-fail).
- `getTestStatistics` dispatches on `_HAVE_CUPY_GPU`: NumPy inputs on a GPU host are auto-moved to the GPU and returned as NumPy (`numpy in → numpy out`); CuPy inputs stay on the GPU (`cupy in → cupy out`).
- `getSingleTestStatistic` is **NumPy-only** (no GPU dispatch): it computes the scalar statistic once over two `(n, C)` sample matrices via the Hermitian sample covariance `(1/n) X^H X`, reusing `_promote` and `_glrt_statistic` (with per-population look counts `n1`, `n2`).
- `scripts/install_cupy.py` (stdlib-only) detects NVIDIA GPU + CUDA Toolkit and installs the right wheel (`cupy-cuda{11,12,13}x`, or `[ctk]` bundle when there's no toolkit). It scrubs conflicting cupy dists first — only one may be installed.

### Numerics that are easy to break

- **Inputs are promoted to float64 / complex128** in `_promote` before covariance determinants. Do not skip this: the GLRT computes `det(C) ~ amp**(2C)` then squares it, which overflows float32 for multi-channel SAR and yields `inf/inf = NaN` across the whole map. Single-channel happens to work on float32 (det == amp**2), which masked this bug historically.
- **Hermitian covariance**: the outer product is `x x^H` using `xp.conj`. `conj` is a no-op for real arrays, so real-amplitude images work unchanged — do not add a real/complex branch.
- **Valid-region-only with NaN border**: convolution uses `mode='valid'` (no zero-padded partial windows → no biased/singular border covariances), and the `(Hv, Wv)` result is embedded into a full `(H, W)` array of NaN offset by `((kh-1)//2, (kw-1)//2)`. Preserve this when changing kernel handling.
- Q is in [0, 1] nominally (1 = no change, →0 = strong change) but is **not clamped** — matching the reference; don't add clamping.

### Extensibility hook

`_glrt_statistic(cov1, cov2, n1, n2, xp)` is shape-agnostic over leading dims and takes per-image look counts (`n1`, `n2`), reducing to the reference's `(C1+C2)/2` when `n1 == n2`. `getSingleTestStatistic` already exercises the unequal-looks path; the remaining planned extension is per-window variable looks in `getTestStatistics`, which should reuse `_glrt_statistic` as-is.

## Conventions

- New detectors should be a subpackage under `src/sarChangeDetection/` mirroring `novakMultiChannelCD/` (subpkg `__init__` re-exports the public function, `__all__` listed). Keep public functions lowercase camelCase (`getTestStatistics`).
- `requires-python = ">=3.10,<3.15"`. Runtime deps stay limited to `numpy`, `numba`, `scipy>=1.15.3`; `matplotlib` is a dev-only dependency (plotting during development, not a runtime dep). CuPy must never be added to `[project.dependencies]`.