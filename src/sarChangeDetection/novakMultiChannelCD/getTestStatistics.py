"""Multi-polarization GLRT change detector (Novák).

Per-pixel generalized likelihood ratio test statistic for equality of the two
local (windowed) polarimetric covariance matrices of a co-registered image pair,
under the complex Wishart / Gaussian model of Conradsen et al. (2003) and the
Novák multi-channel change-detection formulation.

Statistic form — raw Conradsen-style ratio
    Q = det(C1) * det(C2) / det(Cpooled)**2,   Cpooled = (n1*C1 + n2*C2)/(n1+n2)
Q is in [0, 1]: Q = 1 => no change, Q -> 0 => strong change. Smaller values
indicate stronger evidence of change (NOT larger). For a same-shape pair with one
shared kernel, n1 = n2 = kh*kw so Cpooled = (C1+C2)/2 exactly, matching the
reference implementation.

Covariance is Hermitian — the per-pixel outer product is x x^H
(x[..., :, None] * conj(x)[..., None, :]). conj is a no-op for real arrays, so
real-amplitude images work unchanged while complex polarimetric data gets the
correct positive-semidefinite Wishart covariance. This intentional deviation
from the reference (which uses a no-conjugate matmul) is noted here.

Edges are valid-only with a NaN border: convolution runs in mode='valid' so
every output pixel is backed by a fully-overlapping window (no zero-padded
partial windows => no biased/singular border covariances), and the valid
(H_valid, W_valid) result is embedded into a full (H, W) array initialized to
NaN, offset by ((kh-1)//2, (kw-1)//2) to center the window.

Unequal-size / variable-look support is a future extension; the statistic core
``_glrt_statistic`` is already shape-agnostic over leading dims and takes
per-image look counts (n1, n2), so it will be reusable as-is then.
"""

import numpy as np

# cupy is intentionally NOT a declared dependency (see scripts/install_cupy.py
# and README.md). It is imported eagerly at module load via try/except; a usable
# GPU is not required: we detect at import whether a CUDA device is actually
# present (_HAVE_CUPY_GPU = cp.cuda.runtime.getDeviceCount() > 0, treating
# 0/exception as "no GPU") and fall back to the NumPy path if not. When a GPU is
# available, NumPy inputs are auto-moved to the GPU (cupy.asarray), the GLRT
# runs via cupyx/cuSOLVER, and the result is returned as a NumPy array (numpy
# in -> numpy out); cupy inputs stay on the GPU and return a cupy array. The
# module never hard-fails on a CPU-only host.
from scipy.signal import convolve2d as scipyConv2d  # numpy-path convolve (scipy is a declared dep)

try:
    import cupy as cp
    try:
        _HAVE_CUPY_GPU = cp.cuda.runtime.getDeviceCount() > 0
        from cupyx.scipy.signal import convolve2d as cpConvolve2d
    except Exception:
        _HAVE_CUPY_GPU = False
except ImportError:
    cp = None
    _HAVE_CUPY_GPU = False


def _promote(arr, xp):
    """Promote to float64 (real) / complex128 (complex) for numerically stable
    covariance determinants.

    The GLRT statistic involves det(C) ~ (amplitude**2)**C and det(Cpooled)**2,
    which overflows float32 for multi-channel data of even modest amplitude
    (e.g. linear-scale SAR backscatter ~1e8 => det ~ amp**(2*C) far exceeds the
    ~3.4e38 float32 max), yielding inf/inf = NaN across the whole map. float64
    (~1.8e308 max) holds these determinants for any realistic SAR amplitude.
    Single-channel det == amplitude**2 does not overflow float32, which is why
    single-channel worked while multi-channel returned all-NaN. Promoting also
    coerces integer inputs (uint16/int16 backscatter rasters) to a meaningful
    floating covariance. The output map is float64 regardless (see xp.real
    embedding), so this changes only the working precision, not the return dtype.
    """
    if xp.iscomplexobj(arr):
        return xp.asarray(arr, dtype=xp.complex128)
    return xp.asarray(arr, dtype=xp.float64)


def _local_sample_covariances(img1, img2, kernel_size, xp, convolve2d):
    """Per-pixel local C×C sample covariance matrices for both images.

    For each spatial pixel (within the valid region), accumulate the Hermitian
    outer products x x^H over the kh×kw window for each image, producing sample
    covariance estimates of shape (2, H_valid, W_valid, C, C).

    Convolution runs in mode='valid' so every output pixel is backed by a
    fully-overlapping window (no zero-padded partial windows). Each C×C channel
    map is convolved with ones(kernel) and divided by n = kh*kw. conj is a no-op
    for real arrays, so real-amplitude images work unchanged.
    """
    kh, kw = kernel_size
    n = kh * kw
    stack = xp.stack([img1, img2], axis=0)        # (2, H, W, C)
    x = stack[..., :, None]                       # (2, H, W, C, 1)
    xH = xp.conj(stack[..., None, :])             # (2, H, W, 1, C)  (no-op for real)
    outer = x * xH                                # (2, H, W, C, C)
    kernel = xp.ones((kh, kw), dtype=outer.real.dtype)
    H, W, C = img1.shape
    Hv, Wv = H - kh + 1, W - kw + 1
    cov = xp.zeros((2, Hv, Wv, C, C), dtype=outer.dtype)
    for s in range(2):
        for i in range(C):
            for j in range(C):
                cov[s, :, :, i, j] = convolve2d(
                    outer[s, :, :, i, j], kernel, mode='valid')
    cov /= n
    return cov                                    # (2, Hv, Wv, C, C)


def _glrt_statistic(cov1, cov2, n1, n2, xp):
    """GLRT statistic for equality of two C×C covariance matrices given the
    per-image number of looks (n1, n2). Pooled covariance under H0 is the
    looks-weighted average:

        Q = det(C1) * det(C2) / det(Cpooled)**2,
        Cpooled = (n1*C1 + n2*C2) / (n1 + n2)

    This reduces to the reference's (C1+C2)/2 when n1 == n2. Elementwise over
    leading dims, det taken on the last two axes. Hermitian => det is real up
    to rounding; the caller takes xp.real(...).
    """
    pooled = (n1 * cov1 + n2 * cov2) / (n1 + n2)  # == (cov1+cov2)/2 when n1==n2
    det1 = xp.linalg.det(cov1)                     # (...,) over leading dims
    det2 = xp.linalg.det(cov2)
    detP = xp.linalg.det(pooled)
    return (det1 * det2) / (detP ** 2)


def getTestStatistics(img1, img2, kernel_size=(3, 3)):
    """Multi-polarization GLRT test-statistic map between two SAR images.

    Computes, for each pixel, the generalized likelihood ratio test statistic
    for equality of the two local (windowed) polarimetric covariance matrices,
    following the complex Wishart / Gaussian model used in Conradsen et al.
    (2003) and the Novák multi-channel change-detection formulation.

    Parameters
    ----------
    img1, img2 : array-like, shape (H, W, C)
        Two co-registered multi-channel (multi-polarization) SAR images, where
        H, W is the spatial extent and C the number of polarization channels
        (channel-last layout).
        Complex-valued (polarimetric scattering coefficients). May be a NumPy
        or CuPy array. When a CUDA device is available, NumPy inputs are
        automatically moved to the GPU (cupy.asarray) for the computation and
        the result is returned as a NumPy array (numpy in -> numpy out); CuPy
        inputs stay on the GPU and return a CuPy array. On a CPU-only host the
        NumPy path is used regardless. CuPy is an optional, on-demand dependency
        — see scripts/install_cupy.py and README.md. Both arrays must have the
        same shape.
    kernel_size : tuple(int, int), default (3, 3)
        (kh, kw) local-estimation window used to form each pixel's sample
        covariance matrix. The number of looks is n = kh * kw. ``kh`` and
        ``kw`` must be positive odd integers (enforced); this guarantees a
        symmetric, well-defined window center.

    Returns
    -------
    stats : ndarray, shape (H, W), float
        Per-pixel GLRT test statistic (raw Conradsen-style ratio Q). A NumPy
        array on the NumPy path and for NumPy inputs on a GPU host (after a
        round-trip through CuPy); a CuPy array when CuPy inputs are given on a
        GPU host. Smaller values indicate stronger evidence of change (Q = 1 =>
        no change; Q -> 0 => strong change). Border pixels whose window does not
        fully overlap the image are NaN (valid-region-only output).

    Notes
    -----
    Inputs are promoted to float64 (real) / complex128 (complex) before the
    covariance determinants are computed, regardless of the input dtype. The
    statistic involves det(C) ~ (amplitude**2)**C and det(Cpooled)**2, which
    overflows float32 for multi-channel data of even modest amplitude (linear-
    scale SAR backscatter ~1e8 => far past the ~3.4e38 float32 max), yielding
    inf/inf = NaN across the whole map. float64 holds these determinants for
    any realistic SAR amplitude. This is also why single-channel (det ==
    amplitude**2, no overflow) worked while multi-channel returned all-NaN on
    float32 input. The output map is float64 either way; only the working
    precision changes.
    Needs ``kh*kw >= C`` for a non-singular covariance; otherwise det = 0 and
    stats degenerate to NaN/inf at those pixels (documented, not hard-failed).
    The Hermitian covariance path is identical to the real-amplitude path (conj
    is a no-op for real arrays), so no real/complex branching is needed. Q is
    nominally in [0, 1] but may slightly exceed 1 due to floating error; it is
    not clamped, matching the reference. Border pixels are NaN by construction.
    """
    if img1.shape != img2.shape:
        raise ValueError("img1 and img2 must have the same shape")
    if img1.ndim != 3:
        raise ValueError("expected (H, W, C) channel-last arrays")
    kh, kw = kernel_size
    H, W, C = img1.shape
    # kernel sizes MUST be odd and positive: symmetric, well-defined window center.
    if not (isinstance(kh, (int, np.integer)) and isinstance(kw, (int, np.integer))
            and kh > 0 and kw > 0 and kh % 2 == 1 and kw % 2 == 1):
        raise ValueError("kernel_size must be a pair of positive odd integers (kh, kw)")
    Hv, Wv = H - kh + 1, W - kw + 1
    if Hv <= 0 or Wv <= 0:
        raise ValueError("kernel_size larger than image extent")
    n = kh * kw
    off_h, off_w = (kh - 1) // 2, (kw - 1) // 2

    # --- GPU path: auto-accelerate when a CUDA device is present ---
    if _HAVE_CUPY_GPU:
        g1, g2 = _promote(img1, cp), _promote(img2, cp)
        # n < C => singular covariance => det 0 => NaN/inf stats; documented, not hard-failed
        cov = _local_sample_covariances(g1, g2, kernel_size, cp, cpConvolve2d)
        ratio = _glrt_statistic(cov[0], cov[1], n, n, cp)
        out = cp.full((H, W), cp.nan, dtype=float)
        out[off_h:off_h + Hv, off_w:off_w + Wv] = cp.real(ratio)
        # numpy in -> numpy out; cupy in -> cupy out
        if isinstance(img1, cp.ndarray) or isinstance(img2, cp.ndarray):
            return out
        return cp.asnumpy(out)

    # --- CPU / NumPy fallback ---
    a1, a2 = _promote(img1, np), _promote(img2, np)
    cov = _local_sample_covariances(a1, a2, kernel_size, np, scipyConv2d)
    ratio = _glrt_statistic(cov[0], cov[1], n, n, np)
    out = np.full((H, W), np.nan, dtype=float)
    out[off_h:off_h + Hv, off_w:off_w + Wv] = np.real(ratio)
    return out


def getSingleTestStatistic(x1, x2):
    """Single GLRT test statistic between two sets of C-channel samples.

    Computes the same Conradsen/Novák generalized likelihood ratio statistic as
    ``getTestStatistics``, but once over two explicit sample populations
    (rather than per-window over an image): given the two (n, C) sample
    matrices it forms each population's C×C sample covariance and evaluates

        Q = det(C1) * det(C2) / det(Cpooled)**2,
        Cpooled = (n1*C1 + n2*C2) / (n1 + n2)

    NumPy-only path (no CuPy/GPU dispatch).

    Parameters
    ----------
    x1, x2 : array-like, shape (n1, C) and (n2, C)
        Two sets of multi-channel (multi-polarization) samples, where ``n`` is
        the number of samples (looks) and ``C`` the number of polarization
        channels (channel-last over the trailing axis). Complex-valued
        (polarimetric scattering coefficients) or real. Both arrays must have
        the same number of channels (``x1.shape[1] == x2.shape[1]``).

    Returns
    -------
    stat : numpy.float32
        The GLRT test statistic Q (raw Conradsen-style ratio) as a single
        float32 scalar. Smaller values indicate stronger evidence of change
        (Q = 1 => no change; Q -> 0 => strong change). Q is nominally in
        [0, 1] but may slightly exceed 1 due to floating error; it is not
        clamped, matching the reference.

    Notes
    -----
    Inputs are promoted to float64 (real) / complex128 (complex) before the
    covariance determinants are computed, regardless of the input dtype, for
    the same overflow reason as ``getTestStatistics`` (det(C) ~ amp**(2C) then
    squared overflows float32 for multi-channel data). Only the final scalar
    is cast to float32. The Hermitian sample covariance is (1/n) * X^H X;
    ``conj`` is a no-op for real arrays, so real-amplitude samples work
    unchanged. Needs ``n >= C`` per population for a non-singular covariance;
    otherwise det = 0 and the statistic degenerates to NaN/inf (documented,
    not hard-failed).
    """
    if x1.ndim != 2 or x2.ndim != 2:
        raise ValueError("expected (n, C) 2-D arrays")
    if x1.shape[1] != x2.shape[1]:
        raise ValueError("x1 and x2 must have the same number of channels")
    n1, n2 = x1.shape[0], x2.shape[0]
    a1, a2 = _promote(x1, np), _promote(x2, np)
    # Hermitian sample covariance: C = (1/n) * X^H X. conj is a no-op for real
    # arrays => real-amplitude samples work unchanged (same as the image path).
    cov1 = (a1.conj().T @ a1) / n1   # (C, C)
    cov2 = (a2.conj().T @ a2) / n2
    q = _glrt_statistic(cov1, cov2, n1, n2, np)   # scalar; complex, imag ~ 0
    return np.float32(np.real(q))


def _getSingleTestStatistic_selfcheck():
    """Equivalence check: getSingleTestStatistic vs a single pixel of
    getTestStatistics.

    With a kernel_size of (H, W) the kernel-based map has exactly one valid
    output pixel (at ((kh-1)//2, (kw-1)//2)); reshaping that image to (-1, C)
    yields the same window samples getSingleTestStatistic consumes, so the two
    must agree up to the float32 cast getSingleTestStatistic applies. NumPy
    only -> runs on every host (never skipped, unlike the cupy self-check).
    """
    rng = np.random.default_rng(0)
    H, W, C = 3, 3, 3
    a = rng.standard_normal((H, W, C)) + 1j * rng.standard_normal((H, W, C))
    b = rng.standard_normal((H, W, C)) + 1j * rng.standard_normal((H, W, C))
    kernel_size = (H, W)            # kh == H, kw == W => exactly one valid pixel
    kh, kw = kernel_size
    off_h, off_w = (kh - 1) // 2, (kw - 1) // 2

    # Reference: the kernel-based map (float64), single valid pixel.
    ref = getTestStatistics(a, b, kernel_size=kernel_size)
    ref_val = float(ref[off_h, off_w])

    # Got: the population statistic from the same window's samples (float32).
    x1 = a.reshape(-1, C)
    x2 = b.reshape(-1, C)
    got = getSingleTestStatistic(x1, x2)
    got_val = float(got)

    dtype_ok = got.dtype == np.float32
    ref_finite = np.isfinite(ref_val)
    match = bool(np.isclose(got_val, ref_val, rtol=1e-5, atol=1e-6))

    ok = bool(dtype_ok and ref_finite and match)
    print(f"getSingleTestStatistic self-check: {'PASS' if ok else 'FAIL'} "
          f"(dtype={dtype_ok}, ref_finite={ref_finite}, match={match}, "
          f"ref={ref_val:.6e}, got={got_val:.6e})")
    if not ok:
        print("ref dtype", ref.dtype, "got dtype", got.dtype)
    return ok


def _getTestStatistics_selfcheck():
    """Equivalence check: cupy getTestStatistics vs the NumPy reference.

    cupyx convolve2d / cuSOLVER det and scipy/NumPy use different backends, so
    results agree closely but not bit-for-bit; we check allclose on the finite
    valid region, not array_equal. Skipped (and reported as such) when no GPU.
    """
    if not _HAVE_CUPY_GPU:
        print("getTestStatistics self-check: skipped (no GPU available — NumPy path only).")
        return True

    rng = np.random.default_rng(0)
    H, W, C = 12, 12, 3
    a = rng.standard_normal((H, W, C)) + 1j * rng.standard_normal((H, W, C))
    b = rng.standard_normal((H, W, C)) + 1j * rng.standard_normal((H, W, C))
    kernel_size = (3, 3)
    kh, kw = kernel_size
    n = kh * kw
    Hv, Wv = H - kh + 1, W - kw + 1
    off_h, off_w = (kh - 1) // 2, (kw - 1) // 2

    # NumPy reference via the helpers directly (bypasses the auto-accelerate dispatch)
    cov = _local_sample_covariances(a, b, kernel_size, np, scipyConv2d)
    ref_ratio = _glrt_statistic(cov[0], cov[1], n, n, np)
    ref = np.full((H, W), np.nan, dtype=float)
    ref[off_h:off_h + Hv, off_w:off_w + Wv] = np.real(ref_ratio)

    # GPU path via the public function (numpy in -> numpy out)
    got = getTestStatistics(a, b, kernel_size=kernel_size)

    shape_ok = ref.shape == got.shape == (H, W)
    dtype_ok = got.dtype == ref.dtype == np.float64
    finite = np.isfinite(ref) & np.isfinite(got)
    vals_close = bool(np.allclose(got[finite], ref[finite], rtol=1e-4, atol=1e-5))
    max_diff = float(np.nanmax(np.abs(got - ref)))

    ok = bool(shape_ok and dtype_ok and vals_close)
    print(f"getTestStatistics self-check: {'PASS' if ok else 'FAIL'} "
          f"(shape={shape_ok}, dtype={dtype_ok}, values_close={vals_close}, "
          f"max_abs_diff={max_diff:.3e})")
    if not ok:
        print("ref shape", ref.shape, "got shape", got.shape)
    return ok


if __name__ == "__main__":
    _getTestStatistics_selfcheck()
    _getSingleTestStatistic_selfcheck()