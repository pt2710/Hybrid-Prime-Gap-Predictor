import time
import math
from math import isqrt
from numba import njit, prange
import numpy as np
from predict_gaps import PredictPrimeGaps
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import plotting

try:
    import pyopencl as cl
except ImportError:
    cl = None

USE_GPU = False
try:
    import cupy as cp
except ImportError:
    cp = None
use_gpu = USE_GPU and ((cp is not None) or (cl is not None))

KD_CONST = 1.0
SOI = 100

_cl_ctx = None
_cl_queue = None
_cl_program = None

@njit(cache=True)
def modexp(a: np.int64, d: np.int64, n: np.int64) -> np.int64:
    """Modular exponentiation (a^d % n)."""
    result = np.int64(1)
    a = a % n
    while d > 0:
        if d & 1:
            result = (result * a) % n
        a = (a * a) % n
        d >>= 1
    return result

@njit(cache=True)
def is_prime_njit(n: np.int64) -> bool:
    """Numba-accelerated Miller–Rabin for 32-bit range (bases 2,7,61)."""
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11):
        if n == p:
            return True
        if n % p == 0:
            return False
    d = n - 1
    r = 0
    while (d & 1) == 0:
        d >>= 1
        r += 1
    for a in (2, 7, 61):
        if a % n == 0:
            return True
        x = modexp(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = (x * x) % n
            if x == n - 1:
                break
        else:
            return False
    return True

def _sieve_small(limit: int):
    """Return array of primes up to 'limit' (inclusive)."""
    sieve = np.ones(limit+1, dtype=bool)
    sieve[:2] = False
    for p in range(2, isqrt(limit) + 1):
        if sieve[p]:
            sieve[p*p : limit+1 : p] = False
    return np.nonzero(sieve)[0]

_SMALL_PRIMES = _sieve_small(50000)

@njit(cache=True, parallel=True)
def segmented_sieve_njit(low: np.int64, high: np.int64, small_pr: np.ndarray) -> np.ndarray:
    """Numba-segmented sieve on [low,high] with parallel marking."""
    size = high - low + 1
    mask = np.ones(size, np.uint8)
    lim = int(math.sqrt(high))
    for i in prange(small_pr.shape[0]):
        p = small_pr[i]
        if p > lim:
            continue
        start = p*p if p*p >= low else ((low + p - 1) // p) * p
        for idx in range(start - low, size, p):
            mask[idx] = 0
    if low == 0:
        mask[:min(2, size)] = 0
    elif low == 1:
        mask[0] = 0
    return mask

def segmented_sieve(low: int, high: int, use_gpu: bool = False) -> np.ndarray:
    """Segmented sieve with optional GPU (CuPy or OpenCL) acceleration."""
    size = high - low + 1
    if use_gpu:
        if cp is not None:
            mask = cp.ones(size, dtype=bool)
            lim = math.isqrt(high)
            for p in _SMALL_PRIMES:
                if p > lim:
                    break
                start = p*p if p*p >= low else ((low + p - 1) // p) * p
                mask[start - low : size : p] = False
            if low == 0:
                mask[0: min(2, size)] = False
            elif low == 1:
                mask[0] = False
            return cp.asnumpy(mask)

        elif cl is not None:
            global _cl_ctx, _cl_queue, _cl_program
            if _cl_ctx is None:
                platforms = cl.get_platforms()
                dev = None
                for platform in platforms:
                    for device in platform.get_devices(device_type=cl.device_type.GPU):
                        if dev is None or 'AMD' in device.vendor or 'AMD' in device.name:
                            dev = device
                if dev is not None:
                    _cl_ctx = cl.Context([dev])
                    _cl_queue = cl.CommandQueue(_cl_ctx)
                    kernel_code = """
                    __kernel void sieve_seg(__global const int* primes, __global uchar* mask, int low, int size) {
                        int gid = get_global_id(0);
                        int p = primes[gid];
                        int start = p*p >= low ? p*p : ((low + p - 1) / p) * p;
                        for (int x = start; x < low + size; x += p) {
                            mask[x - low] = 0;
                        }
                    }
                    """
                    _cl_program = cl.Program(_cl_ctx, kernel_code).build()
            lim = math.isqrt(high)
            primes_list = [p for p in _SMALL_PRIMES if p <= lim]
            n_pr = len(primes_list)
            if n_pr == 0:
                mask_np = np.ones(size, dtype=np.uint8)
                if low == 0:
                    mask_np[0: min(2, size)] = 0
                elif low == 1:
                    mask_np[0] = 0
                return mask_np
            primes_buf = cl.Buffer(_cl_ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                                   hostbuf=np.array(primes_list, dtype=np.int32))
            mask_buf = cl.Buffer(_cl_ctx, cl.mem_flags.READ_WRITE, size)
            cl.enqueue_fill_buffer(_cl_queue, mask_buf, np.uint8(1), 0, size)
            _cl_program.sieve_seg(_cl_queue, (n_pr,), None, primes_buf, mask_buf,
                                  np.int32(low), np.int32(size))
            mask_np = np.empty(size, dtype=np.uint8)
            cl.enqueue_copy(_cl_queue, mask_np, mask_buf)
            _cl_queue.finish()
            if low == 0:
                mask_np[0: min(2, size)] = 0
            elif low == 1:
                mask_np[0] = 0
            return mask_np

    return segmented_sieve_njit(np.int64(low), np.int64(high), _SMALL_PRIMES)

def is_prime(n: int) -> bool:
    """Deterministic Miller–Rabin for up to 64-bit."""
    n = int(n)
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n == p:
            return True
        if n % p == 0:
            return False
    d = n - 1
    r = 0
    while d & 1 == 0:
        d >>= 1
        r += 1
    bases = (2, 325, 9375, 28178, 450775, 9780504, 1795265022)
    for a in bases:
        if a % n == 0:
            return True
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = (x * x) % n
            if x == n - 1:
                break
        else:
            return False
    return True

def next_prime_c_sieve(n: int, block: int = 1024) -> int:
    """Return first prime > n using a blockwise sieve."""
    m = n + 1
    while True:
        mask = segmented_sieve_njit(np.int64(m), np.int64(m + block - 1), _SMALL_PRIMES)
        if mask.any():
            return int(m + int(mask.argmax()))
        m += block

def next_prime_mr(n: int) -> int:
    """Return first prime > n via Miller–Rabin scan."""
    m = n + 1
    while True:
        if is_prime(np.int64(m)):
            return m
        m += 1

@njit(cache=True)
def next_prime_c_sieve_njit(n: np.int64, block: np.int64, small_pr: np.ndarray) -> np.int64:
    """Return first prime > n using a blockwise sieve (Numba)."""
    m = n + 1
    while True:
        mask = segmented_sieve_njit(m, m + block - 1, small_pr)
        for idx in range(mask.shape[0]):
            if mask[idx]:
                return m + idx
        m += block

@njit(cache=True)
def next_prime_mr_njit(n: np.int64) -> np.int64:
    """Return first prime > n via Miller–Rabin (Numba)."""
    m = n + 1
    while True:
        if is_prime_njit(np.int64(m)):
            return np.int64(m)
        m += 1

def make_hybrid(ppg, pad: int = 64, use_gpu: bool = False):
    """Construct hybrid next-prime function using predictive generator."""
    rbuf = max(int(4 * ppg.rstd), 32)
    resid_avg = 0.0
    min_pad = 2
    max_pad = 4096
    min_rbuf = 32
    max_rbuf = 4096

    def next_prime_hybrid(n: int) -> int:
        nonlocal pad, rbuf, resid_avg
        h_row = ppg.H[0]
        gap_pred = 2 * round((h_row.dot(ppg.blend)) / 2)
        if gap_pred < 2:
            gap_pred = 2
        approx = n + int(gap_pred)
        guess = approx if (approx % 2 == 1) else (approx + 1)
        if guess <= n:
            guess = n + 1 + ((n + 1) & 1)

        found_prime = None
        if gap_pred <= 32:
            found_prime = int(next_prime_mr_njit(np.int64(n)))
            expansions = 0

        if found_prime is None:
            low = n + 1
            high = n + max(pad, guess - n)
            mask = segmented_sieve(low, high, use_gpu)
            if mask.any():
                idx = int(cp.asnumpy(mask.argmax()) if (use_gpu and cp is not None) else mask.argmax())
                found_prime = low + idx
                expansions = 0
            else:
                step = max(pad, rbuf)
                expansions = 0
                while True:
                    expansions += 1
                    low = high + 1
                    high = high + step
                    mask = segmented_sieve(low, high, use_gpu)
                    if mask.any():
                        idx = int(cp.asnumpy(mask.argmax()) if (use_gpu and cp is not None) else mask.argmax())
                        found_prime = low + idx
                        break
                    step <<= 1

        actual_gap = found_prime - n
        resid = actual_gap - gap_pred
        resid_avg = 0.9 * resid_avg + 0.1 * resid
        pad = max(min_pad, min(max_pad, pad + int(resid_avg * 0.5)))
        if expansions > 1:
            rbuf = min(max_rbuf, rbuf * 2)
        elif expansions == 0:
            rbuf = max(min_rbuf, rbuf // 2)

        return found_prime

    return next_prime_hybrid

def build_S_analytic(L: float, N: int, num_primes: int = 100):
    """Analytic parity fluctuation field S(x)."""
    h = L / (N + 1)
    x = np.linspace(h, N * h, N)
    primes = _SMALL_PRIMES[:num_primes]
    sign = KD_CONST * ((-1) ** np.arange(num_primes))
    exp_term = np.exp(-primes[:, None] * x + 1e-12)
    S = np.sum(np.log(1 - sign[:, None] * exp_term), axis=0)
    return -KD_CONST * S

def build_S_dfi(L: float, N: int, S_base, SOI=SOI):
    """Dynamic Fluctuation Index field (DFI) on [0, L] with N points."""
    S_base_np = np.array(S_base)
    rolled = np.stack([np.roll(S_base_np, i) for i in range(5)], axis=1)
    x_n = rolled.sum(axis=1)
    x_i = rolled[:, 0]
    x_r = x_n - x_i
    sigma = (x_n * 4) / (x_r * 5)
    V0 = SOI / rolled.shape[1]
    S_raw = (V0 * sigma) - V0
    old_grid = np.linspace(0, L, N)
    new_grid = (L / (N + 1)) * np.arange(1, N + 1)
    S_dfi = np.interp(new_grid, old_grid, S_raw)
    return S_dfi

def compute_spectral_gap(L: float, N: int, S_field):
    """Spectral gap (λ1 - λ0) of Hamiltonian defined by S_field."""
    S_np = np.array(S_field)
    h = L / (N + 1)
    piE = math.pi * np.exp(-S_np / KD_CONST)
    ln_piE = np.log(piE)
    lap = np.zeros_like(ln_piE)
    lap[1:-1] = (ln_piE[2:] - 2 * ln_piE[1:-1] + ln_piE[:-2]) / (h ** 2)
    main = (2.0 / h ** 2 + 0.5 * lap).astype(np.float64)
    off = (-1.0 / h ** 2) * np.ones(N - 1, dtype=np.float64)
    from scipy.linalg import eigh_tridiagonal
    eigvals = eigh_tridiagonal(main, off, select='i', select_range=(0, 1),
                                eigvals_only=True, check_finite=False)
    return float(eigvals[1] - eigvals[0])

if __name__ == "__main__":
    # generate sample primes and gaps
    limit = 2_000_000
    sieve = np.ones(limit + 1, dtype=bool)
    sieve[:2] = False
    for p in range(2, isqrt(limit) + 1):
        if sieve[p]:
            sieve[p*p : limit + 1 : p] = False
    primes = np.nonzero(sieve)[0]

    idx_center = 100_000 - 1
    ws = 50_000
    pw = primes[idx_center - ws : idx_center + ws + 1]
    gaps = np.diff(pw)

    # initialize predictor and hybrid
    ppg = PredictPrimeGaps(N=len(gaps), alpha=0.985)
    next_prime = make_hybrid(ppg, pad=64, use_gpu=use_gpu)

    # correctness tests
    test_vals = [100, 1_000, 10**6] + ([pw[-1], pw[-1] + 1] if 'pw' in locals() else [])
    correct = True
    for n in test_vals:
        h = next_prime(n)
        ref = next_prime_c_sieve(n, block=1024)
        print(f"n={n}, hybrid -> {h}, sieve -> {ref}")
        if h != ref:
            print("Mismatch!")
            correct = False
    print("Correctness test passed:", correct)

    # benchmark timings
    start_prime = 15_485_863
    t0 = time.perf_counter()
    p = start_prime
    for _ in range(1000):
        p = next_prime_c_sieve(p)
    t_c_sieve = time.perf_counter() - t0

    t0 = time.perf_counter()
    p = start_prime
    for _ in range(1000):
        p = next_prime_mr(p)
    t_mr = time.perf_counter() - t0

    t0 = time.perf_counter()
    p = start_prime
    for _ in range(1000):
        p = next_prime(p)
    t_hybrid = time.perf_counter() - t0


    # 1. time-series comparison (no shift)
    sample_primes = primes[:25000 + 1]
    actual_gaps = np.diff(sample_primes)
    # directly call predict_gap; it already returns 1 for idx 0,1,2 and even ≥2 thereafter
    pred_gaps = np.array([ppg.predict_gap(i) for i in range(len(actual_gaps))])
    
    # 1. time-series comparison
    plt.figure(figsize=(8, 6))
    plt.plot(np.arange(len(actual_gaps)), actual_gaps,
             label='Actual Gaps', color='yellow', linestyle='-')
    plt.plot(np.arange(len(pred_gaps)), pred_gaps,
             label='Predicted Gaps', color='red', linestyle='--')
    plt.xlabel('Index')
    plt.ylabel('Gap Value')
    plt.title('Actual vs. Predicted Gaps')
    plt.legend()
    plt.tight_layout()
    plt.savefig("fig_actual_vs_predicted.png")
    plt.close()

    # 2. hybrid guess vs actual scatter
    approx_primes = sample_primes[:-1] + pred_gaps
    true_primes   = sample_primes[1:]
    plt.figure(figsize=(8, 6))
    plt.scatter(true_primes, approx_primes, s=1, label='Predicted Primes')
    m, M = min(true_primes), max(true_primes)
    plt.plot([m, M], [m, M], 'y--', label='Ideal Line')
    plt.xlabel('True Primes')
    plt.ylabel('Approx Primes')
    plt.title('Hybrid Guess vs. Actual Primes')
    plt.legend()
    plt.tight_layout()
    plt.savefig("fig_guess_vs_actual_scatter.png")
    plt.close()

    # 3. histogram of guess errors
    errors = true_primes - approx_primes
    plt.figure(figsize=(6, 4))
    plt.hist(errors,
             bins=range(int(errors.min()), int(errors.max()) + 2),
             align='left', edgecolor='black',
             label='Error Distribution')
    plt.xlabel('Error (True − Approx)')
    plt.ylabel('Count')
    plt.title('Distribution of Hybrid Guess Errors')
    plt.legend()
    plt.tight_layout()
    plt.savefig("fig_guess_error_hist.png", dpi=300)
    plt.close()

    # --- NEW: Gap size distribution bar chart ---
    gap_counts = pd.Series(actual_gaps).value_counts().sort_index()
    df_gap_counts = gap_counts.rename_axis('GapSize').reset_index(name='Count')
    df_gap_counts.to_csv("gap_size_distribution.csv", index=False)

    plt.figure(figsize=(10, 6))
    gap_counts.plot(kind='bar', label='Gap Count')
    plt.xlabel('Gap Size')
    plt.ylabel('Count')
    plt.title('Prime Gap Size Distribution')
    plt.legend()
    plt.tight_layout()
    plt.savefig("gap_size_distribution.png")
    plt.close()

    # 4. export gap predictions with domain labels
    from numbers_domains import NumbersDomains
    nd = NumbersDomains()
    domains = np.array([nd.evens(g) for g in actual_gaps])
    df_gaps = pd.DataFrame({
        'Index': np.arange(len(actual_gaps)),
        'ActualGap': actual_gaps,
        'PredictedGap': pred_gaps,
        'Domain': domains
    })
    df_gaps.to_csv("gaps_domains_predictions.csv", index=False)

    # 4. distribution of domains
    domain_counts = df_gaps['Domain'].value_counts().sort_index()
    plt.figure(figsize=(6, 4))
    domain_counts.plot(kind='bar', label='Domain Count')
    plt.xlabel('Domain')
    plt.ylabel('Count')
    plt.title('Distribution of Prime Gap Domains')
    plt.legend()
    plt.tight_layout()
    plt.savefig("domain_distribution.png")
    plt.close()

    # 6. residuals by domain (box + swarm)
    residuals = actual_gaps - pred_gaps
    df_res = pd.DataFrame({
        'Index':    np.arange(len(residuals)),
        'Residual': residuals,
        'Domain':   domains
    })
    df_res.to_csv("residuals_by_domain.csv", index=False)

    plt.figure(figsize=(6, 4))
    # boxplot
    df_res.boxplot(
        column='Residual',
        by='Domain',
        showfliers=False,
        grid=False
    )
    # swarm overlay
    valid_domains = sorted(d for d in df_res['Domain'].unique() if d is not None)
    for i, d in enumerate(valid_domains, start=1):
        vals = df_res.loc[df_res.Domain == d, 'Residual']
        x = np.random.normal(loc=i, scale=0.04, size=len(vals))
        plt.plot(x, vals, 'o', alpha=0.3, markersize=2)
    # add legend for swarm points
    plt.plot([], [], 'o', color='black', alpha=0.3, markersize=5, label='Data Points')
    plt.suptitle("")
    plt.xlabel("Domain")
    plt.ylabel("Residual")
    plt.title("Residuals by Domain (box + swarm)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("residuals_by_domain_swarm.png")
    plt.close()

    # 5. Export only non-zero even-domain weights, with reindexed domain codes
    blend = ppg.blend
    unity_weight = blend[0]

    # extract only the even-domain weights (skip blend[0]), keep those > 0
    even_weights = ppg.w
    nz_mask = even_weights > 0
    nz_weights = even_weights[nz_mask]

    # renumber domains from 2 up to 2+len(nz_weights)-1
    domain_codes = [1] + list(range(2, 2 + len(nz_weights)))
    weights = [unity_weight] + nz_weights.tolist()

    df_weights = pd.DataFrame({
        'Domain': domain_codes,
        'Weight': weights
    })
    df_weights.to_csv("weights.csv", index=False)

    # 5. blend weights by domain
    df_weights = pd.DataFrame({
        'Domain': df_weights['Domain'].astype(str),
        'Weight': df_weights['Weight']
    })
    plt.figure(figsize=(6, 4))
    plt.bar(df_weights['Domain'], df_weights['Weight'], label='Blend Weight')
    plt.xlabel('Domain Code')
    plt.ylabel('Weight')
    plt.title('Blend Weights by Domain\n(1=Unity; 2–{}=Even Domains)'.format(1+len(nz_weights)))
    plt.legend()
    plt.tight_layout()
    plt.savefig("weights_distribution.png")
    plt.close()

    # 4. Benchmark timings table and bar chart
    benchmarks = {
        "C-sieve":      t_c_sieve / 1000 * 1e6,
        "Miller–Rabin": t_mr     / 1000 * 1e6,
        "Hybrid":       t_hybrid / 1000 * 1e6,
    }
    plotting.save_benchmark_table(benchmarks, "benchmark_times.csv")
    plotting.plot_benchmark(benchmarks, "fig_benchmark.png")
    
    # 7. spectral field plot (already has legend)
    Nf = ppg.H.shape[0]
    x = np.linspace(1, Nf, Nf)
    S_field = build_S_analytic(KD_CONST, Nf)
    piE = np.pi * np.exp(-S_field / KD_CONST)
    plt.figure(figsize=(8, 6))
    plt.plot(x, S_field, label='Spectral Field')
    plt.plot(x, piE,    label='Entropic Wavefunction')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Spectral Field and Entropic Wavefunction')
    plt.legend()
    plt.tight_layout()
    plt.savefig("fig_spectral_field.png")
    plt.close()

    # 8. print speed summary
    print(f"C-sieve: {t_c_sieve:.6f} s, MR: {t_mr:.6f} s, Hybrid: {t_hybrid:.6f} s")
    print(f"C-sieve time for 1000 primes: {t_c_sieve:.4f} s, MR time: {t_mr:.4f} s, Hybrid time: {t_hybrid:.4f} s")
    print(f"Average per prime: C-sieve {t_c_sieve/1000*1e6:.2f} µs, MR {t_mr/1000*1e6:.2f} µs, Hybrid {t_hybrid/1000*1e6:.2f} µs")

    # 9. compute and save performance metrics
    mse = mean_squared_error(true_primes, approx_primes)
    mae = mean_absolute_error(true_primes, approx_primes)
    tolerance = 2
    correct_preds = np.abs(true_primes - approx_primes) <= tolerance
    accuracy = np.mean(correct_preds)
    recall = np.sum(correct_preds) / len(true_primes)
    reduction = 1 - (len(approx_primes) / len(true_primes))
    num_gaps = np.sum(correct_preds)

    metrics = pd.DataFrame({
        "Metric": ["Accuracy", "Recall", "Reduction", "MSE", "MAE", "Gaps Covered"],
        "Value": [accuracy, recall, reduction, mse, mae, num_gaps]
    })
    metrics.to_csv("metrics.csv", index=False)

    print(f"Accuracy: {accuracy:.6f}")
    print(f"Recall: {recall:.6f}")
    print(f"Reduction: {reduction:.6f}")
    print(f"MSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"Gaps Covered: {num_gaps}")
