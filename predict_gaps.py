import math
import numpy as np
from scipy.optimize import nnls
from numba import njit, prange
from numbers_domains import NumbersDomains

class PredictPrimeGaps:
    def __init__(self, N=1000, alpha=0.95, M=0):
        if alpha >= 1.0:
            raise ValueError("Alpha must be less than 1.0 to avoid data leakage.")
        
        self.alpha = alpha
        self.M = M
        self.domains = NumbersDomains()
        
        # 1) Generate the first N+1 primes and their raw gaps (including the initial gap=1)
        primes = PredictPrimeGaps.generate_primes(N + 1)
        raw_gaps = np.diff(primes)
        self.prime_gaps = raw_gaps
        self.N = len(self.prime_gaps)
        
        # 2) Generate composites and their gaps for baseline smoothing
        comps = PredictPrimeGaps.generate_composites(N + 1)
        raw_comp = np.diff(comps)
        # align composite gaps array to the same length
        self.comp_gaps = raw_comp[: self.N]
        
        # 3) Build baseline (with unity override), domain‐specific fit, etc.
        self.h_best = self.baseline_smoothed_signal()
        self.h_final, self.w = self.domain_specific_copies()
        self.rstd = self.calculate_rstd()
        self.H = self.calculate_H()
        self.blend = np.concatenate(([1.0], self.w))

    @staticmethod
    @njit
    def generate_primes(N):
        primes = np.empty(N, dtype=np.int64)
        count = 0
        candidate = 2
        while count < N:
            is_prime = True
            for p in primes[:count]:
                if p * p > candidate:
                    break
                if candidate % p == 0:
                    is_prime = False
                    break
            if is_prime:
                primes[count] = candidate
                count += 1
            candidate += 1
        return primes

    @staticmethod
    @njit
    def generate_composites(N):
        comps = np.empty(N, dtype=np.int64)
        count = 0
        candidate = 4
        while count < N:
            for i in range(2, int(math.sqrt(candidate)) + 1):
                if candidate % i == 0:
                    comps[count] = candidate
                    count += 1
                    break
            candidate += 1
        return comps

    def baseline_smoothed_signal(self):
        mixed = (self.alpha * self.prime_gaps) + ((1 - self.alpha) * self.comp_gaps)
        sm = 2 * (np.round(mixed) / 2)
        sm = np.maximum(2, sm).astype(int)
        # override the very first unity gap (g=1) back to 1
        unity_positions = (self.prime_gaps == 1)
        sm[unity_positions] = 1
        return sm

    def domain_specific_copies(self):
        # build X only for the six even‐gap domains; g==1 entries have X-row = zeros
        doms = np.array([self.domains.evens(int(g)) for g in self.prime_gaps], dtype=object)
        X = np.zeros((self.N, 6))
        max_gap = np.max(self.prime_gaps[self.prime_gaps > 1])  # ignore the unity gap for scaling
        for j in prange(6):
            m = (doms == j)
            X[m, j] = (self.prime_gaps[m] / max_gap) * self.h_best[m]
        y = self.prime_gaps - self.h_best
        w, _ = nnls(X, y)
        h_fin = self.h_best + X.dot(w)
        h_fin = 2 * (np.round(h_fin) / 2)
        h_fin = np.maximum(2, h_fin).astype(int)
        # restore unity at g=1
        h_fin[self.prime_gaps == 1] = 1
        return h_fin, w

    def calculate_rstd(self):
        return np.std(self.prime_gaps - self.h_final)

    def calculate_H(self):
        doms = np.array([self.domains.evens(int(g)) for g in self.prime_gaps], dtype=object)
        X = np.zeros((self.N, 6))
        max_gap = np.max(self.prime_gaps[self.prime_gaps > 1])
        for j in prange(6):
            m = (doms == j)
            X[m, j] = (self.prime_gaps[m] / max_gap) * self.h_best[m]
        return np.column_stack([self.h_best, X])

    def predict_gap(self, idx: int) -> int:
        # special‐case the unity gap at the very first interval
        if idx < self.N and self.prime_gaps[idx] == 1:
            return 1

        # otherwise delegate to the even‐gap model
        vec = self.H[idx] if idx < self.N else self.H[-1]
        val = float(vec @ self.blend)
        gap = 2 * round(val / 2)
        return max(2, int(gap))
