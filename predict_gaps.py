# Updated `predict_gaps.py` with detailed docstrings and inline comments,
# including a fallback stub for NumbersDomains to allow execution in this environment.

"""
predict_gaps.py

This module implements the PredictPrimeGaps class, which builds and applies a hybrid
regression-based model for predicting prime gaps. The workflow includes:
1) Generating prime and composite sequences.
2) Smoothing the raw prime-gaps signal with composite gaps.
3) Performing a domain-specific non-negative least squares regression.
4) Providing per-index gap predictions.

Core dependencies: numpy, scipy.optimize.nnls, numba for JIT-compiled generators.
"""

import math
import numpy as np
from scipy.optimize import nnls
from numba import njit, prange

# Fallback stub for NumbersDomains if module is missing
try:
    from numbers_domains import NumbersDomains
except ModuleNotFoundError:
    class NumbersDomains:
        """Stub for domain classification; always returns domain 0."""
        def evens(self, gap: int) -> int:
            return 0


class PredictPrimeGaps:
    """
    Hybrid model for predicting prime gaps up to a specified number of primes.

    Attributes
    ----------
    alpha : float
        Smoothing parameter between prime gaps and composite gaps (0 <= alpha < 1).
    M : int
        Reserved for future use (e.g., model order).
    domains : NumbersDomains
        Instance providing domain classification of gap values.
    prime_gaps : np.ndarray
        Array of raw prime gaps for the first N primes.
    comp_gaps : np.ndarray
        Array of composite gaps aligned to prime_gaps length.
    h_best : np.ndarray
        Baseline smoothed gap sequence.
    h_final : np.ndarray
        Domain-tuned smoothed gap sequence.
    w : np.ndarray
        Regression weights for each even-gap domain.
    rstd : float
        Standard deviation of residuals between actual gaps and h_final.
    H : np.ndarray
        Feature matrix including h_best and domain-specific components.
    blend : np.ndarray
        Weight vector [1.0] + w for final prediction.
    N : int
        Number of prime gaps modeled.
    """

    def __init__(self, N: int = 1000, alpha: float = 0.95, M: int = 0):
        """
        Initialize the PredictPrimeGaps model.

        Parameters
        ----------
        N : int
            Number of primes (minus one) to generate, defaults to 1000.
        alpha : float
            Mixing coefficient between prime and composite gaps; must be < 1.0.
        M : int
            Placeholder for future model configuration.

        Raises
        ------
        ValueError
            If alpha >= 1.0 to prevent data leakage.
        """
        # Validate smoothing parameter
        if alpha >= 1.0:
            raise ValueError("Alpha must be less than 1.0 to avoid data leakage.")

        self.alpha = alpha
        self.M = M
        self.domains = NumbersDomains()

        # Generate first N+1 primes and compute raw prime gaps
        primes = PredictPrimeGaps.generate_primes(N + 1)
        raw_gaps = np.diff(primes)
        self.prime_gaps = raw_gaps
        self.N = len(self.prime_gaps)

        # Generate first N+1 composites and compute composite gaps
        comps = PredictPrimeGaps.generate_composites(N + 1)
        raw_comp = np.diff(comps)
        # Align composite gaps to prime gaps length
        self.comp_gaps = raw_comp[: self.N]

        # Compute baseline smoothed signal
        self.h_best = self.baseline_smoothed_signal()
        # Perform domain-specific regression
        self.h_final, self.w = self.domain_specific_copies()
        # Compute residual standard deviation
        self.rstd = self.calculate_rstd()
        # Assemble feature matrix for prediction
        self.H = self.calculate_H()
        # Build blend weights vector
        self.blend = np.concatenate(([1.0], self.w))

    @staticmethod
    @njit
    def generate_primes(N: int) -> np.ndarray:
        """
        Generate the first N prime numbers using trial division.

        Parameters
        ----------
        N : int
            Number of primes to generate.

        Returns
        -------
        np.ndarray
            Array of shape (N,) containing the first N primes.
        """
        primes = np.empty(N, dtype=np.int64)
        count = 0
        candidate = 2
        while count < N:
            is_prime = True
            # Check divisibility by known primes
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
    def generate_composites(N: int) -> np.ndarray:
        """
        Generate the first N composite numbers (non-primes).

        Parameters
        ----------
        N : int
            Number of composites to generate.

        Returns
        -------
        np.ndarray
            Array of shape (N,) containing composite numbers.
        """
        comps = np.empty(N, dtype=np.int64)
        count = 0
        candidate = 4
        while count < N:
            # Find first divisor to classify as composite
            for i in range(2, int(math.sqrt(candidate)) + 1):
                if candidate % i == 0:
                    comps[count] = candidate
                    count += 1
                    break
            candidate += 1
        return comps

    def baseline_smoothed_signal(self) -> np.ndarray:
        """
        Smooth the prime gaps by mixing with composite gaps.

        Returns
        -------
        np.ndarray
            Array of integer gap values after rounding and min-gap enforcement.
        """
        # Weighted average of prime and composite gaps
        mixed = (self.alpha * self.prime_gaps) + ((1 - self.alpha) * self.comp_gaps)
        # Round to nearest 0.5 increment, then scale
        sm = 2 * (np.round(mixed) / 2)
        # Enforce minimum gap of 2 across signal
        sm = np.maximum(2, sm).astype(int)
        # Restore initial unity gaps where prime gap == 1
        unity_positions = (self.prime_gaps == 1)
        sm[unity_positions] = 1
        return sm

    def domain_specific_copies(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Perform non-negative least squares regression for each even-gap domain.

        Returns
        -------
        h_final : np.ndarray
            Refined gap sequence after domain-specific adjustment.
        w : np.ndarray
            Regression weights for the six even-gap domains.
        """
        # Classify each gap into one of six domains
        doms = np.array([self.domains.evens(int(g)) for g in self.prime_gaps], dtype=object)
        # Initialize feature matrix X
        X = np.zeros((self.N, 6))
        # Scale factor based on max gap > 1
        max_gap = np.max(self.prime_gaps[self.prime_gaps > 1])
        # Populate feature columns for each domain
        for j in prange(6):
            mask = (doms == j)
            X[mask, j] = (self.prime_gaps[mask] / max_gap) * self.h_best[mask]
        # Target vector: residuals of baseline
        y = self.prime_gaps - self.h_best
        # Solve non-negative least squares: X w = y
        w, _ = nnls(X, y)
        # Compute adjusted signal and enforce integer spacing
        h_fin = self.h_best + X.dot(w)
        h_fin = 2 * (np.round(h_fin) / 2)
        h_fin = np.maximum(2, h_fin).astype(int)
        # Restore unity gaps where necessary
        h_fin[self.prime_gaps == 1] = 1
        return h_fin, w

    def calculate_rstd(self) -> float:
        """
        Calculate standard deviation of residuals between actual and adjusted gaps.

        Returns
        -------
        float
            Standard deviation of (prime_gaps - h_final).
        """
        residuals = self.prime_gaps - self.h_final
        return float(np.std(residuals))

    def calculate_H(self) -> np.ndarray:
        """
        Assemble the model feature matrix H for prediction.

        Returns
        -------
        np.ndarray
            Matrix of shape (N, 1 + 6): concatenation of h_best and domain features.
        """
        doms = np.array([self.domains.evens(int(g)) for g in self.prime_gaps], dtype=object)
        X = np.zeros((self.N, 6))
        max_gap = np.max(self.prime_gaps[self.prime_gaps > 1])
        for j in prange(6):
            mask = (doms == j)
            X[mask, j] = (self.prime_gaps[mask] / max_gap) * self.h_best[mask]
        # Stack baseline and domain features
        return np.column_stack([self.h_best, X])

    def predict_gap(self, idx: int) -> int:
        """
        Predict the prime gap at a specific index.

        Parameters
        ----------
        idx : int
            Index of the desired gap (0-based).

        Returns
        -------
        int
            Predicted gap value, rounded to the nearest even integer and min 2.
        """
        # Return unity gap if original gap is 1
        if idx < self.N and self.prime_gaps[idx] == 1:
            return 1
        # Select feature vector (use last for out-of-range idx)
        vec = self.H[idx] if idx < self.N else self.H[-1]
        # Compute linear combination with blend weights
        raw_val = float(vec @ self.blend)
        # Round to nearest even integer
        gap = 2 * round(raw_val / 2)
        return max(2, int(gap))

