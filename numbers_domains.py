"""
numbers_domains.py

Provides domain classification for gap values in prime-gap prediction.
Domains are divided into:
- Unity domain (gap == 1)
- Even domains based on 2-adic structure
- Odd domains based on primality and composite structure, including perfect powers
"""

class NumbersDomains:
    """
    Class for classifying integer gap values into conceptual domains.

    Methods
    -------
    unity(g)
        Classify gap == 1 as unity domain.
    evens(g)
        Classify even gaps into specific 2-adic domains.
    odds(g)
        Classify odd gaps into prime, power, or composite domains.
    """

    def __init__(self):
        """
        Initialize the NumbersDomains instance.

        Attributes
        ----------
        prime_cache : dict
            Cache for primality checks to avoid repeated computation.
        """
        self.prime_cache = {}

    def unity(self, g: int) -> int | None:
        """
        Classify the unity domain (gap == 1).

        Parameters
        ----------
        g : int
            Gap value to classify.

        Returns
        -------
        int or None
            Returns 1 for the unity domain, or None if g != 1.
        """
        if g == 1:
            return 1
        return None

    def evens(self, g: int) -> int | None:
        """
        Classify even gap values into one of six domains based on 2-adic order.

        Domains mapping:
          power of two >=4 -> code 2
          g = 2 * odd         -> code 3
          g = 4 * odd         -> code 4
          g = 8 * odd         -> code 5
          g = 16 * odd        -> code 6
          other evens         -> code 7

        Parameters
        ----------
        g : int
            Gap value to classify.

        Returns
        -------
        int or None
            Domain code (2-7) for even gaps, or None otherwise.
        """

        if g == 1:
            return None
        
        if g >= 4 and (g & (g - 1)) == 0:
            return 2
        
        if g % 2 == 0:
        
            if (g // 2) % 2 == 1:
                return 3
            if (g // 4) % 2 == 1:
                return 4
            if (g // 8) % 2 == 1:
                return 5
            if (g // 16) % 2 == 1:
                return 6
            
            return 7
        return None

    def odds(self, g: int) -> int | None:
        """
        Classify odd gap values into five domains.

        Domains mapping:
          prime odds -> code 8
          odd perfect powers b^k (k>=2) -> code 9
          product of two distinct primes -> code 10
          product q*p where q in {7,11,13,17,19}, p prime -> code 11
          all other composite odds -> code 12

        Parameters
        ----------
        g : int
            Gap value to classify.

        Returns
        -------
        int or None
            Domain code (8-12) for odd gaps, or None otherwise.
        """
        if g % 2 == 1:
            if self._is_prime(g):
                return 8
            if self._is_odd_power(g):
                return 9
            if self._is_product_of_two_primes(g):
                return 10
            for q in (7, 11, 13, 17, 19):
                if g % q == 0 and self._is_prime(g // q):
                    return 11
            return 12
        return None

    def _is_prime(self, n: int) -> bool:
        """
        Check if n is prime using cached trial division.

        Parameters
        ----------
        n : int
            Integer to test.

        Returns
        -------
        bool
            True if n is prime, False otherwise.
        """
        if n in self.prime_cache:
            return self.prime_cache[n]
        if n <= 1:
            result = False
        elif n <= 3:
            result = True
        elif n % 2 == 0 or n % 3 == 0:
            result = False
        else:
            result = True
            i = 5
            while i * i <= n:
                if n % i == 0 or n % (i + 2) == 0:
                    result = False
                    break
                i += 6
        self.prime_cache[n] = result
        return result

    def _is_odd_power(self, n: int) -> bool:
        """
        Detect if n is an odd perfect power b^k for k >= 2.

        Parameters
        ----------
        n : int
            Integer to test.

        Returns
        -------
        bool
            True if n == b^k for some integer b > 1 and k >= 2, False otherwise.
        """
        # Start base at 2 (allow squares)
        b = 2
        # Continue while smallest square <= n
        while b * b <= n:
            k = 2
            # Check powers of b up to exceeding n
            while b ** k <= n:
                if b ** k == n:
                    return True
                k += 1
            b += 1
        return False

    def _is_product_of_two_primes(self, n: int) -> bool:
        """
        Check if n is the product of two distinct prime numbers.

        Parameters
        ----------
        n : int
            Integer to test.

        Returns
        -------
        bool
            True if n = p1 * p2 with p1 != p2 both primes, else False.
        """
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                j = n // i
                if i != j and self._is_prime(i) and self._is_prime(j):
                    return True
        return False
    