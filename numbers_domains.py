class NumbersDomains:
    def __init__(self):
        self.prime_cache = {}

    def unity(self, g):
        """
        Unity domain: the gap = 1
        """
        if g == 1:
            return 1  # Unity Domain
        return None

    def evens(self, g):
        """
        Even domains, partitioned by 2-adic order up to 2^4:
          2 → g=2·m, m odd                     → code 3
          3 → g=4·m, m odd                     → code 4
          4 → g=8·m, m odd                     → code 5
          5 → g=16·m, m odd                    → code 6
          6 → g=2^k, k≥2 (i.e. 4,8,16,...)      → code 2
          7 → all other evens                  → code 7
        """
        # never classify unity here
        if g == 1:
            return None

        # highest dyadic power ≥4
        if g >= 4 and (g & (g - 1)) == 0:
            return 2

        if g % 2 == 0:
            # exactly one factor of 2 → 2·odd
            if (g // 2) % 2 == 1:
                return 3
            # exactly two factors of 2 → 4·odd
            if g % 4 == 0 and (g // 4) % 2 == 1:
                return 4
            # exactly three factors of 2 → 8·odd
            if g % 8 == 0 and (g // 8) % 2 == 1:
                return 5
            # exactly four factors of 2 → 16·odd
            if g % 16 == 0 and (g // 16) % 2 == 1:
                return 6
            # residual evens
            return 7

        return None  # not an even gap

    def odds(self, g):
        """
        Odd domains:
          8  → prime odds
          9  → odd perfect powers b^k (k≥3)
          10 → product of two distinct primes
          11 → q·p for q∈{7,11,13,17,19}, p prime
          12 → other composite odds
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
        return None  # not an odd gap

    def _is_prime(self, n):
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

    def _is_odd_power(self, n):
        b = 3
        while b ** 3 <= n:
            k = 3
            while b ** k <= n:
                if b ** k == n:
                    return True
                k += 1
            b += 2
        return False

    def _is_product_of_two_primes(self, n):
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                j = n // i
                if i != j and self._is_prime(i) and self._is_prime(j):
                    return True
        return False
