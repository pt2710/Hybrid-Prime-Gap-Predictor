import pytest
from src.hybrid_prime_finder import next_prime_hybrid

def test_next_prime_hybrid_simple():
    assert next_prime_hybrid(2) == 3
    assert next_prime_hybrid(3) == 5
