import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.your_module import core_function


def test_core_function_returns_expected():
    assert core_function(0) == 42
