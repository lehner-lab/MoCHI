"""
Unit and regression test for the pymochi package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import pymochi


def test_pymochi_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "pymochi" in sys.modules
