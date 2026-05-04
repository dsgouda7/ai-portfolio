"""
Test configuration and fixtures
"""

import pytest
import os
import sys

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


@pytest.fixture(scope='session')
def test_data_dir():
    """Create test data directory."""
    test_dir = './data/test_data'
    os.makedirs(test_dir, exist_ok=True)
    yield test_dir


@pytest.fixture(autouse=True)
def cleanup_test_sessions():
    """Clean up test sessions after each test."""
    yield
    # Cleanup code here if needed
