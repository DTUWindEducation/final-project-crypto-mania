import pytest
import numpy as np
import pandas as pd
import os
from pathlib import Path

@pytest.fixture
def test_data_dir():
    """Fixture for test data directory"""
    return Path(__file__).parent.parent / "inputs" / "IEA-15-240-RWT"

@pytest.fixture
def mock_opt_file(tmp_path):
    """Create a valid mock .opt file"""
    content = """Header to skip
3.0 0.0 1.0 100 50
5.0 0.0 2.0 500 200
5.0 1.0 2.5 450 180
7.0 0.0 3.0 800 300"""
    file = tmp_path / "test.opt"
    file.write_text(content)
    return file

@pytest.fixture
def mock_blade_file(tmp_path):
    """Create a valid mock blade file"""
    content = """Header line 1\nHeader line 2\nHeader line 3\nHeader line 4\nHeader line 5\nHeader line 6
1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0
2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0"""
    file = tmp_path / "blade.dat"
    file.write_text(content)
    return file

@pytest.fixture
def mock_polar_data():
    """Create mock polar data"""
    return [
        pd.DataFrame({
            'Alpha': [-5, 0, 5, 10],
            'Cl': [0.2, 0.5, 1.0, 1.2],
            'Cd': [0.01, 0.01, 0.02, 0.1]
        }) for _ in range(3)
    ]

@pytest.fixture
def mock_blade_geometry():
    """Create mock blade geometry with enough data points"""
    return {
        'BlSpn': np.linspace(0, 70, 20),  # 20 points from 0m to 70m
        'BlAFID': np.array([1] * 20),      # All use airfoil 1
        'BlTwist': np.linspace(15, 1, 20),
        'BlChord': np.linspace(4, 1, 20)
    }