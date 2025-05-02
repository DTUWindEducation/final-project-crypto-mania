import numpy as np
import pandas as pd
import pytest

def test_load_resp(mock_opt_file):
    """Test loading response data"""
    from src import load_resp
    
    V, phi, omega, P, T = load_resp(mock_opt_file)
    
    assert isinstance(V, np.ndarray)
    assert len(V) == 4
    assert np.allclose(V, [3.0, 5.0, 5.0, 7.0])
    assert np.allclose(phi, [0.0, 0.0, 1.0, 0.0])

def test_load_blade_data(mock_blade_file):
    """Test loading blade data"""
    from src import load_blade_data
    
    result = load_blade_data(mock_blade_file)
    assert len(result) == 10
    assert all(isinstance(arr, np.ndarray) for arr in result)
    assert np.allclose(result[0], [1.0, 2.0])  # BlSpn

def test_load_polar_data(test_data_dir):
    """Test loading polar data"""
    from src import load_polar_data
    
    polar_data = load_polar_data(test_data_dir / "Airfoils", num_files=3)
    assert len(polar_data) == 3
    assert all(isinstance(df, pd.DataFrame) for df in polar_data if df is not None)

def test_load_af_coords(test_data_dir):
    """Test loading airfoil coordinates"""
    from src import load_af_coords
    
    coords_data = load_af_coords(test_data_dir / "Airfoils", num_files=3)
    assert len(coords_data) == 3
    assert all(isinstance(df, pd.DataFrame) for df in coords_data if df is not None)
    if coords_data[0] is not None:
        assert 'x/c' in coords_data[0].columns
        assert 'y/c' in coords_data[0].columns

def test_load_polar_data_missing_file(test_data_dir, capsys):
    """Test loading polar data with missing files"""
    from src import load_polar_data
    
    # Test with non-existent directory
    polar_data = load_polar_data(test_data_dir / "NonExistentDir", num_files=3)
    captured = capsys.readouterr()
    assert all(df is None for df in polar_data)

def test_load_af_coords_missing_file(test_data_dir, capsys):
    """Test loading airfoil coordinates with missing files"""
    from src import load_af_coords
    
    # Test with non-existent directory
    coords_data = load_af_coords(test_data_dir / "NonExistentDir", num_files=3)
    captured = capsys.readouterr()
    assert all(df is None for df in coords_data)