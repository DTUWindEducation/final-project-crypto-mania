import numpy as np
import pytest

def test_compute_optimal_strategy():
    """Test optimal strategy computation"""
    from src import compute_optimal_strategy
    
    V = np.array([3, 5, 5, 7, 7])
    phi = np.array([0, 0, 1, 0, 2])
    omega = np.array([1, 2, 2.5, 3, 3.5])
    P = np.array([100, 500, 450, 800, 950])
    T = np.array([50, 200, 180, 300, 350])
    
    V_opt, phi_opt, omega_opt, P_opt, T_opt = compute_optimal_strategy(V, phi, omega, P, T)
    
    assert len(V_opt) == 3

def test_identify_operational_mode():
    """Test operational mode identification"""
    from src import identify_operational_mode
    
    # Test array input
    modes, infos = identify_operational_mode(
        v0=np.array([3.0, 5.0, 12.0]),
        phi_opt=np.array([0.0, 0.5, 10.0]),
        omega_opt=np.array([1.0, 6.0, 9.0])
    )
    assert modes == ["Start-up", "Partial load", "Full load"]

