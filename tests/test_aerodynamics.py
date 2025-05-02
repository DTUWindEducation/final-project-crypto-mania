import numpy as np
import pytest

def test_compute_induction_factors(mock_blade_geometry, mock_polar_data):
    """Test induction factor calculation"""
    from src import compute_induction_factors
    
    r = mock_blade_geometry['BlSpn']
    a, a_prime = compute_induction_factors(
        r=r,
        V0=8.0,
        theta_p=0.0,
        omega=7.5,
        **mock_blade_geometry,
        polar_data=mock_polar_data
    )
    
    assert isinstance(a, np.ndarray)
    assert len(a) == len(r)
    assert np.all(a >= 0) and np.all(a <= 1)
    assert np.all(a_prime >= -0.5) and np.all(a_prime <= 0.5)

def test_get_airfoil_coefficients(mock_blade_geometry, mock_polar_data):
    """Test airfoil coefficient calculation"""
    from src import get_airfoil_coefficients
    
    Cl, Cd = get_airfoil_coefficients(
        5.0,  # alpha
        30.0,  # r_position
        mock_blade_geometry['BlSpn'],
        mock_blade_geometry['BlAFID'],
        mock_polar_data
    )
    
    assert 0.5 <= Cl <= 1.5
    assert 0.01 <= Cd <= 0.1

def test_compute_cl_cd_vs_r_alpha(mock_blade_geometry, mock_polar_data):
    """Test computation of Cl/Cd matrices"""
    from src import compute_cl_cd_vs_r_alpha
    
    r, alpha, Cl_matrix, Cd_matrix = compute_cl_cd_vs_r_alpha(
        mock_blade_geometry['BlSpn'],
        mock_blade_geometry['BlAFID'],
        mock_polar_data,
        alpha_range=np.linspace(-5, 15, 5)
    )
    
    assert len(r) == len(mock_blade_geometry['BlSpn'])
    assert len(alpha) == 5
    assert Cl_matrix.shape == (len(r), len(alpha))
    assert Cd_matrix.shape == (len(r), len(alpha))
    assert not np.isnan(Cl_matrix).all()
    assert not np.isnan(Cd_matrix).all()

def test_compute_dT():
    """Test thrust element computation"""
    from src import compute_dT
    
    r = 50.0
    dr = 1.0
    rho = 1.225
    V_inflow = 10.0
    axial_factor = 0.3
    
    dT = compute_dT(r, dr, rho, V_inflow, axial_factor)
    
    assert isinstance(dT, float)
    assert dT > 0

def test_compute_dM():
    """Test torque element computation"""
    from src import compute_dM
    
    r = 50.0
    dr = 1.0
    rho = 1.225
    V_inflow = 10.0
    axial_factor = 0.3
    tangential_factor = 0.1
    omega = 1.5
    
    dM = compute_dM(r, dr, rho, V_inflow, axial_factor, tangential_factor, omega)
    
    assert isinstance(dM, float)
    assert dM > 0

def test_compute_aerodynamic_power():
    """Test power computation"""
    from src import compute_aerodynamic_power
    
    torque = 1e6  # 1 MNm
    rotational_speed = 1.5  # rad/s
    
    power = compute_aerodynamic_power(torque, rotational_speed)
    
    assert isinstance(power, float)
    assert np.isclose(power, 1.5e6)

