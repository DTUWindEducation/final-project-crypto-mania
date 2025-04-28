import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy import pi
import os
import scipy as sp
from scipy import interpolate

def load_resp(path_resp):
    """Loads response data from an .opt file and applies time filtering."""
    # Read the file, skipping the first row of column names and use sep='\s+' for whitespace separation
    data = pd.read_csv(path_resp, sep='\s+', header=None, skiprows=1)  # Skip the first row with non-relevant info
    
    # Rename columns to a more convenient format
    data.columns = ['wind_speed', 'pitch', 'rot_speed', 'aero_power', 'aero_thrust']
    
    # Extract relevant data
    V = data['wind_speed'].values
    phi = data['pitch'].values
    omega = data['rot_speed'].values
    P = data['aero_power'].values
    T = data['aero_thrust'].values
    
    return V, phi, omega, P, T

def load_blade_data(file_path):
    # Read the file, skipping the first few lines if necessary, and use whitespace as separator
    data = pd.read_csv(file_path, sep='\s+', header=None, skiprows=6)

    # Rename columns to a more convenient format
    data.columns = ['BlSpn', 'BlCrvAC', 'BlSwpAC', 'BlCrvAng', 'BlTwist', 'BlChord', 'BlAFID', 'BlCb', 'BlCenBn', 'BlCenBt']
    
    # Extract relevant data
    BlSpn = data['BlSpn'].values
    BlCrvAC = data['BlCrvAC'].values
    BlSwpAC = data['BlSwpAC'].values
    BlCrvAng = data['BlCrvAng'].values
    BlTwist = data['BlTwist'].values
    BlChord = data['BlChord'].values
    BlAFID = data['BlAFID'].values
    BlCb = data['BlCb'].values
    BlCenBn = data['BlCenBn'].values
    BlCenBt = data['BlCenBt'].values
    
    return BlSpn, BlCrvAC, BlSwpAC, BlCrvAng, BlTwist, BlChord, BlAFID, BlCb, BlCenBn, BlCenBt

def load_polar_data(directory, base_filename="IEA-15-240-RWT_AeroDyn15_Polar_", num_files=50):
    polar_data = []
    column_names = ['Alpha', 'Cl', 'Cd', 'Cm']

    for i in range(num_files):
        filename = f"{base_filename}{i:02d}.dat"
        filepath = os.path.join(directory, filename)

        try:
            # Determine how many lines to skip (header row included in both cases)
            skip = 20 if i < 5 else 54

            # Read file without assuming any header row; manually assign names
            df = pd.read_csv(filepath, sep='\s+', skiprows=skip, header=None, names=column_names, engine='python')
            
            # Append to list
            polar_data.append(df)

        except Exception as e:
            print(f"Error reading file {filename}: {e}")
            polar_data.append(None)

    return polar_data

def load_af_coords(directory, base_filename="IEA-15-240-RWT_AF", num_files=50):
    coords_data = []

    for i in range(num_files):
        filename = f"{base_filename}{i:02d}_Coords.txt"
        filepath = os.path.join(directory, filename)

        try:
            # Skip the first 9 lines (headers and comments)
            df = pd.read_csv(filepath, sep='\s+', skiprows=8, header=None, names=["x/c", "y/c"], engine='python')
            coords_data.append(df)

        except Exception as e:
            print(f"Error reading file {filename}: {e}")
            coords_data.append(None)

    return coords_data

def plot_V_vs_phi(V, phi):
    """Plots wind speed (V) against pitch (phi)."""
    plt.figure(figsize=(10, 6))
    plt.plot(phi, V, label="Wind Speed vs Pitch", color='b')
    plt.xlabel("Pitch (phi) [deg]")
    plt.ylabel("Wind Speed (V) [m/s]")
    plt.title("Wind Speed vs Pitch")
    plt.grid(True)
    plt.legend()
    plt.show()


def compute_cl_cd_vs_r_alpha(BlSpn, BlAFID, polar_data, alpha_range=np.linspace(-20, 20, 100)):
    r_values = BlSpn
    alpha_values = alpha_range
    Cl_matrix = np.zeros((len(r_values), len(alpha_values)))
    Cd_matrix = np.zeros((len(r_values), len(alpha_values)))
    
    for i, (r, af_id) in enumerate(zip(r_values, BlAFID)):
        af_id = int(af_id) - 1  # Convert to 0-based index
        
        # Skip invalid airfoil IDs or missing data
        if af_id < 0 or af_id >= len(polar_data) or polar_data[af_id] is None:
            Cl_matrix[i, :] = np.nan
            Cd_matrix[i, :] = np.nan
            continue
        
        polar_df = polar_data[af_id]
        
        # Interpolate Cl and Cd
        Cl_interp = sp.interpolate.interp1d(
            polar_df['Alpha'], polar_df['Cl'], 
            kind='linear', fill_value="extrapolate"
        )
        Cd_interp = sp.interpolate.interp1d(
            polar_df['Alpha'], polar_df['Cd'], 
            kind='linear', fill_value="extrapolate"
        )
        
        Cl_matrix[i, :] = Cl_interp(alpha_range)
        Cd_matrix[i, :] = Cd_interp(alpha_range)
    
    return r_values, alpha_values, Cl_matrix, Cd_matrix

def plot_airfoils(coords_data):
    """Plots the airfoil shapes in one figure."""
    plt.figure(figsize=(8, 4))
    
    for i, df in enumerate(coords_data):
        if df is not None:
            plt.plot(df['x/c'], df['y/c'], label=f'Airfoil {i}')
    
    plt.xlabel('x/c')
    plt.ylabel('y/c')
    plt.title('Airfoil Shapes')
    plt.grid(True)
    plt.show()


def compute_induction_factors(r, V0, theta_p, omega, BlSpn, BlTwist, BlChord, BlAFID, polar_data,
                            B=3, rho=1.225, tol=1e-5, max_iter=200, relaxation=0.05):
    """
    Robust BEM solver for axial (a) and tangential (a') induction factors.
    
    Args:
        r: Span positions [m]
        V0: Wind speed [m/s]
        theta_p: Pitch angle [deg]
        omega: Rotational speed [rad/s]
        BlSpn: Blade span coordinates [m]
        BlTwist: Blade twist [deg]
        BlChord: Blade chord [m]
        BlAFID: Airfoil IDs
        polar_data: Airfoil polar data
        B: Number of blades (default 3)
        rho: Air density (default 1.225)
        tol: Convergence tolerance (default 1e-5)
        max_iter: Max iterations (default 200)
        relaxation: Relaxation factor (default 0.05)
    
    Returns:
        a: Axial induction factors
        a_prime: Tangential induction factors
    """
    a = np.zeros_like(r)
    a_prime = np.zeros_like(r)
    
    # Create interpolators with bounds checking
    def safe_interp1d(x, y, kind='linear'):
        return sp.interpolate.interp1d(x, y, kind=kind, bounds_error=False, fill_value=(y[0], y[-1]))
    
    twist_interp = safe_interp1d(BlSpn, BlTwist)
    chord_interp = safe_interp1d(BlSpn, BlChord)
    afid_interp = sp.interpolate.interp1d(BlSpn, BlAFID, kind='nearest', bounds_error=False, fill_value=BlAFID[0])
    
    for i, ri in enumerate(r):
        if ri < 0.5:  # Skip root region
            a[i] = 0
            a_prime[i] = 0
            continue
            
        try:
            # Get local properties
            twist = twist_interp(ri)
            chord = chord_interp(ri)
            af_id = int(afid_interp(ri)) - 1
            
            if af_id < 0 or af_id >= len(polar_data) or polar_data[af_id] is None:
                raise ValueError(f"Invalid airfoil ID {af_id+1}")
                
            sigma = (B * chord) / (2 * np.pi * ri)
            
            # Initialize with better estimates
            a_prev = 0.2
            a_prime_prev = 0.01
            
            for iter in range(max_iter):
                # 1. Flow angle calculation
                V_axial = V0 * (1 - a_prev)
                V_tangential = omega * ri * (1 + a_prime_prev)
                
                if abs(V_tangential) < 1e-6:
                    phi = np.pi/2
                else:
                    phi = np.arctan2(V_axial, V_tangential)
                
                # 2. Angle of attack with limits
                alpha = np.rad2deg(phi) - (theta_p + twist)
                alpha = np.clip(alpha, -10, 10)  # Conservative limits
                
                # 3. Get airfoil coefficients with fallback
                polar_df = polar_data[af_id]
                try:
                    Cl = np.interp(alpha, polar_df['Alpha'], polar_df['Cl'], left=0, right=0)
                    Cd = np.interp(alpha, polar_df['Alpha'], polar_df['Cd'], left=1.0, right=1.0)
                except:
                    Cl = 0.5
                    Cd = 0.01
                
                # 4. Normal and tangential coefficients
                Cn = Cl * np.cos(phi) + Cd * np.sin(phi)
                Ct = Cl * np.sin(phi) - Cd * np.cos(phi)
                
                # 5. Tip loss factor with safeguards
                sin_phi = max(np.sin(phi), 0.01)
                tip_loss_arg = -B/2 * (BlSpn[-1] - ri)/(ri * sin_phi)
                tip_loss_arg = np.clip(tip_loss_arg, -100, 100)
                F = (2/np.pi) * np.arccos(np.exp(tip_loss_arg))
                F = max(F, 0.01)
                
                # 6. Update axial induction
                denominator_a = 4 * F * sin_phi**2 / (sigma * Cn)
                if denominator_a <= -0.99:
                    a_new = 0.5
                else:
                    a_new = 1/(denominator_a + 1)
                
                # Glauert correction
                if a_new > 0.4:
                    K = 4 * F * sin_phi**2 / (sigma * Cn)
                    sqrt_term = (K*(1-2*0.4)+2)**2 + 4*(K*0.4**2-1)
                    if sqrt_term >= 0:
                        a_new = 0.5*(2 + K*(1-2*0.4) - np.sqrt(sqrt_term))
                    else:
                        a_new = 0.4
                
                # 7. Update tangential induction
                denominator_a_prime = 4 * F * sin_phi * np.cos(phi) / (sigma * Ct)
                if abs(denominator_a_prime - 1) > 1e-6:
                    a_prime_new = 1/(denominator_a_prime - 1)
                else:
                    a_prime_new = 0
                
                # Apply relaxation
                a_new = relaxation * a_new + (1-relaxation) * a_prev
                a_prime_new = relaxation * a_prime_new + (1-relaxation) * a_prime_prev
                
                # Check convergence
                if (abs(a_new - a_prev) < tol and 
                    abs(a_prime_new - a_prime_prev) < tol):
                    break
                    
                a_prev = a_new
                a_prime_prev = a_prime_new
            else:
                print(f"Warning: Max iterations reached at r={ri:.2f}m")
            
            # Store final values with physical limits
            a[i] = np.clip(a_new, 0, 0.99)
            a_prime[i] = np.clip(a_prime_new, -0.5, 0.5)
            
        except Exception as e:
            print(f"Error at r={ri:.2f}m: {str(e)}")
            a[i] = np.nan
            a_prime[i] = np.nan
    
    return a, a_prime


def compute_dT(r, dr, rho, V_inflow, axial_factor):
    return abs(4 * np.pi * r * rho * V_inflow**2 * axial_factor * (1 - axial_factor) * dr)

def compute_dM(r, dr, rho, V_inflow, axial_factor, tangential_factor, omega):
    return 4 * np.pi * r**3 * rho * V_inflow * omega * tangential_factor * (1 - axial_factor) * dr

def compute_aerodynamic_power(torque, rotational_speed):
    return torque * rotational_speed