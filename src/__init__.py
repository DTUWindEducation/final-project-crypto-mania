import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy import pi
import os
import scipy as sp
from scipy import interpolate
import scipy.interpolate as sp_interp
from typing import Tuple, Dict, List

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
            
            # Initialize
            a_prev = 0
            a_prime_prev = 0
            
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

def compute_optimal_strategy(V, phi, omega, P, T):
    """
    Computes optimal operational strategy (pitch angle and rotational speed) as function of wind speed.
    
    Args:
        V: Wind speed array [m/s]
        phi: Pitch angle array [deg]
        omega: Rotational speed array [rad/s]
        P: Power array [W]
        T: Thrust array [N]
        
    Returns:
        Tuple of (V_unique, phi_optimal, omega_optimal, P_optimal, T_optimal) where:
            V_unique: Unique wind speeds
            phi_optimal: Optimal pitch angles for each wind speed
            omega_optimal: Optimal rotational speeds for each wind speed
            P_optimal: Resulting power for each wind speed
            T_optimal: Resulting thrust for each wind speed
    """
    # Convert to numpy arrays if they're pandas Series
    V = np.asarray(V)
    phi = np.asarray(phi)
    omega = np.asarray(omega)
    P = np.asarray(P)
    T = np.asarray(T)
    
    # Get unique wind speeds
    V_unique = np.unique(V)
    
    # Initialize output arrays
    phi_optimal = np.zeros_like(V_unique)
    omega_optimal = np.zeros_like(V_unique)
    P_optimal = np.zeros_like(V_unique)
    T_optimal = np.zeros_like(V_unique)
    
    # For each unique wind speed, find the operational point with maximum power
    for i, v in enumerate(V_unique):
        mask = (V == v)
        
        if not np.any(mask):
            continue
            
        # Find index of maximum power for this wind speed
        max_power_idx = np.argmax(P[mask])
        
        # Store optimal values
        phi_optimal[i] = phi[mask][max_power_idx]
        omega_optimal[i] = omega[mask][max_power_idx]
        P_optimal[i] = P[mask][max_power_idx]
        T_optimal[i] = T[mask][max_power_idx]
    
    return V_unique, phi_optimal, omega_optimal, P_optimal, T_optimal

def get_airfoil_coefficients(alpha, r_position, BlSpn, BlAFID, polar_data):
    """
    Gets interpolated airfoil coefficients (Cl, Cd) at specific span position and angle of attack.
    
    Args:
        alpha: Angle of attack [deg] (can be array)
        r_position: Span position [m]
        BlSpn: Blade span coordinates [m]
        BlAFID: Airfoil IDs
        polar_data: Airfoil polar data
        
    Returns:
        Cl: Lift coefficient at given position and angle(s)
        Cd: Drag coefficient at given position and angle(s)
    """
    # Find nearest span position
    idx = np.argmin(np.abs(BlSpn - r_position))
    af_id = int(BlAFID[idx]) - 1  # Convert to 0-based index
    
    if af_id < 0 or af_id >= len(polar_data) or polar_data[af_id] is None:
        return np.nan, np.nan
    
    # Create interpolators
    polar_df = polar_data[af_id]
    Cl_interp = sp.interpolate.interp1d(
        polar_df['Alpha'], polar_df['Cl'], 
        kind='linear', bounds_error=False, fill_value=(polar_df['Cl'].iloc[0], polar_df['Cl'].iloc[-1])
    )
    Cd_interp = sp.interpolate.interp1d(
        polar_df['Alpha'], polar_df['Cd'], 
        kind='linear', bounds_error=False, fill_value=(polar_df['Cd'].iloc[0], polar_df['Cd'].iloc[-1])
    )
    
    return Cl_interp(alpha), Cd_interp(alpha)



def analyze_airfoil_performance(BlSpn, BlAFID, polar_data):
    """Analyze airfoil performance at key blade sections"""
    print("\n" + "="*80)
    print("AIRFOIL PERFORMANCE ANALYSIS AT KEY BLADE SECTIONS")
    print("="*80)
    
    # Select 3 representative positions (root, mid, tip)
    positions = [
        {"name": "Root", "r": BlSpn[10]},  # Skip first few root elements
        {"name": "Mid-span", "r": BlSpn[len(BlSpn)//2]},
        {"name": "Tip", "r": BlSpn[-5]}  # Skip very tip
    ]
    
    # Create angle of attack range
    alpha_test = np.linspace(-5, 15, 21)  # -5° to 15° in 1° steps
    
    # Create figure for plots
    plt.figure(figsize=(15, 6))
    
    for i, pos in enumerate(positions):
        # Get coefficients
        Cl, Cd = get_airfoil_coefficients(alpha_test, pos["r"], BlSpn, BlAFID, polar_data)
        
        # Plot Cl and Cd
        plt.subplot(1, 2, 1)
        plt.plot(alpha_test, Cl, 'o-', label=f'{pos["name"]} (r={pos["r"]:.1f}m)')
        
        plt.subplot(1, 2, 2)
        plt.plot(alpha_test, Cd, 'o-', label=f'{pos["name"]} (r={pos["r"]:.1f}m)')
    
    # Format Cl plot
    plt.subplot(1, 2, 1)
    plt.title('Lift Coefficient vs Angle of Attack')
    plt.xlabel('Angle of Attack [deg]')
    plt.ylabel('Lift Coefficient (Cl)')
    plt.grid(True)
    plt.legend()
    
    # Format Cd plot
    plt.subplot(1, 2, 2)
    plt.title('Drag Coefficient vs Angle of Attack')
    plt.xlabel('Angle of Attack [deg]')
    plt.ylabel('Drag Coefficient (Cd)')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed table
    print("\nDetailed Airfoil Coefficients:")
    print("-"*85)
    print(f"{'Position':<12}{'r [m]':<8}{'α [°]':<6}{'Cl':<8}{'Cd':<8}{'Cl/Cd':<10}")
    print("-"*85)
    
    stall_threshold = 0.1  # Cd increase threshold for stall detection
    for pos in positions:
        alphas = [-5, 0, 5, 10, 15]  # Key angles to analyze
        prev_cd = None
        
        for alpha in alphas:
            Cl, Cd = get_airfoil_coefficients(alpha, pos["r"], BlSpn, BlAFID, polar_data)
            cl_cd = Cl/Cd if Cd > 0 else float('inf')
            
            # Detect stall (sudden increase in Cd)
            stall_status = ""
            if prev_cd is not None and (Cd - prev_cd) > stall_threshold:
                stall_status = "STALL"
            prev_cd = Cd
            
            print(f"{pos['name']:<12}{pos['r']:<8.1f}{alpha:<6.1f}{Cl:<8.3f}{Cd:<8.4f}{cl_cd:<10.1f}{stall_status}")


def identify_operational_mode(V0, phi_opt, omega_opt, omega_rated=None):
    """
    Identifies turbine operational mode based on wind speed and optimal parameters.
    
    Args:
        V0: Wind speed [m/s] (can be single value or array)
        phi_opt: Optimal pitch angle [deg] (same length as V0)
        omega_opt: Optimal rotational speed [rad/s] (same length as V0)
        omega_rated: Optional rated rotational speed
        
    Returns:
        If single input: tuple of (mode_string, info_dict)
        If array input: tuple of (mode_list, info_list)
    """
    # Convert inputs to numpy arrays
    V0 = np.asarray(V0)
    phi_opt = np.asarray(phi_opt)
    omega_opt = np.asarray(omega_opt)
    
    # Determine rated speed
    omega_max = np.max(omega_opt) if omega_rated is None else omega_rated
    
    # Initialize outputs
    modes = []
    infos = []
    
    for v, phi, omega in zip(V0, phi_opt, omega_opt):
        if v < 4:
            mode = "Start-up"
            info = {
                "description": "Below cut-in speed", 
                "color": "red",
                "pitch": phi,
                "speed": omega
            }
        elif (phi < 1) and (omega < 0.9*omega_max):
            mode = "Partial load"
            info = {
                "description": "Max power tracking",
                "control": "Variable speed, fixed pitch",
                "color": "green",
                "pitch": phi,
                "speed": omega
            }
        elif (phi < 5) and (omega >= 0.9*omega_max):
            mode = "Transition"
            info = {
                "description": "Approaching rated power",
                "control": "Pitch begins to vary",
                "color": "orange",
                "pitch": phi,
                "speed": omega
            }
        else:
            mode = "Full load"
            info = {
                "description": "Power regulation",
                "control": "Fixed speed, variable pitch",
                "color": "blue",
                "pitch": phi,
                "speed": omega
            }
        
        modes.append(mode)
        infos.append(info)
    
    # Return appropriate format based on input
    if V0.ndim == 0:  # Single value input
        return modes[0], infos[0]
    return modes, infos