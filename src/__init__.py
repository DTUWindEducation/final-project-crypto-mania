"""
This module provides tools for analyzing wind turbine aerodynamic performance.

It includes:
- Functions for calculating thrust, torque, and aerodynamic power.
- Utilities for interpolating airfoil coefficients based on spanwise position and angle of attack.
- Visualization tools for airfoil performance.
- Functions to determine optimal operational strategy and identify turbine operating modes.

Dependencies:
- numpy
- pandas
- matplotlib
- scipy
"""
import os
from typing import Union, Tuple, List, Dict, Optional
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from numpy import pi
import scipy as sp
import scipy.interpolate as sp_interp

def load_resp(path_resp: str):
    """Load response data from an .opt file and apply time filtering.

    Args:
        path_resp (str): Path to the .opt file.

    Returns:
        tuple: Tuple containing arrays of wind speed, pitch, rotational speed, 
               aerodynamic power, and aerodynamic thrust.
    """
    # Read the file, skipping the first row of column names and using whitespace separator
    data = pd.read_csv(path_resp, sep=r'\s+', header=None, skiprows=1)

    # Rename columns to a more convenient format
    data.columns = ['wind_speed', 'pitch', 'rot_speed', 'aero_power', 'aero_thrust']

    # Extract relevant data
    v = data['wind_speed'].values
    phi = data['pitch'].values
    omega = data['rot_speed'].values
    pwr = data['aero_power'].values
    thrust = data['aero_thrust'].values

    return v, phi, omega, pwr, thrust

def load_blade_data(file_path: str) -> Tuple[np.ndarray, ...]:
    """Load blade geometry data from a file.

    Args:
        file_path (str): Path to the blade data file.

    Returns:
        tuple: Tuple of numpy arrays containing blade spanwise coordinates, 
               curve axis control, sweep axis control, curvature angle, twist,
               chord, airfoil ID, center of mass (chordwise), and bending centers.
    """
    data = pd.read_csv(file_path, sep=r'\s+', header=None, skiprows=6)
    data.columns = [
        'BlSpn', 'BlCrvAC', 'BlSwpAC', 'BlCrvAng', 'BlTwist',
        'BlChord', 'BlAFID', 'BlCb', 'BlCenBn', 'BlCenBt'
    ]

    bl_spn = data['BlSpn'].values
    bl_crv_ac = data['BlCrvAC'].values
    bl_swp_ac = data['BlSwpAC'].values
    bl_crv_ang = data['BlCrvAng'].values
    bl_twist = data['BlTwist'].values
    bl_chord = data['BlChord'].values
    bl_afid = data['BlAFID'].values
    bl_cb = data['BlCb'].values
    bl_cen_bn = data['BlCenBn'].values
    bl_cen_bt = data['BlCenBt'].values

    return (
        bl_spn, bl_crv_ac, bl_swp_ac, bl_crv_ang,
        bl_twist, bl_chord, bl_afid, bl_cb, bl_cen_bn, bl_cen_bt
    )

def load_polar_data(
    directory: str,
    base_filename: str = "IEA-15-240-RWT_AeroDyn15_Polar_",
    num_files: int = 50
) -> List[Optional[pd.DataFrame]]:
    """Load a list of polar data files into pandas DataFrames.

    Args:
        directory (str): Path to the directory containing the polar files.
        base_filename (str, optional): Base name of the polar files.
        Defaults to 'IEA-15-240-RWT_AeroDyn15_Polar_'.
        num_files (int, optional): Number of polar files to load. Defaults to 50.

    Returns:
        List[Optional[pd.DataFrame]]: List of DataFrames (or None if loading fails for a file).
    """
    polar_data = []
    column_names = ['Alpha', 'Cl', 'Cd', 'Cm']

    for i in range(num_files):
        filename = f"{base_filename}{i:02d}.dat"
        filepath = os.path.join(directory, filename)

        try:
            skip = 20 if i < 5 else 54
            df = pd.read_csv(
                filepath,
                sep=r'\s+',
                skiprows=skip,
                header=None,
                names=column_names,
                engine='python'
            )
            polar_data.append(df)
        except (OSError, pd.errors.ParserError) as err:
            print(f"Error reading file {filename}: {err}")
            polar_data.append(None)

    return polar_data

def load_af_coords(
    directory: str,
    base_filename: str = "IEA-15-240-RWT_AF",
    num_files: int = 50
) -> List[Optional[pd.DataFrame]]:
    """Load airfoil coordinate data from text files.

    Args:
        directory (str): Path to the directory containing coordinate files.
        base_filename (str, optional): Base name of the airfoil files.
        Defaults to 'IEA-15-240-RWT_AF'.
        num_files (int, optional): Number of files to load. Defaults to 50.

    Returns:
        List[Optional[pd.DataFrame]]: List of DataFrames with
        airfoil coordinates or None for failed reads.
    """
    coords_data = []

    for i in range(num_files):
        filename = f"{base_filename}{i:02d}_Coords.txt"
        filepath = os.path.join(directory, filename)

        try:
            df = pd.read_csv(
                filepath,
                sep=r'\s+',
                skiprows=8,
                header=None,
                names=["x/c", "y/c"],
                engine='python'
            )
            coords_data.append(df)
        except (OSError, pd.errors.ParserError) as err:
            print(f"Error reading file {filename}: {err}")
            coords_data.append(None)

    return coords_data

def plot_V_vs_phi(V: np.ndarray, phi: np.ndarray, show: bool = True) -> Figure:
    """Plot wind speed (V) against pitch angle (phi).

    Args:
        V (np.ndarray): Array of wind speeds [m/s].
        phi (np.ndarray): Array of pitch angles [deg].
        show (bool, optional): Whether to display the plot. Defaults to True.

    Returns:
        Figure: The created matplotlib figure object.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(phi, V, label="Wind Speed vs Pitch", color='b')
    ax.set_xlabel("Pitch (phi) [deg]")
    ax.set_ylabel("Wind Speed (V) [m/s]")
    ax.set_title("Wind Speed vs Pitch")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()

    if show:
        plt.show()

    return fig


def compute_cl_cd_vs_r_alpha(
    blspn: np.ndarray,
    blafid: np.ndarray,
    polar_data: List[Optional[pd.DataFrame]],
    alpha_range: np.ndarray = np.linspace(-20, 20, 100)
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute Cl and Cd values as a function of blade span and angle of attack.

    Args:
        BlSpn (np.ndarray): Blade spanwise positions.
        BlAFID (np.ndarray): Airfoil IDs at each blade section.
        polar_data (List[Optional[pd.DataFrame]]): List of polar data DataFrames (or None).
        alpha_range (np.ndarray, optional): Array of angle 
        of attack values [deg]. Defaults to linspace(-20, 20, 100).

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
            r_values, alpha_values, Cl_matrix, Cd_matrix
    """
    r_values = blspn
    alpha_values = alpha_range
    cl_matrix = np.zeros((len(r_values), len(alpha_values)))
    cd_matrix = np.zeros((len(r_values), len(alpha_values)))
    for i, (r, af_id) in enumerate(zip(r_values, blafid)):
        af_id = int(af_id) - 1  # Convert to 0-based index

        if af_id < 0 or af_id >= len(polar_data) or polar_data[af_id] is None:
            cl_matrix[i, :] = np.nan
            cd_matrix[i, :] = np.nan
            continue
        polar_df = polar_data[af_id]

        # Interpolate Cl and Cd
        cl_interp = sp_interp.interp1d(
            polar_df['Alpha'], polar_df['Cl'],
            kind='linear', fill_value="extrapolate"
        )
        cd_interp = sp_interp.interp1d(
            polar_df['Alpha'], polar_df['Cd'],
            kind='linear', fill_value="extrapolate"
        )
        cl_matrix[i, :] = cl_interp(alpha_range)
        cd_matrix[i, :] = cd_interp(alpha_range)
    return r_values, alpha_values, cl_matrix, cd_matrix

def plot_airfoils(coords_data, show_plot=True):
    """Plots the airfoil shapes in one figure.
    
    Args:
        coords_data: List of DataFrames with airfoil coordinates
        show_plot: If True, displays the plot (set to False for testing)
    """
    fig = plt.figure(figsize=(8, 4))
    for i, df in enumerate(coords_data):
        if df is not None:
            plt.plot(df['x/c'], df['y/c'], label=f'Airfoil {i}')
    plt.xlabel('x/c')
    plt.ylabel('y/c')
    plt.title('Airfoil Shapes')
    plt.grid(True)
    if show_plot:
        plt.show()
    return fig  # Always return the figure


def compute_induction_factors(r, V0, theta_p, omega, BlSpn, BlTwist, BlChord, BlAFID, polar_data,
                            b=3, rho=1.225, tol=1e-5, max_iter=200, relaxation=0.05):
    """Compute axial (a) and tangential (a_prime) induction factors.

    Args:
        r: Spanwise positions.
        V0: Freestream wind speed.
        theta_p: Blade pitch angle [deg].
        omega: Rotor rotational speed [rad/s].
        BlSpn, BlTwist, BlChord, BlAFID: Blade geometry parameters.
        polar_data: List of polar DataFrames for airfoils.
        B: Number of blades.
        rho: Air density.
        tol: Convergence tolerance.
        max_iter: Maximum iterations per span point.
        relaxation: Under-relaxation factor.

    Returns:
        Tuple of axial and tangential induction factor arrays.
    """
    a = np.zeros_like(r)
    a_prime = np.zeros_like(r)
    # Create interpolators with bounds checking
    def safe_interp1d(x, y, kind='linear'):
        return sp.interpolate.interp1d(x, y, kind=kind,
                                       bounds_error=False, fill_value=(y[0], y[-1]))
    twist_interp = safe_interp1d(BlSpn, BlTwist)
    chord_interp = safe_interp1d(BlSpn, BlChord)
    afid_interp = sp.interpolate.interp1d(BlSpn, BlAFID,
                                          kind='nearest', bounds_error=False, fill_value=BlAFID[0])
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
            sigma = (b * chord) / (2 * np.pi * ri)
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
                tip_loss_arg = -b/2 * (BlSpn[-1] - ri)/(ri * sin_phi)
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


def compute_dT(r, dr, rho, v_inflow, axial_factor):
    """Compute the differential thrust at a given radial position on the blade.

    Args:
        r: Radial positions along the blade (array).
        dr: Radial slice thickness.
        rho: Air density.
        V_inflow: Inflow wind speed.
        axial_factor: Axial induction factor at each radial position.

    Returns:
        Differential thrust at each radial position.
    """
    return abs(4 * np.pi * r * rho * v_inflow**2 * axial_factor * (1 - axial_factor) * dr)

def compute_dM(r, dr, rho, v_inflow, axial_factor, tangential_factor, omega):
    """Compute the differential moment at a given radial position on the blade.

    Args:
        r: Radial positions along the blade (array).
        dr: Radial slice thickness.
        rho: Air density.
        V_inflow: Inflow wind speed.
        axial_factor: Axial induction factor at each radial position.
        tangential_factor: Tangential induction factor at each radial position.
        omega: Rotational speed of the rotor.

    Returns:
        Differential moment at each radial position.
    """
    return 4 * np.pi * r**3 * rho * v_inflow * omega * tangential_factor * (1 - axial_factor) * dr

def compute_aerodynamic_power(torque, rotational_speed):
    """Compute the aerodynamic power generated by the turbine.

    Args:
        torque: Torque generated by the turbine.
        rotational_speed: Rotational speed of the rotor.

    Returns:
        Aerodynamic power generated by the turbine.
    """
    return torque * rotational_speed

def compute_optimal_strategy(
    V: np.ndarray,
    phi: np.ndarray,
    omega: np.ndarray,
    P: np.ndarray,
    T: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Computes the optimal pitch and rotational speed for each wind speed.

    Args:
        V: Wind speed array [m/s].
        phi: Pitch angle array [deg].
        omega: Rotational speed array [rad/s].
        P: Power array [W].
        T: Thrust array [N].

    Returns:
        A tuple of arrays:
            - V_unique: Unique wind speeds.
            - phi_optimal: Optimal pitch angles for each wind speed.
            - omega_optimal: Optimal rotational speeds.
            - P_optimal: Corresponding power.
            - T_optimal: Corresponding thrust.
    """
    # Ensure input types are arrays
    V = np.asarray(V)
    phi = np.asarray(phi)
    omega = np.asarray(omega)
    P = np.asarray(P)
    T = np.asarray(T)

    # Unique wind speeds
    v_unique = np.unique(V)

    # Preallocate result arrays
    phi_optimal = np.zeros_like(v_unique)
    omega_optimal = np.zeros_like(v_unique)
    p_optimal = np.zeros_like(v_unique)
    t_optimal = np.zeros_like(v_unique)

    # Find optimal operation point (max power) for each wind speed
    for i, wind_speed in enumerate(v_unique):
        mask = V == wind_speed

        if not np.any(mask):
            continue

        max_power_idx = np.argmax(P[mask])

        phi_optimal[i] = phi[mask][max_power_idx]
        omega_optimal[i] = omega[mask][max_power_idx]
        p_optimal[i] = P[mask][max_power_idx]
        t_optimal[i] = T[mask][max_power_idx]

    return v_unique, phi_optimal, omega_optimal, p_optimal, t_optimal

def get_airfoil_coefficients(
    alpha: Union[float, np.ndarray],
    r_position: float,
    blspn: np.ndarray,
    blafid: np.ndarray,
    polar_data: list
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gets interpolated airfoil coefficients (Cl, Cd) 
    at specific span position and angle(s) of attack.

    Args:
        alpha: Angle of attack in degrees (can be float or array).
        r_position: Spanwise blade position [m].
        BlSpn: Array of blade span coordinates [m].
        BlAFID: Array of airfoil IDs.
        polar_data: List of DataFrames with polar data for each airfoil.

    Returns:
        Tuple containing:
            - Cl: Interpolated lift coefficient(s).
            - Cd: Interpolated drag coefficient(s).
    """
    # Find nearest index to the specified span position
    idx = int(np.argmin(np.abs(blspn - r_position)))
    af_id = int(blafid[idx]) - 1  # Convert to 0-based indexing

    # Validate airfoil ID
    if af_id < 0 or af_id >= len(polar_data) or polar_data[af_id] is None:
        return np.full_like(alpha, np.nan), np.full_like(alpha, np.nan)

    # Retrieve polar data
    polar_df = polar_data[af_id]

    # Set up interpolators for Cl and Cd
    cl_interp = sp.interpolate.interp1d(
        polar_df["Alpha"], polar_df["Cl"],
        kind="linear", bounds_error=False,
        fill_value=(polar_df["Cl"].iloc[0], polar_df["Cl"].iloc[-1])
    )
    cd_interp = sp.interpolate.interp1d(
        polar_df["Alpha"], polar_df["Cd"],
        kind="linear", bounds_error=False,
        fill_value=(polar_df["Cd"].iloc[0], polar_df["Cd"].iloc[-1])
    )

    # Return interpolated values
    return cl_interp(alpha), cd_interp(alpha)

def analyze_airfoil_performance(
    blspn: np.ndarray,
    blafid: np.ndarray,
    polar_data: List[DataFrame]
) -> None:
    """
    Analyze airfoil performance at root, mid-span, and tip blade positions.
    
    Plots Cl and Cd vs. angle of attack and prints a summary table including
    Cl/Cd ratios and stall detection.
    
    Args:
        BlSpn: Array of spanwise positions along the blade [m].
        BlAFID: Array of airfoil IDs corresponding to BlSpn.
        polar_data: List of polar data DataFrames (each with 'Alpha', 'Cl', 'Cd').
    """
    print("\n" + "=" * 80)
    print("AIRFOIL PERFORMANCE ANALYSIS AT KEY BLADE SECTIONS")
    print("=" * 80)

    # Define representative span positions (excluding blade root/tip extremes)
    positions = [
        {"name": "Root", "r": blspn[10]},
        {"name": "Mid-span", "r": blspn[len(blspn) // 2]},
        {"name": "Tip", "r": blspn[-5]},
    ]

    # Define angle of attack range for plotting
    alpha_test = np.linspace(-5, 15, 21)

    # Create figure for plotting
    plt.figure(figsize=(15, 6))

    for i, pos in enumerate(positions):
        cl_values, cd_values = get_airfoil_coefficients(
            alpha_test, pos["r"], blspn, blafid, polar_data
        )

        # Plot Cl
        plt.subplot(1, 2, 1)
        plt.plot(alpha_test, cl_values, "o-", label=f'{pos["name"]} (r={pos["r"]:.1f}m)')

        # Plot Cd
        plt.subplot(1, 2, 2)
        plt.plot(alpha_test, cd_values, "o-", label=f'{pos["name"]} (r={pos["r"]:.1f}m)')

    # Format Lift plot
    plt.subplot(1, 2, 1)
    plt.title("Lift Coefficient vs Angle of Attack")
    plt.xlabel("Angle of Attack [deg]")
    plt.ylabel("Lift Coefficient (Cl)")
    plt.grid(True)
    plt.legend()

    # Format Drag plot
    plt.subplot(1, 2, 2)
    plt.title("Drag Coefficient vs Angle of Attack")
    plt.xlabel("Angle of Attack [deg]")
    plt.ylabel("Drag Coefficient (Cd)")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Print Cl/Cd summary table at selected angles
    print("\nDetailed Airfoil Coefficients:")
    print("-" * 85)
    print(f"{'Position':<12}{'r [m]':<8}{'α [°]':<6}{'Cl':<8}{'Cd':<8}{'Cl/Cd':<10}")
    print("-" * 85)

    stall_threshold = 0.1  # Cd increase indicating stall

    for pos in positions:
        alphas = [-5, 0, 5, 10, 15]
        prev_cd = None

        for alpha in alphas:
            cl_val, cd_val = get_airfoil_coefficients(alpha, pos["r"], blspn, blafid, polar_data)
            cl_over_cd = cl_val / cd_val if cd_val > 0 else float("inf")

            stall_marker = ""
            if prev_cd is not None and (cd_val - prev_cd) > stall_threshold:
                stall_marker = "STALL"
            prev_cd = cd_val

            print(
    f"{pos['name']:<12}"
    f"{pos['r']:<8.1f}"
    f"{alpha:<6.1f}"
    f"{cl_val:<8.3f}"
    f"{cd_val:<8.4f}"
    f"{cl_over_cd:<10.1f}"
    f"{stall_marker}"
)


def identify_operational_mode(
    v0: Union[float, np.ndarray],
    phi_opt: Union[float, np.ndarray],
    omega_opt: Union[float, np.ndarray],
    omega_rated: float = None
) -> Union[Tuple[str, Dict], Tuple[List[str], List[Dict]]]:
    """
    Identifies turbine operational mode based on wind speed, pitch angle, and rotational speed.

    Args:
        V0: Wind speed [m/s] (scalar or array)
        phi_opt: Optimal pitch angle [deg] (same shape as V0)
        omega_opt: Optimal rotational speed [rad/s] (same shape as V0)
        omega_rated: Rated rotational speed [rad/s], optional

    Returns:
        If scalar input: (mode_string, info_dict)
        If array input: (list of mode strings, list of info dicts)
    """
    # Convert to numpy arrays for consistent processing
    v0_arr = np.atleast_1d(v0)
    phi_arr = np.atleast_1d(phi_opt)
    omega_arr = np.atleast_1d(omega_opt)

    # Determine rated speed
    omega_max = np.max(omega_arr) if omega_rated is None else omega_rated

    modes = []
    infos = []

    for v, phi, omega in zip(v0_arr, phi_arr, omega_arr):
        if v < 4:
            mode = "Start-up"
            info = {
                "description": "Below cut-in speed",
                "color": "red",
                "pitch": phi,
                "speed": omega
            }
        elif phi < 1 and omega < 0.9 * omega_max:
            mode = "Partial load"
            info = {
                "description": "Max power tracking",
                "control": "Variable speed, fixed pitch",
                "color": "green",
                "pitch": phi,
                "speed": omega
            }
        elif phi < 5 and omega >= 0.9 * omega_max:
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

    # Return single result if input was scalar
    if np.isscalar(v0):
        return modes[0], infos[0]

    return modes, infos
