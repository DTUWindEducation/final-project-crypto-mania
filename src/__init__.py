import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
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



def compute_lift_drag_for_span_and_alpha(BlSpn, alpha, polar_data):
    """
    Computes the lift (Cl) and drag (Cd) coefficients for a given span position (r)
    and angle of attack (α) across all airfoils.

    Parameters:
    - BlSpn: Spanwise position (r) of the blade
    - alpha: Angle of attack (α) for which to compute C_l and C_d
    - polar_data: List of pandas DataFrames for each airfoil polar data (Cl, Cd vs Alpha)

    Returns:
    - Cl_values: List of Cl values for each airfoil at given alpha
    - Cd_values: List of Cd values for each airfoil at given alpha
    """
    Cl_values = []
    Cd_values = []
    
    # Loop over all airfoils in polar data
    for airfoil_polar in polar_data:
        if airfoil_polar is not None:
            # Extract alpha, Cl, and Cd values
            alpha_values = airfoil_polar['Alpha'].values
            Cl = airfoil_polar['Cl'].values
            Cd = airfoil_polar['Cd'].values

            # Interpolate Cl and Cd using cubic splines
            Cl_interp = interpolate.interp1d(alpha_values, Cl, kind='cubic', fill_value="extrapolate")
            Cd_interp = interpolate.interp1d(alpha_values, Cd, kind='cubic', fill_value="extrapolate")
            
            # Compute Cl and Cd at the given angle of attack
            Cl_values.append(Cl_interp(alpha))
            Cd_values.append(Cd_interp(alpha))
    
    return Cl_values, Cd_values

def compute_lift_drag_for_all_angles(BlSpn, polar_data, angle_range=(-180, 180, 1)):
    """
    Computes Cl and Cd for all airfoils over the entire span and angle of attack range.

    Parameters:
    - BlSpn: Array of span positions (r) for the blade
    - polar_data: List of airfoil polar data (50 airfoils)
    - angle_range: Tuple (min_angle, max_angle, step) for the angle of attack

    Returns:
    - cl_all: 2D array with Cl values for each angle of attack and airfoil
    - cd_all: 2D array with Cd values for each angle of attack and airfoil
    - angles: Array of angles of attack
    """
    # Create an array of angle values from -180 to 180 degrees (with the specified step)
    angles = np.arange(angle_range[0], angle_range[1] + angle_range[2], angle_range[2])
    
    # Initialize lists to hold Cl and Cd for all airfoils
    cl_all = []
    cd_all = []
    
    # Loop through each angle of attack and compute Cl and Cd for each airfoil at all span positions
    for alpha in angles:
        Cl_list, Cd_list = compute_lift_drag_for_span_and_alpha(BlSpn, alpha, polar_data)
        cl_all.append(Cl_list)
        cd_all.append(Cd_list)
    
    # Convert lists to numpy arrays for easier manipulation
    cl_all = np.array(cl_all)
    cd_all = np.array(cd_all)
    
    return cl_all, cd_all, angles
