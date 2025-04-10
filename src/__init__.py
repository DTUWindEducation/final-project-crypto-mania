import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy as sp

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