import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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
    data = pd.read_csv(file_path, sep='\s+', header=None, skiprows=6)  # Skip the first row with non-relevant info

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