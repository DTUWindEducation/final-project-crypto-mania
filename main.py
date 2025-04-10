import src
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import time

# Define data directory
DATA_DIR = Path(__file__).resolve().parent / 'inputs/IEA-15-240-RWT'

# Load response data
turbine_data = DATA_DIR / 'IEA_15MW_RWT_Onshore.opt'
V, phi, omega, P, T = src.load_resp(turbine_data)

# Load blade data
blade_data = DATA_DIR / 'IEA-15-240-RWT_AeroDyn15_blade.dat'
BlSpn, BlCrvAC, BlSwpAC, BlCrvAng, BlTwist, BlChord, BlAFID, BlCb, BlCenBn, BlCenBt = src.load_blade_data(blade_data)

# Load airfoils polar data
airfoil_data_path = DATA_DIR / 'Airfoils'
polar_data = src.load_polar_data(airfoil_data_path)

# Load coords data
airfoil_data_path = DATA_DIR / 'Airfoils'
coords = src.load_af_coords(airfoil_data_path)

# Plot phi vs V
src.plot_V_vs_phi(phi, V)

src.plot_airfoils(coords)
