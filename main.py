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


# Compute Cl and Cd vs r and α
r_values, alpha_values, Cl_matrix, Cd_matrix = src.compute_cl_cd_vs_r_alpha(
    BlSpn, BlAFID, polar_data, alpha_range=np.linspace(-180, 180, 100)
)

# Plot Cl and Cd for a few span positions
num_positions = 5
span_indices = sorted(set(np.linspace(0, len(r_values)-1, num_positions, dtype=int).tolist() + [0, len(r_values)-1]))
plt.figure(figsize=(12, 6))

for idx in span_indices:
    r = r_values[idx]
    plt.plot(alpha_values, Cl_matrix[idx, :], label=f"Cl at r={r:.1f}m")
    plt.plot(alpha_values, Cd_matrix[idx, :], '--', label=f"Cd at r={r:.1f}m")

plt.xlabel("Angle of Attack (α) [deg]")
plt.ylabel("Coefficient Value")
plt.title("Lift (Cl) and Drag (Cd) Coefficients vs Angle of Attack at Selected Span Positions")
plt.grid(True)
plt.legend()
plt.show()

src.plot_airfoils(coords)

