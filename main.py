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

# Plot airfoil shapes
src.plot_airfoils(coords)

# Define a color palette
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

# Prepare the plot for induction factors
plt.figure(figsize=(12, 6))

# Test multiple operating conditions
test_cases = [
    {'V0': 6, 'theta_p': 0, 'omega': 5},
    {'V0': 8, 'theta_p': 2, 'omega': 6},
    {'V0': 10, 'theta_p': 5, 'omega': 7.5},
    {'V0': 12, 'theta_p': 10, 'omega': 9}
]

for idx, case in enumerate(test_cases):
    print(f"\nRunning case: V0={case['V0']}m/s, theta_p={case['theta_p']}°, omega={case['omega']}rad/s")

    try:
        a, a_prime = src.compute_induction_factors(
            r=BlSpn,
            V0=case['V0'],
            theta_p=case['theta_p'],
            omega=case['omega'],
            BlSpn=BlSpn,
            BlTwist=BlTwist,
            BlChord=BlChord,
            BlAFID=BlAFID,
            polar_data=polar_data
        )

        valid = ~np.isnan(a) & ~np.isnan(a_prime)
        print(f"Successfully computed {sum(valid)}/{len(BlSpn)} points")
        print(f"a: min={np.nanmin(a):.3f}, max={np.nanmax(a):.3f}, mean={np.nanmean(a):.3f}")
        print(f"a': min={np.nanmin(a_prime):.3f}, max={np.nanmax(a_prime):.3f}, mean={np.nanmean(a_prime):.3f}")

        color = colors[idx % len(colors)]
        label_base = f"V0={case['V0']} m/s, θ={case['theta_p']}°, ω={case['omega']} rad/s"
        plt.plot(BlSpn[valid], a[valid], color=color, linestyle='-', label=f"a | {label_base}")
        plt.plot(BlSpn[valid], a_prime[valid], color=color, linestyle='--', label=f"a' | {label_base}")

        # ---- NEW: Calculate dT, dM, and aerodynamic power ----
        rho = 1.225
        dr = np.gradient(BlSpn)  # differential span element

        dT = src.compute_dT(BlSpn[valid], dr[valid], rho, case['V0'] * (1 - a[valid]), a[valid])
        dM = src.compute_dM(BlSpn[valid], dr[valid], rho, case['V0'] * (1 - a[valid]), a[valid], a_prime[valid], case['omega'])

        Total_Thrust = np.nansum(dT) * 3   # Multiply by number of blades
        Total_Torque = abs(np.nansum(dM) * 3)

        P_aero = src.compute_aerodynamic_power(Total_Torque, case['omega'])

        print(f"Total aerodynamic thrust: {Total_Thrust/1e3:.2f} kN")
        print(f"Total aerodynamic torque: {Total_Torque/1e3:.2f} kNm")
        print(f"Aerodynamic power: {P_aero/1e6:.2f} MW")

    except Exception as e:
        print(f"Failed to compute case: {str(e)}")

# Finalize plot
plt.xlabel('Span position [m]')
plt.ylabel('Induction factor')
plt.title("Axial (a) and Tangential (a′) Induction Factors Across Operating Conditions")
plt.ylim(-0.2, 1.2)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

        # Store values for plotting
r_valid = BlSpn[valid]
plt.figure(figsize=(15, 8))


        # Plot differential Thrust
plt.subplot(2, 2, 1)
plt.plot(r_valid, dT, label=f"V0={case['V0']} m/s")
plt.xlabel("Span position r [m]")
plt.ylabel("dT [N/m]")
plt.title("Differential Thrust along Blade Span")
plt.grid(True)
plt.legend()

        # Plot differential Torque
plt.subplot(2, 2, 2)
plt.plot(r_valid, dM, label=f"V0={case['V0']} m/s")
plt.xlabel("Span position r [m]")
plt.ylabel("dM [Nm/m]")
plt.title("Differential Torque along Blade Span")
plt.grid(True)
plt.legend()

        # Plot cumulative Thrust
plt.subplot(2, 2, 3)
cumulative_Thrust = np.cumsum(dT) * 3  # Three blades
plt.plot(r_valid, abs(cumulative_Thrust), label=f"V0={case['V0']} m/s")
plt.xlabel("Span position r [m]")
plt.ylabel("Cumulative Thrust [N]")
plt.title("Cumulative Thrust along Blade Span")
plt.grid(True)
plt.legend()

        # Plot cumulative Torque
plt.subplot(2, 2, 4)
cumulative_Torque = np.cumsum(dM) * 3
plt.plot(r_valid, abs(cumulative_Torque), label=f"V0={case['V0']} m/s")
plt.xlabel("Span position r [m]")
plt.ylabel("Cumulative Torque [Nm]")
plt.title("Cumulative Torque along Blade Span")
plt.grid(True)
plt.legend()

plt.suptitle(f"Load distributions for V0={case['V0']} m/s, θ={case['theta_p']}°, ω={case['omega']} rad/s", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


