"""
Different Imports
"""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import src


class TurbineSimulator:
    """
    Class for TurbineSimulator
    """
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self._load_data()

    def _load_data(self):
        # Load response data
        turbine_data = self.data_dir / 'IEA_15MW_RWT_Onshore.opt'
        self.v, self.phi, self.omega, self.p, self.t = src.load_resp(turbine_data)

        # Load blade data
        blade_data = self.data_dir / 'IEA-15-240-RWT_AeroDyn15_blade.dat'
        (
            self.blspn,
            self.blcrvac,
            self.blswpac,
            self.blcrvang,
            self.bltwist,
            self.blchord,
            self.blafid,
            self.blcb,
            self.blcenbn,
            self.blcenbt) = src.load_blade_data(blade_data)

        # Load airfoils polar and coords
        airfoil_data_path = self.data_dir / 'Airfoils'
        self.polar_data = src.load_polar_data(airfoil_data_path)
        self.coords = src.load_af_coords(airfoil_data_path)

    def plot_initial_data(self):
        """
        Definition on plot
        """
        src.plot_V_vs_phi(self.phi, self.v)

        # Compute Cl and Cd
        (
            self.r_values,
            self.alpha_values,
            self.cl_matrix,
            self.cd_matrix
        ) = src.compute_cl_cd_vs_r_alpha(
            self.blspn,
            self.blafid,
            self.polar_data,
            alpha_range=np.linspace(-180, 180, 100)
        )

        num_positions = 5
        linspace_indices = np.linspace(
            0,
            len(self.r_values) - 1,
            num_positions,
            dtype=int
        ).tolist()

        endpoints = [0, len(self.r_values) - 1]

        span_indices = sorted(set(linspace_indices + endpoints))

        plt.figure(figsize=(12, 6))
        for idx in span_indices:
            r = self.r_values[idx]
            plt.plot(self.alpha_values, self.cl_matrix[idx, :], label=f"Cl at r={r:.1f}m")
            plt.plot(self.alpha_values, self.cd_matrix[idx, :], '--', label=f"Cd at r={r:.1f}m")
        plt.xlabel("Angle of Attack (α) [deg]")
        plt.ylabel("Coefficient Value")
        plt.title("Lift (Cl) and Drag (Cd) Coefficients vs " \
            "Angle of Attack at Selected Span Positions")
        plt.grid(True)
        plt.legend()
        plt.show()

        src.plot_airfoils(self.coords)

    def run_test_cases(self):
        """
        Definition for running test cases
        """
        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
        plt.figure(figsize=(12, 6))

        test_cases = [
            {'V0': 6, 'theta_p': 0, 'omega': 5},
            {'V0': 8, 'theta_p': 2, 'omega': 6},
            {'V0': 10, 'theta_p': 5, 'omega': 7.5},
            {'V0': 12, 'theta_p': 10, 'omega': 9}
        ]

        for idx, case in enumerate(test_cases):
            print(
                    f"\nRunning case: V0={case['V0']}m/s, "
                    f"theta_p={case['theta_p']}°, "
                    f"omega={case['omega']}rad/s"
                )
            try:
                a, a_prime = src.compute_induction_factors(
                    r=self.blspn,
                    V0=case['V0'],
                    theta_p=case['theta_p'],
                    omega=case['omega'],
                    BlSpn=self.blspn,
                    BlTwist=self.bltwist,
                    BlChord=self.blchord,
                    BlAFID=self.blafid,
                    polar_data=self.polar_data
                )

                valid = ~np.isnan(a) & ~np.isnan(a_prime)
                print(f"Successfully computed {sum(valid)}/{len(self.blspn)} points")
                print(
                    f"a: min={np.nanmin(a):.3f}, "
                    f"max={np.nanmax(a):.3f}, "
                    f"mean={np.nanmean(a):.3f}"
                )
                print(
                    f"a': min={np.nanmin(a_prime):.3f}, "
                    f"max={np.nanmax(a_prime):.3f}, "
                    f"mean={np.nanmean(a_prime):.3f}"
                )

                color = colors[idx % len(colors)]
                label_base = f"V0={case['V0']} m/s, θ={case['theta_p']}°, ω={case['omega']} rad/s"
                plt.plot(
                    self.blspn[valid],
                    a[valid],
                    color=color,
                    linestyle='-',
                    label=f"a | {label_base}")
                plt.plot(
                    self.blspn[valid],
                    a_prime[valid],
                    color=color,
                    linestyle='--',
                    label=f"a' | {label_base}")

                # Compute dT, dM, and power
                rho = 1.225
                dr = np.gradient(self.blspn)
                dt = src.compute_dT(
                self.blspn[valid],
                dr[valid],
                rho,
                case['V0'] * (1 - a[valid]),
                a[valid]
                )
                dm = src.compute_dM(
                self.blspn[valid],
                dr[valid],
                rho,
                case['V0'] * (1 - a[valid]),
                a[valid],
                a_prime[valid],
                case['omega']
                )

                total_thrust = np.nansum(dt) * 3
                total_torque = abs(np.nansum(dm) * 3)
                p_aero = src.compute_aerodynamic_power(total_torque, case['omega'])

                print(f"Total aerodynamic thrust: {total_thrust / 1e3:.2f} kN")
                print(f"Total aerodynamic torque: {total_torque / 1e3:.2f} kNm")
                print(f"Aerodynamic power: {p_aero / 1e6:.2f} MW")

            except Exception as e:
                print(f"Failed to compute case: {str(e)}")

        plt.xlabel('Span position [m]')
        plt.ylabel('Induction factor')
        plt.title("Axial (a) and Tangential (a′) Induction Factors Across Operating Conditions")
        plt.ylim(-0.2, 1.2)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def analyze_strategy(self):
        """
        Definition for analyze strat
        """
        (
    v_unique,
    phi_opt,
    omega_opt,
    p_opt,
    t_opt
) = src.compute_optimal_strategy(
    self.v,
    self.phi,
    self.omega,
    self.p,
    self.t
)

        # Plot pitch and rotational speed
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        plt.plot(v_unique, phi_opt, 'b-', label='Optimal pitch angle')
        plt.xlabel('Wind speed [m/s]')
        plt.ylabel('Pitch angle [deg]')
        plt.title('Optimal Pitch Angle vs Wind Speed')
        plt.grid(True)
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(v_unique, omega_opt, 'r-', label='Optimal rotational speed')
        plt.xlabel('Wind speed [m/s]')
        plt.ylabel('Rotational speed [rad/s]')
        plt.title('Optimal Rotational Speed vs Wind Speed')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.show()

        # Plot power and thrust
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        plt.plot(v_unique, p_opt / 1e3, 'g-', label='Optimal power')
        plt.xlabel('Wind speed [m/s]')
        plt.ylabel('Power [MW]')
        plt.title('Power Curve')
        plt.grid(True)
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(v_unique, t_opt / 1e3, 'm-', label='Optimal thrust')
        plt.xlabel('Wind speed [m/s]')
        plt.ylabel('Thrust [kN]')
        plt.title('Thrust Curve')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.show()

        # Display numeric strategy
        mask = (v_unique >= 1) & (v_unique <= 25)
        v_filtered = v_unique[mask]
        phi_filtered = phi_opt[mask]
        omega_filtered = omega_opt[mask]

        print("\nOptimal Operational Strategy")
        print("----------------------------")
        print(f"{'V0 [m/s]':<10}{'θp [deg]':<12}{'ω [rad/s]':<12}")
        print("-" * 30)
        for v, p, w in zip(v_filtered, phi_filtered, omega_filtered):
            print(f"{v:<10.1f}{p:<12.2f}{w:<12.2f}")
        print("\nNote: Values are rounded to 2 decimal places for display")

        # Operational mode analysis
        modes, infos = src.identify_operational_mode(v_unique, phi_opt, omega_opt)
        if isinstance(modes, list):
            print("\nOperational Mode Analysis:")
            print("-------------------------")
            for v, mode, info in zip(v_unique, modes, infos):
                print(
    f"At {v:.1f} m/s: {mode} - {info['description']} "
    f"(Pitch: {info['pitch']:.2f}°, Speed: {info['speed']:.2f} rad/s)"
)
        else:
            print(f"\nAt {v_unique:.1f} m/s: {modes} - {infos['description']}")

    def run_airfoil_analysis(self):
        """
        Function for airfoil analysis
        """
        src.analyze_airfoil_performance(self.blspn, self.blafid, self.polar_data)


if __name__ == '__main__':
    DATA_DIR = Path(__file__).resolve().parent / 'inputs/IEA-15-240-RWT'
    sim = TurbineSimulator(DATA_DIR)
    sim.plot_initial_data()
    sim.run_test_cases()
    sim.analyze_strategy()
    sim.run_airfoil_analysis()
