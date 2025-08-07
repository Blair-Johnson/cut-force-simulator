# Milling Force and Power Simulator (Imperial Version)
# This program simulates cutting forces in an end milling operation
# based on a mechanistic model. It accepts all parameters in Imperial units
# via command-line arguments.
#
# VERSION 15.0: Adds chip thickness calculation to console output.
# Aligns the zero-point on the combined force/torque plot.

import numpy as np
import matplotlib.pyplot as plt
import argparse

def moving_average(data, window_size):
    """Applies a simple moving average filter to smooth data."""
    if window_size <= 1:
        return data
    smoothed = np.convolve(data, np.ones(window_size)/window_size, mode='valid')
    pad_size = (len(data) - len(smoothed)) // 2
    return np.pad(smoothed, (pad_size, len(data) - len(smoothed) - pad_size), 'edge')

def simulate_milling_forces(params):
    """
    Simulates milling forces for one revolution using an axial slicing model.
    """
    # --- 1. Unpack Imperial Parameters & Convert to Metric for Calculation ---
    IN_TO_MM = 25.4
    PSI_TO_MPA = 0.00689476
    N_TO_LBF = 1 / 4.44822
    W_TO_HP = 1 / 745.7
    W_TO_KW = 1 / 1000.0
    N_M_TO_IN_LBF = 8.85075

    D_mm = params['tool_diameter_in'] * IN_TO_MM
    R_mm = D_mm / 2.0
    R_m = R_mm / 1000.0
    num_flutes = params['num_flutes']
    helix_angle_deg = params['helix_angle_deg']
    helix_angle_rad = np.deg2rad(helix_angle_deg)
    helix_dir = params['helix_dir']
    
    doc_mm = params['doc_in'] * IN_TO_MM
    woc_mm = params['woc_in'] * IN_TO_MM
    spindle_speed_rpm = params['spindle_speed_rpm']
    feed_rate_mm_min = params['feed_rate_in_min'] * IN_TO_MM
    
    uts_mpa = params['material_uts_psi'] * PSI_TO_MPA
    efficiency = params['machine_efficiency']
    Kr = params['radial_force_ratio']
    milling_type = params.get('milling_type', 'conventional').lower()
    n_slices = params['n_slices']
    smoothing_window = params['smoothing_window']
    power_factor = params['power_factor']

    # --- 2. Calculate Derived Parameters (in Metric) ---
    kc = 2.0 * uts_mpa
    f_z_mm = feed_rate_mm_min / (spindle_speed_rpm * num_flutes)
    omega_rad_s = spindle_speed_rpm * 2 * np.pi / 60.0
    dz = doc_mm / n_slices

    # --- 3. Determine Flute Engagement Angles ---
    if woc_mm > D_mm: woc_mm = D_mm
    phi_exit_conv = np.arccos(1 - (2 * woc_mm / D_mm))

    if milling_type == 'climb':
        phi_entry = np.pi - phi_exit_conv
        phi_exit = np.pi
        print(f"Simulating: Climb Milling with {n_slices} axial slices.")
    else:
        phi_entry = 0
        phi_exit = phi_exit_conv
        print(f"Simulating: Conventional Milling with {n_slices} axial slices.")

    # --- 4. Simulation Loop with Axial Discretization ---
    num_steps = 360
    rotation_angles = np.linspace(0, 2 * np.pi, num_steps, endpoint=False)
    
    Fx_total_N = np.zeros(num_steps)
    Fy_total_N = np.zeros(num_steps)
    Fz_total_N = np.zeros(num_steps)
    Ft_total_N = np.zeros(num_steps)

    for i, phi_tool in enumerate(rotation_angles):
        for k in range(n_slices):
            z = k * dz
            phi_lag = (z / R_mm) * np.tan(helix_angle_rad)
            for j in range(num_flutes):
                phi_flute_base = phi_tool - j * (2 * np.pi / num_flutes)
                phi_flute_at_z = (phi_flute_base - phi_lag) % (2 * np.pi)

                if phi_entry <= phi_flute_at_z <= phi_exit:
                    h = f_z_mm * np.sin(phi_flute_at_z)
                    if h < 1e-6: continue

                    dFt_slice = kc * dz * h
                    dFr_slice = Kr * dFt_slice
                    
                    dFx = -dFt_slice * np.cos(phi_flute_at_z) + dFr_slice * np.sin(phi_flute_at_z)
                    dFy = -dFt_slice * np.sin(phi_flute_at_z) - dFr_slice * np.cos(phi_flute_at_z)
                    
                    dFz = -dFt_slice * np.tan(helix_angle_rad)
                    if helix_dir == 'left':
                        dFz *= -1

                    Fx_total_N[i] += dFx
                    Fy_total_N[i] += dFy
                    Fz_total_N[i] += dFz
                    Ft_total_N[i] += dFt_slice

    # --- 5. Smooth Force Data to Remove Artifacts ---
    print(f"Applying smoothing filter with window size: {smoothing_window}")
    Fx_total_N = moving_average(Fx_total_N, smoothing_window)
    Fy_total_N = moving_average(Fy_total_N, smoothing_window)
    Fz_total_N = moving_average(Fz_total_N, smoothing_window)
    Ft_total_N = moving_average(Ft_total_N, smoothing_window)
    
    # --- 6. Calculate Power, Torque, Current, and Convert Results ---
    torque_Nm = Ft_total_N * R_m
    cutting_power_W = torque_Nm * omega_rad_s
    machine_power_W = cutting_power_W / efficiency
    
    Fx_total_lbf = Fx_total_N * N_TO_LBF
    Fy_total_lbf = Fy_total_N * N_TO_LBF
    Fz_total_lbf = Fz_total_N * N_TO_LBF
    torque_in_lbf = torque_Nm * N_M_TO_IN_LBF
    
    avg_machine_power_W = np.mean(machine_power_W)
    I_rms_A = avg_machine_power_W / (120 * power_factor)

    if params['power_in_hp']:
        power_unit_conv = W_TO_HP
    else:
        power_unit_conv = W_TO_KW
        
    machine_power_out = machine_power_W * power_unit_conv
    cutting_power_out = cutting_power_W * power_unit_conv

    # Calculate chip thickness stats
    f_z_in = f_z_mm / IN_TO_MM
    max_actual_h_mm = f_z_mm * np.sin(phi_exit_conv)
    max_actual_h_in = max_actual_h_mm / IN_TO_MM

    stats = {
        'F_peak_x_lbf': np.max(np.abs(Fx_total_lbf)), 'F_avg_x_lbf': np.mean(np.abs(Fx_total_lbf)),
        'F_peak_y_lbf': np.max(np.abs(Fy_total_lbf)), 'F_avg_y_lbf': np.mean(np.abs(Fy_total_lbf)),
        'F_peak_z_lbf': np.max(np.abs(Fz_total_lbf)), 'F_avg_z_lbf': np.mean(np.abs(Fz_total_lbf)),
        'T_peak_in_lbf': np.max(np.abs(torque_in_lbf)), 'T_avg_in_lbf': np.mean(np.abs(torque_in_lbf)),
        'P_peak_cutting': np.max(cutting_power_out), 'P_avg_cutting': np.mean(cutting_power_out),
        'P_peak_machine': np.max(machine_power_out), 'P_avg_machine': np.mean(machine_power_out),
        'I_rms_A': I_rms_A,
        'f_z_in': f_z_in, 'max_actual_h_in': max_actual_h_in
    }

    results = {
        'rotation_angles_deg': np.degrees(rotation_angles),
        'Fx_lbf': Fx_total_lbf, 'Fy_lbf': Fy_total_lbf, 'Fz_lbf': Fz_total_lbf,
        'torque_in_lbf': torque_in_lbf,
        'power_out': machine_power_out, 'stats': stats
    }
    
    return results

def perform_fft_analysis(results, params):
    """Performs FFT on force data, prints results, and optionally plots the spectrum."""
    print("\n--- Frequency Analysis ---")
    
    num_revs = params['num_revolutions']
    if num_revs > 1:
        print(f"Analyzing {num_revs} revolutions to improve frequency resolution.")
    
    force_signal = np.tile(results['Fx_lbf'], num_revs)
    
    rpm = params['spindle_speed_rpm']
    num_flutes = params['num_flutes']
    N = len(force_signal)
    T = (60.0 / rpm) * num_revs
    dt = T / N
    
    yf = np.fft.fft(force_signal)
    xf = np.fft.fftfreq(N, dt)
    
    positive_mask = xf >= 0
    xf_plot = xf[positive_mask]
    yf_plot = 2.0/N * np.abs(yf[positive_mask])
    yf_plot[0] = yf_plot[0] / 2.0

    spindle_freq = rpm / 60.0
    tooth_pass_freq = spindle_freq * num_flutes
    
    print(f"Spindle Frequency: {spindle_freq:.2f} Hz")
    print(f"Tooth Passing Frequency: {tooth_pass_freq:.2f} Hz")
    print(f"FFT Resolution (lowest freq): {1/T:.2f} Hz")

    peak_mag_idx = np.argmax(yf_plot[1:]) + 1
    peak_freq = xf_plot[peak_mag_idx]
    peak_mag = yf_plot[peak_mag_idx]
    print(f"Peak detected frequency: {peak_freq:.2f} Hz with magnitude {peak_mag:.2f} lbf")

    if params['analyze_fft']:
        plt.figure(figsize=(12, 7))
        plt.plot(xf_plot, yf_plot, color='black', label='Force Spectrum')
        
        plt.axvline(x=spindle_freq, color='cyan', linestyle='--', label=f'Spindle Freq ({spindle_freq:.1f} Hz)')
        plt.axvline(x=tooth_pass_freq, color='magenta', linestyle='--', label=f'Tooth Pass Freq ({tooth_pass_freq:.1f} Hz)')
        
        plt.title('Frequency Distribution of Feed Force (Fx)')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Force Magnitude (lbf)')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.xlim(0, tooth_pass_freq * 2.5)
        plt.tight_layout()

def plot_time_domain(results, params):
    """Plots the simulation force and torque results in the time/angle domain."""
    angles = results['rotation_angles_deg']
    separate_plots = params['separate_plots']
    
    fx = results['Fx_lbf']
    fy = results['Fy_lbf']
    fz = results['Fz_lbf']
    torque = results['torque_in_lbf']
    
    milling_type = params.get('milling_type', 'conventional').capitalize()
    
    if separate_plots:
        fig, axs = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
        fig.suptitle(f'Simulated Milling Forces & Torque ({milling_type} Milling)', fontsize=16)

        axs[0].plot(angles, fx, label='Feed Force (Fx)', color='blue')
        axs[0].set_ylabel('Force (lbf)')
        axs[0].grid(True, linestyle='--', alpha=0.6)
        axs[0].legend()
        
        axs[1].plot(angles, fy, label='Transverse Force (Fy)', color='green')
        axs[1].set_ylabel('Force (lbf)')
        axs[1].grid(True, linestyle='--', alpha=0.6)
        axs[1].legend()

        axs[2].plot(angles, fz, label='Axial Force (Fz)', color='red')
        axs[2].set_ylabel('Force (lbf)')
        axs[2].grid(True, linestyle='--', alpha=0.6)
        axs[2].legend()

        axs[3].plot(angles, torque, label='Torque', color='purple')
        axs[3].set_xlabel('Tool Rotation Angle (degrees)')
        axs[3].set_ylabel('Torque (in-lbf)')
        axs[3].grid(True, linestyle='--', alpha=0.6)
        axs[3].legend()

    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8))
        fig.suptitle(f'Simulated Milling Forces & Torque ({milling_type} Milling)', fontsize=16)

        p1, = ax1.plot(angles, fx, label='Feed Force (Fx)', color='blue')
        p2, = ax1.plot(angles, fy, label='Transverse Force (Fy)', color='green')
        p3, = ax1.plot(angles, fz, label='Axial Force (Fz)', color='red')
        ax1.set_xlabel('Tool Rotation Angle (degrees)')
        ax1.set_ylabel('Force (lbf)')
        
        ax2 = ax1.twinx()
        p4, = ax2.plot(angles, torque, label='Torque', color='purple', linestyle=':')
        ax2.set_ylabel('Torque (in-lbf)', color='purple')
        ax2.tick_params(axis='y', labelcolor='purple')

        # Symmetrically scale axes around zero for aligned origins
        max_abs_force = np.max(np.abs(np.concatenate((fx, fy, fz))))
        ax1.set_ylim(-max_abs_force * 1.1, max_abs_force * 1.1)
        
        max_abs_torque = np.max(np.abs(torque))
        ax2.set_ylim(-max_abs_torque * 1.1, max_abs_torque * 1.1)
        
        ax1.grid(True, linestyle='--', alpha=0.6)
        
        lines = [p1, p2, p3, p4]
        ax1.legend(lines, [l.get_label() for l in lines])

    plt.tight_layout(rect=[0, 0, 1, 0.96])

def main():
    parser = argparse.ArgumentParser(
        description="Simulate milling forces using an axial slicing model and Imperial units.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Tool Geometry
    parser.add_argument('--dia', type=float, default=0.25, dest='tool_diameter_in', help='Tool diameter (in)')
    parser.add_argument('--flutes', type=int, default=3, dest='num_flutes', help='Number of flutes')
    parser.add_argument('--helix', type=float, default=45.0, dest='helix_angle_deg', help='Helix angle (degrees)')
    parser.add_argument('--helix-dir', type=str, default='right', choices=['right', 'left'], dest='helix_dir', help='Helix direction (right=up-cut, left=down-cut)')
    
    # Process Parameters
    parser.add_argument('--doc', type=float, default=0.25, dest='doc_in', help='Axial Depth of Cut (in)')
    parser.add_argument('--woc', type=float, default=0.05, dest='woc_in', help='Radial Width of Cut (in)')
    parser.add_argument('--rpm', type=int, default=10000, dest='spindle_speed_rpm', help='Spindle speed (RPM)')
    parser.add_argument('--feed', type=float, default=60.0, dest='feed_rate_in_min', help='Feed rate (in/min)')
    
    # Material and Machine
    parser.add_argument('--uts', type=float, default=45000, dest='material_uts_psi', help='Material Ultimate Tensile Strength (PSI)')
    parser.add_argument('--kr', type=float, default=0.4, dest='radial_force_ratio', help='Radial-to-Tangential force ratio (Kr)')
    parser.add_argument('--eff', type=float, default=0.85, dest='machine_efficiency', help='Machine/spindle efficiency (0.0 to 1.0)')
    parser.add_argument('--pf', type=float, default=0.85, dest='power_factor', help='System Power Factor for current calculation (0.0 to 1.0)')
    
    # Simulation and Plotting Control
    parser.add_argument('--type', type=str, default='climb', choices=['climb', 'conventional'], dest='milling_type', help='Milling type')
    parser.add_argument('--slices', type=int, default=1000, dest='n_slices', help='Number of axial slices for simulation accuracy')
    parser.add_argument('--smooth', type=int, default=7, dest='smoothing_window', help='Size of moving average window to smooth force plots (e.g., 3-15). Use 1 for no smoothing.')
    parser.add_argument('--revolutions', type=int, default=16, dest='num_revolutions', help='Number of revolutions to analyze in FFT for higher frequency resolution.')
    parser.add_argument('--separate-plots', action='store_true', dest='separate_plots', help='Plot each force component on a separate subplot instead of on the same axes.')
    parser.add_argument('--power-in-hp', action='store_true', dest='power_in_hp', help='Report power in Horsepower instead of the default Kilowatts.')
    parser.add_argument('--fft', action='store_true', dest='analyze_fft', help='Plot a frequency analysis (FFT) of the cutting forces.')

    args = parser.parse_args()
    simulation_parameters = vars(args)

    simulation_data = simulate_milling_forces(simulation_parameters)

    stats = simulation_data['stats']
    power_unit = 'hp' if args.power_in_hp else 'kW'
    
    # --- Calculate and Print Process Parameters ---
    sfm = (args.spindle_speed_rpm * args.tool_diameter_in * np.pi) / 12
    mrr = args.woc_in * args.doc_in * args.feed_rate_in_min
    print(f"\n--- Process Parameters ---")
    print(f"SFM:                                {sfm:.0f} ft/min")
    print(f"MRR:                                {mrr:.3f} in^3/min")
    
    print("\n--- Chip Load ---")
    print(f"Feed per Tooth:                     {stats['f_z_in']:.4f} in/tooth")
    print(f"Max Actual Chip Thickness:          {stats['max_actual_h_in']:.4f} in")

    print("\n--- Force & Torque Results ---")
    print(f"Peak Abs. Feed Force (Fx):          {stats['F_peak_x_lbf']:.2f} lbf")
    print(f"Average Abs. Feed Force (Fx):       {stats['F_avg_x_lbf']:.2f} lbf")
    print(f"Peak Abs. Transverse Force (Fy):    {stats['F_peak_y_lbf']:.2f} lbf")
    print(f"Average Abs. Transverse Force (Fy): {stats['F_avg_y_lbf']:.2f} lbf")
    print(f"Peak Abs. Axial Force (Fz):         {stats['F_peak_z_lbf']:.2f} lbf")
    print(f"Average Abs. Axial Force (Fz):      {stats['F_avg_z_lbf']:.2f} lbf")
    print(f"Peak Abs. Torque:                   {stats['T_peak_in_lbf']:.2f} in-lbf")
    print(f"Average Abs. Torque:                {stats['T_avg_in_lbf']:.2f} in-lbf")
    
    print("\n--- Power & Current Results ---")
    print(f"Average Cutting Power:              {stats['P_avg_cutting']:.3f} {power_unit}")
    print(f"Peak Cutting Power:                 {stats['P_peak_cutting']:.3f} {power_unit}")
    print(f"Average Machine Power:              {stats['P_avg_machine']:.3f} {power_unit}")
    print(f"Peak Machine Power:                 {stats['P_peak_machine']:.3f} {power_unit}")
    print(f"Est. RMS Current (120V):            {stats['I_rms_A']:.2f} A")
    
    perform_fft_analysis(simulation_data, simulation_parameters)
    plot_time_domain(simulation_data, simulation_parameters)
    
    plt.show()

if __name__ == "__main__":
    main()
