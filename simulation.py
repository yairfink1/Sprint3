import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

L1 = 1.01e-5  # Inductance of L1 in H
C1_pluss_C2 = 4.00e-6  # C1 and C2 in parallel in F


filename = '0.5 V 100 khz.csv'
df = pd.read_csv(filename)
print(filename)  # Display the first few rows of the dataframe
voltage_amplitude = float(filename.split()[0])  # Amplitude of the voltage signal
angular_frequency = 2 * np.pi * float(filename.split()[2])*1e3  # Angular frequency for 100 kHz

def calculate_z(angular_frequency, R, L1, L2, C1, C2):
    zl1 = angular_frequency * L1 * 1j  # Impedance of L1
    zl2 = angular_frequency * L2 * 1j  # Impedance of L
    zc1 = 1 / (angular_frequency * C1 * 1j)  # Impedance of C1
    zc2 = 1 / (angular_frequency * C2 * 1j)  # Impedance of C2
    # Parallel
    zparrel = 1 / (1 / zc1 + 1 / (zc2+ zl2))  # Impedance of C1 and C2 in parallel with L2
    z_total = R + zl1 +  zparrel  # Total impedance
    # polar form
    magnitude = np.abs(z_total)
    phase = np.angle(z_total)  # Phase angle in radians
    return magnitude, phase    

plt.figure(figsize=(10, 6))
angular_frequency = 2 * np.pi * np.linspace(0, 100e3, 1000)  # Angular frequency for 100 kHz
mag, _ = calculate_z(angular_frequency, 1, L1, 1e-5, 1e-6, C1_pluss_C2 - 1e-6)  # Example values for R, L2, C1, C2
plt.plot(angular_frequency / (2 * np.pi), 1/mag, label='Magnitude', color='blue')
plt.xlabel('Frequency (Hz)')
plt.show()


def calculate_z_known_C1_plus_C2(R, L2, C1):
    C2 = C1_pluss_C2 - C1  # Calculate C2 from the known value of C1 + C2
    zl1 = angular_frequency * L1 * 1j  # Impedance of L1
    zl2 = angular_frequency * L2 * 1j  # Impedance of L
    zc1 = 1 / (angular_frequency * C1 * 1j)  # Impedance of C1
    zc2 = 1 / (angular_frequency * C2 * 1j)  # Impedance of C2
    # Parallel
    zparrel = 1 / (1 / zc1 + 1 / (zc2+ zl2))  # Impedance of C1 and C2 in parallel with L2
    z_total = R + zl1 +  zparrel  # Total impedance
    # polar form
    magnitude = np.abs(z_total)
    phase = np.angle(z_total)  # Phase angle in radians
    return magnitude, phase

def calculate_z_known_all_but_L2(L2):
    R = 1  # Resistance in ohms
    C1 = 1e-6  # Capacitance of C1 in F
    C2 = C1_pluss_C2 - C1  # Calculate C2 from the known value of C1 + C2
    zl1 = angular_frequency * L1 * 1j  # Impedance of L1
    zl2 = angular_frequency * L2 * 1j  # Impedance of L
    zc1 = 1 / (angular_frequency * C1 * 1j)  # Impedance of C1
    zc2 = 1 / (angular_frequency * C2 * 1j)  # Impedance of C2
    # Parallel
    zparrel = 1 / (1 / zc1 + 1 / (zc2+ zl2))  # Impedance of C1 and C2 in parallel with L2
    z_total = R + zl1 +  zparrel  # Total impedance
    # polar form
    magnitude = np.abs(z_total)
    phase = np.angle(z_total)  # Phase angle in radians
    return magnitude, phase     

def calculate_current(t,voltage_amplitude, angular_frequency, R, L1, L2, C1, C2):
    magnitude, phase = calculate_z(angular_frequency, R, L1, L2, C1, C2)
    return  voltage_amplitude*np.sin(angular_frequency*t + phase) / magnitude

def calculate_current_known_C1_plus_C2(t, R, L2, C1):
    magnitude, phase = calculate_z_known_C1_plus_C2(R, L2, C1)
    return voltage_amplitude * np.sin(angular_frequency * t + phase) / magnitude

def calculate_current_known_all_but_L2(t, L2):
    magnitude, phase = calculate_z_known_all_but_L2(L2)
    return voltage_amplitude * np.sin(angular_frequency * t + phase) / magnitude

df_temp = df[(df['time_s'] > 0.005) & (df['time_s'] < 0.006)]  # Filter the dataframe for time less than 0.1 seconds

# fit the data
R_guess = 1.8  # Initial guess for R
L1_guess = L1  # Initial guess for L1
L2_guess = 1e-3  # Initial guess for L2
C1_guess = 1e-6  # Initial guess for C1
C2_guess = 3e-6  # Initial guess for C2


fit_based_known = True
fit_all_but_L2 = True  # If True, fit all parameters except L2
if fit_all_but_L2:
    # Fit the data using curve_fit with known C1 + C2
    initial_guess = [L2_guess]  # Initial guess for L2
    params, pcov = curve_fit(calculate_current_known_all_but_L2, df_temp['time_s'], df_temp['Current_A'], p0=initial_guess, maxfev=1000000)
    print("Fitted parameters (known all but L2):")
    print(f"L2: {params[0]} +- {np.sqrt(pcov[0, 0])}")
    y_fit = calculate_current_known_all_but_L2(df_temp['time_s'], *params)
    y_guess = calculate_current_known_all_but_L2(df_temp['time_s'], *initial_guess)
    # calculate R^2
    residuals = df_temp['Current_A'] - y_fit
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((df_temp['Current_A'] - np.mean(df_temp['Current_A']))**2)
    r_squared = 1 - (ss_res / ss_tot)
    print(f"R^2: {r_squared}")

# if fit_based_known:
#     # Fit the data using curve_fit with known C1 + C2
#     initial_guess = [R_guess, L2_guess, C1_guess]  # Initial guess for the parameters
#     params, pcov = curve_fit(calculate_current_known_C1_plus_C2, df_temp['time_s'], df_temp['Current_A'], p0=initial_guess, maxfev=1000000)
#     print("Fitted parameters (known C1 + C2):")
#     print(f"R: {params[0]} +- {np.sqrt(pcov[0, 0])}")
#     print(f"L2: {params[1]} +- {np.sqrt(pcov[1, 1])}")
#     print(f"C1: {params[2]} +- {np.sqrt(pcov[2, 2])}")
#     print(f"C2: {C1_pluss_C2 - params[2]} +- {np.sqrt(pcov[2, 2])}")  # C2 is derived from C1 + C2
#     y_fit = calculate_current_known_C1_plus_C2(df_temp['time_s'], *params)
#     y_guess = calculate_current_known_C1_plus_C2(df_temp['time_s'], *initial_guess)
#     # calculate R^2
#     residuals = df_temp['Current_A'] - y_fit
#     ss_res = np.sum(residuals**2)
#     ss_tot = np.sum((df_temp['Current_A'] - np.mean(df_temp['Current_A']))**2)
#     r_squared = 1 - (ss_res / ss_tot)
#     print(f"R^2: {r_squared}")



else:
    initial_guess = [voltage_amplitude, angular_frequency, R_guess, L1_guess, L2_guess, C1_guess, C2_guess]  # Initial guess for the parameters
    params, pcov = curve_fit(calculate_current, df_temp['time_s'], df_temp['Current_A'], p0=initial_guess, maxfev=1000000)
    print("Fitted parameters:")
    print(f"voltage amplitude: {params[0]} +- {np.sqrt(pcov[0, 0])}")
    print(f"angular frequency: {params[1]} +- {np.sqrt(pcov[1, 1])}")
    print(f"R: {params[2]} +- {np.sqrt(pcov[2, 2])}")
    print(f"L1: {params[3]} +- {np.sqrt(pcov[3, 3])}")
    print(f"L2: {params[4]} +- {np.sqrt(pcov[4, 4])}")
    print(f"C1: {params[5]} +- {np.sqrt(pcov[5, 5])}")
    print(f"C2: {params[6]} +- {np.sqrt(pcov[6, 6])}")
    y_fit = calculate_current(df_temp['time_s'], *params)
    y_guess = calculate_current(df_temp['time_s'], *initial_guess)
    # calculate R^2
    residuals = df_temp['Current_A'] - y_fit
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((df_temp['Current_A'] - np.mean(df_temp['Current_A']))**2)
    r_squared = 1 - (ss_res / ss_tot)
    print(f"R^2: {r_squared}")

# initial guess
# Plot the original data and the fitted curve
plt.figure(figsize=(10, 6))
plt.plot(df_temp['time_s'], df_temp['Current_A'], label='3 V 10 kHz', marker='o', markersize=1, linestyle='-', color='blue')
plt.plot(df_temp['time_s'], y_fit, label='Fitted Curve', linestyle='--', color='red')
plt.plot(df_temp['time_s'], y_guess, label='Initial Guess', linestyle='--', color='green')
plt.xlabel('Time (s)')
plt.ylabel('I (A)')
plt.title(filename)
plt.legend()
plt.grid()
plt.savefig(f'{filename[:-4]}.png')
plt.show()
