import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit



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

def calculate_current(t,voltage_amplitude, angular_frequency, R, L1, L2, C1, C2):
    magnitude, phase = calculate_z(angular_frequency, R, L1, L2, C1, C2)
    return  voltage_amplitude*np.sin(angular_frequency*t + phase) / magnitude

# plot 3 V 100 khz
df = pd.read_csv('3 V 10 khz.csv')
print(df.head())  # Display the first few rows of the dataframe
voltage_amplitude = 3  # Amplitude of the voltage signal
angular_frequency = 2 * np.pi * 10000  # Angular frequency for 10 khz

df_temp = df[(df['time_s'] > 0.005) & (df['time_s'] < 0.006)]  # Filter the dataframe for time less than 0.1 seconds

# fit the data
initial_guess = [voltage_amplitude, angular_frequency, 1, 1/94772, 1e-3, 1e-6, 1e-6]  # Initial guess for the parameters
params, pcov = curve_fit(calculate_current, df_temp['time_s'], df_temp['Current_A'], p0=initial_guess)
print("Fitted parameters:")
print(f"voltage amplitude: {params[0]} +- {np.sqrt(pcov[0, 0])}")
print(f"angular frequency: {params[1]} +- {np.sqrt(pcov[1, 1])}")
print(f"R: {params[2]} +- {np.sqrt(pcov[2, 2])}")
print(f"L1: {params[3]} +- {np.sqrt(pcov[3, 3])}")
print(f"L2: {params[4]} +- {np.sqrt(pcov[4, 4])}")
print(f"C1: {params[5]} +- {np.sqrt(pcov[5, 5])}")
print(f"C2: {params[6]} +- {np.sqrt(pcov[6, 6])}")

# Extract the fitted parameters
y_fit = calculate_current(df_temp['time_s'], *params)
# Plot the original data and the fitted curve
plt.figure(figsize=(10, 6))
plt.plot(df_temp['time_s'], df_temp['Current_A'], label='3 V 10 kHz', marker='o', markersize=1, linestyle='-', color='blue')
plt.plot(df_temp['time_s'], y_fit, label='Fitted Curve', linestyle='--', color='red')
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.title('3 V 100 kHz Signal')
plt.legend()
plt.grid()
plt.savefig('3_V_100_kHz.png')
plt.show()
