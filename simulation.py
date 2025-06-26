import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# plot 3 V 100 khz
df = pd.read_csv('3 V 100 khz.csv')
print(df.head())  # Display the first few rows of the dataframe
plt.figure(figsize=(10, 5))
df_temp = df[(df['time_s'] > 0.005) & (df['time_s'] < 0.006)]  # Filter the dataframe for time less than 0.1 seconds
plt.plot(df_temp['time_s'], df_temp['Current_A'], label='3 V 100 kHz', marker='o', markersize=1, linestyle='-', color='blue')
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.title('3 V 100 kHz Signal')
plt.legend()
plt.grid()
plt.savefig('3_V_100_kHz.png')
plt.show()

# amplitude extraction
df['Amplitude'] = df['Current_A'].abs()  # Calculate the absolute value of the current to get amplitude
# Find the maximum amplitude
max_amplitude = df['Amplitude'].max()
print(f'Maximum Amplitude: {max_amplitude}')