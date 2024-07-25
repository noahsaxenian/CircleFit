import pandas as pd
import numpy as np

# Read the mobility data from the TSV file
file_path = 'C:/Users/noahs/Documents/ceeo/modal stuff/Code/data/Plate/Plate 04/csv/Plate 04 H_004_trf.tsv'  # Replace with your file path
data = pd.read_csv(file_path, delimiter='\t')

# Extract frequency, real, and imaginary parts from the dataframe
frequencies = data['freq (Hz)'].values
real_parts = data['real'].values
imaginary_parts = data['complex'].values

# Convert frequency to angular frequency (rad/s)
omega = 2 * np.pi * frequencies

# Combine real and imaginary parts to form complex mobility data
mobility = real_parts + 1j * imaginary_parts

# Calculate receptance
receptance = mobility / (1j * omega)

# Create a new dataframe to store the results
result_df = pd.DataFrame({
    'freq (Hz)': frequencies,
    'real': receptance.real,
    'complex': receptance.imag
})

# Save the receptance data to a new TSV file
result_df.to_csv('C:/Users/noahs/Documents/ceeo/modal stuff/Code/data/Plate/Plate 04//csv/Plate 04 H_004_receptance.tsv', sep='\t', index=False)

# Print the result for verification
print(data)
print(result_df)
