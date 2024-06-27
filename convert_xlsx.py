# This script filters a large xlsx file down to the desired rows and columns
# renames headers to work with circle_fit library
# converts mobility data to receptance
# exports as tsv to be easily imported

import pandas as pd
import numpy as np

# Load your FRF data from a CSV file
file_path = 'C:/Users/noahs/Documents/ceeo/modal stuff/Siemens Plate Test/Plate FRFs.xlsx'
sheet_name = 'SI FRFs and synthesized'  # Replace with the actual sheet name

# Read the Excel file from the specified sheet, skip header rows (adjust skiprows as necessary)
df = pd.read_excel(file_path, sheet_name=sheet_name, skiprows=232)  # Adjust skiprows as needed

# Select the desired columns by their index (adjust column indices as necessary)
data = df.iloc[:, [0, 1, 2]]  # Adjust indices to select correct columns

# Rename the columns
data.columns = ['freq (Hz)', 'real', 'complex']

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
result_df.to_csv('C:/Users/noahs/Documents/ceeo/modal stuff/Siemens Plate Test/point1_data_receptance.tsv', sep='\t', index=False)

# Print the result for verification
print(result_df)