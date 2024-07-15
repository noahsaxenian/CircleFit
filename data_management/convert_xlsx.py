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
#df = pd.read_excel(file_path, sheet_name=sheet_name, skiprows=232)  # Adjust skiprows as needed
df = pd.read_excel(file_path, sheet_name=sheet_name, skiprows=232)  # Adjust skiprows as needed
#print(df)

data = []
simulated = []

for i in range(16):
    datapoint = df.iloc[:, [12*i, (12*i)+1, (12*i)+2]]
    simulatedpoint = df.iloc[:, [(12*i)+3, (12 * i) + 4, (12 * i) + 5]]
    datapoint.columns = ['freq (Hz)', 'real', 'complex']
    simulatedpoint.columns = ['freq (Hz)', 'real', 'complex']
    data.append(datapoint)
    simulated.append(simulatedpoint)


# # Function to compare DataFrames
# def compare_dataframes(df_list):
#     identical_pairs = []
#     n = len(df_list)
#
#     for i in range(n):
#         for j in range(i + 1, n):
#             if df_list[i].equals(df_list[j]):
#                 identical_pairs.append((i, j))
#
#     return identical_pairs
#
#
# # Get the list of identical DataFrame pairs
# identical_pairs = compare_dataframes(data)
#
# # Print the results
# if identical_pairs:
#     print("Identical DataFrame pairs (by indices):")
#     for pair in identical_pairs:
#         print(f"DataFrame {pair[0]} and DataFrame {pair[1]}")
# else:
#     print("No identical DataFrames found.")



for i in range(len(data)):
    # print(data[i])
    # print(simulated[i])
    # Save the receptance data to a new TSV file
    if i < 4:
        data[i].to_csv(f'C:/Users/noahs/Documents/ceeo/modal stuff/Siemens Plate Test/point{i+1}_data.tsv', sep='\t', index=False)
        simulated[i].to_csv(f'C:/Users/noahs/Documents/ceeo/modal stuff/Siemens Plate Test/point{i + 1}_regenerated.tsv', sep='\t', index=False)
    elif i < 8:
        data[i].to_csv(f'C:/Users/noahs/Documents/ceeo/modal stuff/Siemens Plate Test/point{i+2}_data.tsv', sep='\t', index=False)
        simulated[i].to_csv(f'C:/Users/noahs/Documents/ceeo/modal stuff/Siemens Plate Test/point{i + 2}_regenerated.tsv', sep='\t', index=False)
    elif i < 12:
        data[i].to_csv(f'C:/Users/noahs/Documents/ceeo/modal stuff/Siemens Plate Test/point{i+3}_data.tsv', sep='\t', index=False)
        simulated[i].to_csv(f'C:/Users/noahs/Documents/ceeo/modal stuff/Siemens Plate Test/point{i + 3}_regenerated.tsv', sep='\t', index=False)
    else:
        data[i].to_csv(f'C:/Users/noahs/Documents/ceeo/modal stuff/Siemens Plate Test/point{i+4}_data.tsv', sep='\t', index=False)
        simulated[i].to_csv(f'C:/Users/noahs/Documents/ceeo/modal stuff/Siemens Plate Test/point{i + 4}_regenerated.tsv', sep='\t', index=False)
