# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 09:47:06 2025

@author: islam9
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


def eval_Li_evap_at_T_Cel(temperature):
    """Calculate lithium evaporation flux at a given temperature in Celsius."""
    a1 = 5.055
    b1 = -8023.0
    xm1 = 6.939
    tempK = temperature + 273.15

    if np.any(tempK <= 0):
        raise ValueError("Temperature must be above absolute zero (-273.15°C).")

    vpres1 = 760 * 10 ** (a1 + b1 / tempK)  # Vapor pressure
    sqrt_argument = xm1 * tempK

    if np.any(sqrt_argument <= 0):
        raise ValueError("Invalid value for sqrt: xm1 * tempK has non-positive values.")

    fluxEvap = 1e4 * 3.513e22 * vpres1 / np.sqrt(sqrt_argument)  # Evaporation flux
    return fluxEvap


def count_files_in_folder(folder_path):
    return len([file for file in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file))])


folders = {
     "nx_P1": r"./PePi1MW/C_Li_omp",
     "nx_P2": r"./PePi3.4MW/C_Li_omp",
     "nx_P3": r"./PePi4.6MW/C_Li_omp",   
}


file_counts = {key: count_files_in_folder(path) for key, path in folders.items()}

nx_P1 = file_counts["nx_P1"]
nx_P2 = file_counts["nx_P2"]
nx_P3 = file_counts["nx_P3"]

datasets = [
      
    {
     'path': r'./PePi1MW',
     'nx': nx_P1,
     'dt': 10e-3,
    'label_tsurf': 'P1MW '
 },
    
    {
     'path': r'./PeP3.4MW',
     'nx': nx_P2,
     'dt': 10e-3,
     'label_tsurf': 'Drifts : b0=60'
 },
    
  {
     'path': r'./PePi4.6MW',
     'nx': nx_P3,
     'dt': 10e-3,
     'label_tsurf': 'Drifts : b0=8'
 }

]


def process_dataset(data_path, nx, dt):

  
    max_value_tsurf = []
    max_q = []
    evap_flux_max = []
    max_q_Li_list = []
    C_Li_omp = []
    Te = []
    n_Li3 = []
    ne = []

    # Define directories for the input data
    q_perp_dir = os.path.join(data_path, 'q_perp')
    T_surf_dir = os.path.join(data_path, 'Tsurf_Li')
    q_Li_dir = os.path.join(data_path, 'q_Li_surface')
    C_Li_dir = os.path.join(data_path, 'C_Li_omp')
    n_Li3_dir = os.path.join(data_path, 'n_Li3')
    ne_dir = os.path.join(data_path, 'n_e')
    Te_dir = os.path.join(data_path, 'T_e')

    # Loop through the files
    for i in range(1, nx): 
        # Construct file paths
        filename_tsurf = os.path.join(T_surf_dir, f'T_surfit_{i}.0.csv')
        filename_qsurf = os.path.join(q_perp_dir, f'q_perpit_{i}.0.csv')
        filename_qsurf_Li = os.path.join(q_Li_dir, f'q_Li_surface_{i}.0.csv')
        filename_C_Li = os.path.join(C_Li_dir, f'CLi_prof{i}.0.csv')
        filename_n_Li = os.path.join(n_Li3_dir, f'n_Li3_{i}.0.csv')
        filename_T_e = os.path.join(Te_dir, f'T_e_{i}.0.csv')
        filename_n_e = os.path.join(ne_dir, f'n_e_{i}.0.csv.npy')

        # Initialize variables for this iteration
        max_tsurf = np.nan
        max_q_i = np.nan
        evap_flux = np.nan
        max_q_Li_i = np.nan
        C_Li_i = np.nan
        ne_i = np.nan
        Te_i = np.nan

        try:
            # Process Tsurf file
            df_tsurf = pd.read_csv(filename_tsurf)
            max_tsurf = np.max(df_tsurf.values)
        except FileNotFoundError:
            max_tsurf = np.nan  
        max_value_tsurf.append(max_tsurf)

        try:
            # Process q_perp file
            df_qsurf = pd.read_csv(filename_qsurf)
            max_q_i = np.max(df_qsurf.values)

            # Process q_Li_surface file
            df_qsurf_Li = pd.read_csv(filename_qsurf_Li)
            max_q_Li_i = np.max(df_qsurf_Li.values)

            # Process C_Li_omp file
            sep = 8
            ixmp = 67
            df_C_Li = pd.read_csv(filename_C_Li)
            C_Li_i = df_C_Li.values[sep]

                   # Process T_e file
            T_e = np.loadtxt(filename_T_e)
            Te_i = T_e[ixmp, sep]
            
            
            nLi = np.loadtxt(filename_n_Li)
            n_Li_3i  = nLi[ixmp, sep]
            
            n_e = np.load(filename_n_e)
            ne_i  = n_e[ixmp, sep]
            
        except FileNotFoundError:
            max_q_i = np.nan
            max_q_Li_i = np.nan
            C_Li_i = np.nan
            ne_i = np.nan
            Te_i = np.nan

        # Append results to lists
        max_q.append(max_q_i)
        max_q_Li_list.append(max_q_Li_i)
        C_Li_omp.append(C_Li_i)

        Te.append(Te_i)
        n_Li3.append(n_Li_3i)
        ne.append(ne_i)

    # Helper function to replace NaN values with linear interpolation
    def replace_with_linear_interpolation(arr):
        arr = pd.Series(arr)
        # Perform linear interpolation in both directions
        arr_interpolated = arr.interpolate(method='linear', limit_direction='both')
        # Ensure that the interpolation doesn't leave any NaN at the ends
        arr_interpolated = arr_interpolated.fillna(method='bfill').fillna(method='ffill')
        # Return as a numpy array
        return arr_interpolated.to_numpy()

    # Interpolate missing values for all variables
    max_value_tsurf = replace_with_linear_interpolation(max_value_tsurf)
    max_q = replace_with_linear_interpolation(max_q)
    max_q_Li_list = replace_with_linear_interpolation(max_q_Li_list)
    C_Li_omp = replace_with_linear_interpolation(C_Li_omp)
    nLi3 = replace_with_linear_interpolation(n_Li3)
    Te = replace_with_linear_interpolation(Te)
    ne = replace_with_linear_interpolation(ne)

    # Calculate evaporation flux
    for max_tsurf in max_value_tsurf:
        if not np.isnan(max_tsurf):
            try:
                evap_flux = eval_Li_evap_at_T_Cel(max_tsurf)  # Ensure this function is defined
            except Exception as e:
                print(f"Error calculating evaporation flux: {e}. Skipping.")
        else:
            evap_flux = np.nan
        evap_flux_max.append(evap_flux)

    # Interpolate evaporation flux
    evap_flux_max = replace_with_linear_interpolation(evap_flux_max)

    # Final calculations
    max_q = np.array(max_q)
    evap_flux_max = np.array(evap_flux_max)
    q_surface = max_q - 2.26e-19 * evap_flux_max

    # Generate time axis
    time_axis = dt * np.arange(1, len(max_q) + 1)

    return max_value_tsurf, max_q, evap_flux_max, q_surface, time_axis, max_q_Li_list, C_Li_omp, nLi3, Te, ne



colors = ['r', 'g', 'k', 'b', 'm', 'y', 'c', 'purple']  # More colors

# Plot 1: Time evolution of max_q_Li, max_value_tsurf, and C_Li_omp
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6, 6), sharex=True)

xset = 1
for idx, dataset in enumerate(datasets):
    max_value_tsurf, max_q, evap_flux_max, q_surface, time_axis, max_q_Li, C_Li_omp, nLi3, Te,ne = process_dataset(
        dataset['path'], dataset['nx'], dataset['dt']
    )
    ax1.plot(time_axis, np.array(max_q_Li) / 1e6, linestyle='-', linewidth=2, label=f'{dataset["label_tsurf"]}', color=colors[idx % len(colors)])
    ax2.plot(time_axis, max_value_tsurf, linestyle='-', linewidth=2, label=f'{dataset["label_tsurf"]}', color=colors[idx % len(colors)])
    ax3.plot(time_axis, C_Li_omp * 100, linestyle='-', linewidth=2, label=f'{dataset["label_tsurf"]}', color=colors[idx % len(colors)])

ax1.set_ylabel('q$_{s}^{max}$ (MW/m$^2$)', fontsize=18)
ax1.set_xlim([0, xset])
ax1.set_ylim([0, 3])
ax1.legend(loc='best', fontsize=12, ncol=2)
ax1.grid(True)
ax1.tick_params(axis='both', labelsize=14)

ax2.set_ylabel("T$_{surf}^{max}$ ($^\circ$C)", fontsize=18)
ax2.set_ylim([0, 750])
ax2.set_xlim([0, xset])
ax2.grid(True)
ax2.tick_params(axis='both', labelsize=14)

ax3.set_ylabel("C$_{Li-sep}^{omp}$ (%)", fontsize=18)
ax3.set_ylim([0, 6])
ax3.set_xlim([0, xset])
ax3.set_xlabel('t$_{sim}$ (s)', fontsize=18)
ax3.grid(True)
ax3.tick_params(axis='both', labelsize=14)

plt.tight_layout()
plt.savefig('qsurf_T_surf_CLi_omp.png', dpi=300)
plt.show()

# Plot 2: Time evolution of max_q, max_value_tsurf, and C_Li_omp with special handling for idx == 3
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6, 6), sharex=True)

xset = 1


for idx, dataset in enumerate(datasets):
    color = colors[idx % len(colors)]  # Cycle through the colors list

    max_value_tsurf, max_q, evap_flux_max, q_surface, time_axis, max_q_Li, C_Li_omp, nLi3, Te,ne = process_dataset(
        dataset['path'], dataset['nx'], dataset['dt']
    )


    ax1.plot(time_axis, np.array(nLi3), linestyle='-', linewidth=2, label=f'{dataset["label_tsurf"]}', color=color)
    ax2.plot(time_axis, Te, linestyle='-', linewidth=2, label=f'{dataset["label_tsurf"]}', color=color)
    ax3.plot(time_axis, ne, linestyle='-', linewidth=2, label=f'{dataset["label_tsurf"]}', color=color)

ax1.set_ylabel('n$_{Li3+}^{omp}$ (m$^{-3}$)', fontsize=18)
ax1.set_xlim([0, xset])
ax1.set_ylim([0, 3e18])
ax1.legend(loc='best', fontsize=12, ncol=1)
ax1.grid(True)
ax1.tick_params(axis='both', labelsize=14)

ax2.set_ylabel("T$_{e-sep}^{omp}$ (eV)", fontsize=18)
ax2.set_ylim([0, 120])
ax2.set_xlim([0, xset])
ax2.grid(True)
ax2.tick_params(axis='both', labelsize=14)

ax3.set_ylabel('n$_{e-sep}^{omp}$ (m$^{-3}$)', fontsize=18)
ax3.set_ylim([0, 5e19])
ax3.set_xlim([0, xset])
ax3.set_xlabel('t$_{sim}$ (s)', fontsize=18)
ax3.grid(True)
ax3.tick_params(axis='both', labelsize=14)

plt.tight_layout()
plt.savefig('nLi3_Te_ne_omp.png', dpi=300)
plt.show()

# Plot 3: Time evolution of max_q_Li, max_value_tsurf, and C_Li_omp with dashed linestyle for C_Li_omp
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6, 6), sharex=True)

xset = 1
user_defined_start_time = 0

for idx, dataset in enumerate(datasets):
    color = colors[idx % len(colors)]  # Cycle through the colors list

    max_value_tsurf, max_q, evap_flux_max, q_surface, time_axis, max_q_Li, C_Li_omp, nLi3, Te, ne = process_dataset(
        dataset['path'], dataset['nx'], dataset['dt']
    )

    if idx == 3:
        time_axis = time_axis + user_defined_start_time
        color = 'k'  # Override the color to black

    ax1.plot(time_axis, np.array(max_q_Li) / 1e6, linestyle='-', linewidth=2, label=f'{dataset["label_tsurf"]}', color=color)
    ax2.plot(time_axis, max_value_tsurf, linestyle='-', linewidth=2, label=f'{dataset["label_tsurf"]}', color=color)
    ax3.plot(time_axis, C_Li_omp * 100, linestyle='--', linewidth=2, label=f'{dataset["label_tsurf"]}', color=color)

ax1.set_ylabel('q$_{s}^{max}$ (MW/m$^2$)', fontsize=18)
ax1.set_xlim([0, xset])
ax1.set_ylim([0, 8])
ax1.legend(loc='best', fontsize=12, ncol=2)
ax1.grid(True)
ax1.tick_params(axis='both', labelsize=14)

ax2.set_ylabel("T$_{surf}^{max}$ ($^\circ$C)", fontsize=18)
ax2.set_ylim([0, 750])
ax2.set_xlim([0, xset])
ax2.grid(True)
ax2.tick_params(axis='both', labelsize=14)

ax3.set_ylabel("C$_{Li-sep}^{omp}$ (%)", fontsize=18)
ax3.set_ylim([0, 3])
ax3.set_xlim([0, xset])
ax3.set_xlabel('t$_{sim}$ (s)', fontsize=18)
ax3.grid(True)
ax3.tick_params(axis='both', labelsize=14)

plt.tight_layout()
plt.show()

# Plot 4: Scatter plot of max_value_tsurf vs C_Li_omp
for idx, dataset in enumerate(datasets):
    max_value_tsurf, max_q, evap_flux_max, q_surface, time_axis, max_q_Li, C_Li_omp, nLi3, Te, ne = process_dataset(
        dataset['path'], dataset['nx'], dataset['dt']
    )

    color = 'k' if idx == 3 else colors[idx % len(colors)]  # Black for idx == 3
    plt.plot(max_value_tsurf, C_Li_omp * 100, linestyle='-', linewidth=2, label=f'{dataset["label_tsurf"]}', color=color)

plt.xlabel("T$_{surf}^{max}$ ($^\circ$C)", fontsize=18)
plt.ylabel("C$_{Li-sep}^{omp}$ (%)", fontsize=18)
plt.legend(fontsize=14)
plt.axhline(3, color='black', linestyle=':', linewidth=2, label='y = 3')  # Reference line at y=3
plt.ylim([0, 6])
plt.xlim([0, 700])

plt.grid(True)
plt.tick_params(axis='both', labelsize=14)
plt.tight_layout()
plt.savefig('T_surf_CLi_omp.png', dpi=300)
plt.show()
