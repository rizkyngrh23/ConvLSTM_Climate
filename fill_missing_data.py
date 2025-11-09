# Mengisi data kosong dengan IDW untuk multiple monthly folders

import xarray as xr
import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm
import time
import os
import glob

def idw_interpolation(x, y, values, xi, yi, power=2):
    tree = cKDTree(np.vstack((x, y)).T)
    distances, indices = tree.query(np.vstack((xi, yi)).T, k=len(x))
    
    interpolated_values = np.zeros_like(xi)
    for i in range(len(xi)):
        dist = distances[i]
        vals = values[indices[i]]
        dist = np.where(dist == 0, 1e-10, dist)
        weights = 1 / (dist ** power)
        interpolated_values[i] = np.sum(weights * vals) / np.sum(weights)
    
    return interpolated_values

def process_monthly_folder(folder_path, output_folder):
    """Process all NetCDF files in a monthly folder"""
    print(f"\n=== Processing folder: {os.path.basename(folder_path)} ===")
    
    # Find all NetCDF files in the folder
    nc_files = glob.glob(os.path.join(folder_path, "*.nc"))
    
    if not nc_files:
        print(f"No NetCDF files found in {folder_path}")
        return
    
    print(f"Found {len(nc_files)} NetCDF files")
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    for nc_file in nc_files:
        print(f"\nProcessing file: {os.path.basename(nc_file)}")
        
        try:
            # Load dataset
            dataset = xr.open_dataset(nc_file, decode_times=False)
            
            # Get the first data variable (assuming it's precipitation or similar)
            data_vars = list(dataset.data_vars.keys())
            if not data_vars:
                print(f"No data variables found in {nc_file}")
                continue
                
            data_var = data_vars[0]
            data = dataset[data_var]
            
            print(f"  - Data variable: {data_var}")
            print(f"  - Dimensi: {data.dims}")
            print(f"  - Shape: {data.shape}")
            print(f"  - Jumlah NaN sebelum interpolasi: {data.isnull().sum().item():,}")
            
            X = data['X'].values
            Y = data['Y'].values
            T = data['T'].values
            
            values = data.values
            interpolated_values = np.full_like(values, np.nan)
            
            print(f"  - Memulai interpolasi IDW untuk {len(T)} time steps...")
            start_time = time.time()
            
            # Progress bar untuk loop time steps
            for t in tqdm(range(values.shape[0]), desc=f"Interpolasi {os.path.basename(nc_file)}", unit="time step"):
                grid_x, grid_y = np.meshgrid(X, Y)
                points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
                values_at_t = values[t].ravel()
                mask = ~np.isnan(values_at_t)
                
                if np.sum(mask) == 0:
                    # All values are NaN, skip this time step
                    continue
                    
                known_points = points[mask]
                known_values = values_at_t[mask]
                
                interpolated_at_t = idw_interpolation(
                    known_points[:, 0], known_points[:, 1],
                    known_values,
                    points[:, 0], points[:, 1]
                )
                interpolated_values[t] = interpolated_at_t.reshape(len(Y), len(X))
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            print(f"  - Interpolasi selesai dalam {processing_time:.2f} detik")
            
            # Create interpolated dataset
            interpolated_data = xr.DataArray(interpolated_values, coords=[T, Y, X], dims=['T', 'Y', 'X'])
            print(f"  - Jumlah NaN setelah interpolasi IDW: {interpolated_data.isnull().sum().item():,}")
            
            # Save result
            output_filename = f"filled_{os.path.basename(nc_file)}"
            output_path = os.path.join(output_folder, output_filename)
            print(f"  - Menyimpan hasil ke: {output_path}")
            interpolated_data.to_netcdf(output_path)
            
            print(f"  - File berhasil diproses: {output_filename}")
            
        except Exception as e:
            print(f"Error processing {nc_file}: {str(e)}")
            continue

print("=== MENGISI DATA KOSONG DENGAN IDW UNTUK MULTIPLE MONTHLY FOLDERS ===")

# Define input and output paths
input_folders = [
    r"D:\Kuliah\Skripsi\Data_baru\Output\AccLead\dec",
    r"D:\Kuliah\Skripsi\Data_baru\Output\AccLead\feb", 
    r"D:\Kuliah\Skripsi\Data_baru\Output\AccLead\jan",
    r"D:\Kuliah\Skripsi\Data_baru\Output\AccLead\mar",
    r"D:\Kuliah\Skripsi\Data_baru\Output\AccLead\nov"
]

output_base = r"D:\Kuliah\Skripsi\Data_baru\Output\AccLead\filled"

print("Input folders:")
for folder in input_folders:
    print(f"  - {folder}")

print(f"\nOutput base folder: {output_base}")

# Process each monthly folder
for folder in input_folders:
    if os.path.exists(folder):
        month_name = os.path.basename(folder)
        output_folder = os.path.join(output_base, month_name)
        process_monthly_folder(folder, output_folder)
    else:
        print(f"Folder tidak ditemukan: {folder}")

