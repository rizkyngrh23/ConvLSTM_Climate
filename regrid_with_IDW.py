import xarray as xr
import numpy as np
from scipy.spatial import cKDTree
import os

# --- Konfigurasi ---
months = ['nov', 'dec', 'jan', 'feb', 'mar']
model_names = ['canv2', 'cancm', 'ic3', 'nemo']

base_input_path = 'D:/Kuliah/Skripsi/Data_baru/Output/AccLead'
base_output_path = 'D:/Kuliah/Skripsi/Data_baru/Output/Regrid'

# --- DEFINISI TUGAS (DIPERBAIKI) ---
# Tasks ini sekarang cocok dengan output 3D dari skrip AccLead
tasks = [
    {
        'name': 'Prec Standar',
        'input_prefix': 'acc',
        'output_prefix': 'regrid',
        'var_names': ['prec', 'precipitation'] # Mencari 'prec' di file 'acc_...'
    },
    {
        'name': 'SST Nino', # Diganti dari 'Prec Nino'
        'input_prefix': 'acc_nino',
        'output_prefix': 'regrid_nino',
        'var_names': ['sst', 'tos'] # DIPERBAIKI: Mencari 'sst' di file 'acc_nino_...'
    }
    # Task 'SST' ketiga dihapus karena redundan.
]

# --- Fungsi Interpolasi (Tidak berubah) ---
def idw_interpolation(lons, lats, values, lon_new, lat_new, power=2):
    """Melakukan interpolasi IDW pada grid 2D."""
    grid_lat, grid_lon = np.meshgrid(lat_new, lon_new, indexing='ij')
    grid_points = np.vstack((grid_lat.flatten(), grid_lon.flatten())).T
    
    points = np.vstack((lats.flatten(), lons.flatten())).T
    values = values.flatten()
    
    tree = cKDTree(points)
    
    distances, indices = tree.query(grid_points, k=len(points), p=2)
    distances = np.maximum(distances, 1e-10)
    
    weights = 1.0 / distances**power
    
    weighted_values = np.sum(weights * values[indices], axis=1) / np.sum(weights, axis=1)
    
    return weighted_values.reshape(len(lat_new), len(lon_new))

# --- Proses Utama (Logika 3D sudah benar) ---
os.makedirs(base_output_path, exist_ok=True)
print(f"Memulai regridding, output akan disimpan di: {base_output_path}")

for month in months:
    print(f"\n=== Memproses bulan: {month.upper()} ===")
    
    month_output_path = os.path.join(base_output_path, month)
    os.makedirs(month_output_path, exist_ok=True)
    
    for model in model_names:
        print(f"  --- Model: {model.upper()} ---")
        
        for task in tasks:
            print(f"    * Memproses Tipe: {task['name']}")
            
            input_file = f'{base_input_path}/{month}/{task["input_prefix"]}_{month}_{model}.nc'
            output_file = f'{month_output_path}/{task["output_prefix"]}_{month}_{model}.nc'

            try:
                dataset = xr.open_dataset(input_file, decode_times=False)
                
                var_name = None
                for v in task['var_names']:
                    if v in dataset.data_vars:
                        var_name = v
                        break
                
                if var_name is None:
                    var_name = list(dataset.data_vars.keys())[0] # Fallback
                    print(f'      - Peringatan: Var {task["var_names"]} tidak ditemukan, menggunakan var default: {var_name}')
                
                print(f'      - Variabel ditemukan: {var_name}')
                data = dataset[var_name]
                
                # Logika ini sudah benar, karena inputnya (dari AccLead) adalah 3D (L, Y, X)
                lons = dataset.X.values
                lats = dataset.Y.values
                leads = dataset.L.values
                values = data.values

                lon_min, lon_max = lons.min(), lons.max()
                lat_min, lat_max = lats.min(), lats.max()
                
                lon_new = np.arange(lon_min, lon_max + 0.05, 0.05)
                lat_new = np.arange(lat_min, lat_max + 0.05, 0.05)
                
                lons_grid, lats_grid = np.meshgrid(lons, lats)
                
                grid_z_new = np.empty((len(leads), len(lat_new), len(lon_new)))
                grid_z_new.fill(np.nan)

                # Loop 3D (per lead) sudah benar
                for i, lead in enumerate(leads):
                    values_lead = values[i, :, :]
                    
                    grid_z = idw_interpolation(lons_grid, lats_grid, values_lead, lon_new, lat_new)
                    grid_z_new[i, :, :] = grid_z

                    if np.isnan(grid_z).any():
                        print(f'      - Peringatan: NaN ditemukan di data interpolasi untuk {month}_{model} (Tipe: {task["name"]}) pada lead {lead}')

                # Siapkan koordinat baru
                new_coords = {'L': leads, 'Y': lat_new, 'X': lon_new}

                # Cek dan pertahankan koordinat 'S' (jika ada dan bukan dimensi)
                if 'S' in dataset.coords and 'S' not in data.dims:
                    print("      - Koordinat scalar 'S' ditemukan dan akan dipertahankan.")
                    new_coords['S'] = dataset.S.values

                ds_new = xr.Dataset(
                    {
                        var_name: (['L', 'Y', 'X'], grid_z_new) 
                    },
                    coords=new_coords
                )
                
                ds_new.to_netcdf(output_file)
                
                print(f'      - Dimensi asli: {data.dims}')
                print(f'      - Dimensi setelah regrid: {ds_new[var_name].dims}')
                print(f'      - Shape asli: {data.shape}')
                print(f'      - Shape setelah regrid: {ds_new[var_name].shape}')
                print(f'      - Disimpan sebagai: {output_file}')
                print(f'      - Nilai (min/max/mean): '
                      f'{ds_new[var_name].min().values:.2f} / '
                      f'{ds_new[var_name].max().values:.2f} / '
                      f'{ds_new[var_name].mean().values:.2f}')
                
                dataset.close()
                
            except FileNotFoundError:
                print(f'      - File tidak ditemukan: {input_file}')
            except Exception as e:
                print(f'      - Error: {str(e)}')
    
    print(f"=== Selesai memproses bulan: {month.upper()} ===\n")

print("Semua file telah diproses!")
print(f"Output disimpan di: {base_output_path}")