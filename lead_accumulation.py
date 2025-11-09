import xarray as xr
import os

months = ['nov', 'dec', 'jan', 'feb', 'mar']

models = ['canv2', 'cancm', 'ic3', 'nemo']

# Hanya ada satu path input data mentah
base_input_path = 'D:/Kuliah/Skripsi/Data_baru'
# Path output untuk skrip ini
base_output_path = 'D:/Kuliah/Skripsi/Data_baru/Output/AccLead'

os.makedirs(base_output_path, exist_ok=True)

# --- DEFINISI TUGAS (DIPERBAIKI) ---
# Sekarang hanya ada 2 tugas, membaca dari base_input_path
tasks = [
    {
        'name': 'prec_standar',
        'base_path': base_input_path,
        'file_template': '{month}/{month}_{model}.nc',
        'var_names': ['prec', 'precipitation'], 
        'output_prefix': 'acc'
    },
    {
        'name': 'sst_nino', 
        'base_path': base_input_path,
        'file_template': '{month}/{month}_nino_{model}.nc',
        'var_names': ['sst', 'tos'], 
        'output_prefix': 'acc_nino'
    }
    # Task ketiga ('sst_regrid_input') telah dihapus karena salah/redundan
]

print(f"Memulai pemrosesan...")
print(f"Output akan disimpan di: {base_output_path}")

for month in months:
    print(f"\n=== Memproses bulan: {month.upper()} ===")
    
    month_output_path = os.path.join(base_output_path, month)
    os.makedirs(month_output_path, exist_ok=True)
    
    for model in models:
        print(f"  --- Model: {model.upper()} ---")
        
        for task in tasks:
            print(f"    * Memproses Tipe: {task['name']}")
            
            input_file = os.path.join(task['base_path'], task['file_template'].format(month=month, model=model))
            output_file = os.path.join(month_output_path, f"{task['output_prefix']}_{month}_{model}.nc")
            
            try:
                dataset = xr.open_dataset(input_file, decode_times=False)
                
                var_name = None
                for v in task['var_names']:
                    if v in dataset.data_vars:
                        var_name = v
                        break
                
                if var_name is None:
                    var_name = list(dataset.data_vars.keys())[0]
                    print(f"      - Peringatan: Variabel {task['var_names']} tidak ditemukan. Menggunakan var default: {var_name}")
                
                print(f"      - Variabel: {var_name}")
                
                data_array = dataset[var_name]
                
                print(f"      - Dimensi Asli: {data_array.dims}")
                
                # Ini adalah logika penting: merata-ratakan dimensi 'S' jika ada
                if 'S' in data_array.dims and len(data_array.S) > 1:
                    print(f"      - Dimensi 'S' ({len(data_array.S)} langkah) ditemukan. Mengambil rata-rata...")
                    data_array = data_array.mean(dim='S', keep_attrs=True)
                    print(f"      - Dimensi Baru: {data_array.dims}")
                
                processed_dataset = data_array.to_dataset(name=var_name)
                
                print(f"      - Shape: {processed_dataset[var_name].shape}")
                print(f"      - Koordinat: {list(processed_dataset.coords.keys())}")
                
                processed_dataset.to_netcdf(output_file)
                
                print(f"      - Disimpan sebagai: {output_file}")
                print(f"      - Nilai (min/max/mean): "
                      f"{processed_dataset[var_name].min().values:.2f} / "
                      f"{processed_dataset[var_name].max().values:.2f} / "
                      f"{processed_dataset[var_name].mean().values:.2f}")
                
                dataset.close()
                
            except FileNotFoundError:
                print(f"      - File tidak ditemukan: {input_file}")
            except Exception as e:
                print(f"      - Error: {str(e)}")
    
    print(f"=== Selesai memproses bulan: {month.upper()} ===\n")

print("Semua file telah diproses!")
print(f"Output disimpan di: {base_output_path}")