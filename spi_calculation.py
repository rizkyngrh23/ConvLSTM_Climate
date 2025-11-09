import xarray as xr
import numpy as np
from scipy import stats
from scipy.special import gamma
import os
from tqdm import tqdm

def fit_gamma_distribution(data):
    valid_data = data[data > 0]
    
    if len(valid_data) < 2:
        return None, None
    
    if np.std(valid_data) < 1e-6:
        return None, None
    
    try:
        try:
            shape, loc, scale = stats.gamma.fit(valid_data, floc=0)
        except:
            mean_val = np.mean(valid_data)
            var_val = np.var(valid_data)
            if var_val > 0:
                shape = (mean_val ** 2) / var_val
                scale = var_val / mean_val
            else:
                return None, None
        
        if shape <= 0 or scale <= 0 or np.isnan(shape) or np.isnan(scale):
            return None, None
            
        return shape, scale
    except:
        return None, None

def gamma_to_normal_cdf(x, shape, scale):
    if shape is None or scale is None:
        return np.nan
    
    cdf = stats.gamma.cdf(x, shape, scale=scale)
    
    normal_cdf = stats.norm.ppf(cdf)
    
    return normal_cdf

def calculate_spi_6month(precip_data, window=6):
    print(f"Menghitung SPI-{window} untuk data dengan shape: {precip_data.shape}")
    
    if 'T' in precip_data.dims:
        time_dim = 'T'
    elif 'L' in precip_data.dims:
        time_dim = 'L'
    else:
        raise ValueError("No time dimension found. Expected 'T' or 'L'")
    
    print("Menghitung akumulasi 6 bulan...")
    cum_precip = precip_data.rolling({time_dim: window}, min_periods=window, center=False).sum()
    
    spi_values = np.full_like(cum_precip.values, np.nan)
    
    if len(cum_precip.dims) == 3:
        time_len, lat_len, lon_len = cum_precip.shape
    else:
        lat_len, lon_len = cum_precip.shape
        time_len = 1
    
    print(f"Memproses {lat_len} x {lon_len} grid points...")
    print(f"Data range: min={precip_data.min().values:.6f}, max={precip_data.max().values:.6f}")
    print(f"Cumulative data range: min={cum_precip.min().values:.6f}, max={cum_precip.max().values:.6f}")
    
    valid_points = 0
    total_points = lat_len * lon_len
    
    for i in tqdm(range(lat_len), desc="Processing latitude"):
        for j in range(lon_len):
            if time_len > 1:
                time_series = cum_precip.values[:, i, j]
            else:
                time_series = cum_precip.values[i, j]
            
            valid_mask = ~np.isnan(time_series)
            valid_data = time_series[valid_mask]
            
            if len(valid_data) < 5:
                continue
            
            if np.std(valid_data) < 1e-6:
                continue
            
            shape, scale = fit_gamma_distribution(valid_data)
            
            if shape is not None and scale is not None:
                valid_points += 1
                for t in range(len(time_series)):
                    if not np.isnan(time_series[t]) and time_series[t] > 0:
                        if time_len > 1:
                            spi_values[t, i, j] = gamma_to_normal_cdf(time_series[t], shape, scale)
                        else:
                            spi_values[i, j] = gamma_to_normal_cdf(time_series[t], shape, scale)
    
    print(f"Successfully processed {valid_points}/{total_points} grid points ({valid_points/total_points*100:.1f}%)")
    
    spi_da = xr.DataArray(
        spi_values,
        coords=cum_precip.coords,
        dims=cum_precip.dims,
        name='spi'
    )
    
    return spi_da

def calculate_spi_from_spatial_distribution(precip_values, coords):
    print("Calculating SPI from spatial distribution of original data...")
    
    lat_len, lon_len = precip_values.shape
    
    spi_values = np.full_like(precip_values, np.nan)
    
    print(f"Processing {lat_len} x {lon_len} grid points...")
    print(f"Data range: min={precip_values.min():.6f}, max={precip_values.max():.6f}")
    
    valid_points = 0
    total_points = lat_len * lon_len
    
    for i in tqdm(range(lat_len), desc="Processing latitude"):
        for j in range(lon_len):
            precip_val = precip_values[i, j]
            
            if np.isnan(precip_val) or precip_val <= 0:
                continue
            
            spatial_data = precip_values.flatten()
            spatial_data = spatial_data[~np.isnan(spatial_data)]
            spatial_data = spatial_data[spatial_data > 0]
            
            if len(spatial_data) < 10:
                continue
            
            if np.std(spatial_data) < 1e-6:
                continue
            
            shape, scale = fit_gamma_distribution(spatial_data)
            
            if shape is not None and scale is not None:
                valid_points += 1
                spi_values[i, j] = gamma_to_normal_cdf(precip_val, shape, scale)
    
    print(f"Successfully processed {valid_points}/{total_points} grid points ({valid_points/total_points*100:.1f}%)")
    
    spi_da = xr.DataArray(
        spi_values,
        coords=coords,
        dims=['Y', 'X'],
        name='spi'
    )
    
    return spi_da

base_input_path = 'D:/Kuliah/Skripsi/Data_baru/Output/CNN_Results'
base_output_path = 'D:/Kuliah/Skripsi/Data_baru/Output/SPI_CNN'

os.makedirs(base_output_path, exist_ok=True)

cnn_months = ['nov', 'dec', 'jan', 'feb', 'mar'] 

print("=== SPI CALCULATION FOR CNN PREDICTION DATA ===")
print(f"Processing {len(cnn_months)} months...")

for month in cnn_months:
    print(f"\n--- Processing {month.upper()} ---")
    
    month_input_dir = f'{base_input_path}/{month}'
    
    if not os.path.exists(month_input_dir):
        print(f"✗ Month directory not found: {month_input_dir}")
        continue
    
    prediction_files = [f for f in os.listdir(month_input_dir) if f.startswith('predictions_') and f.endswith('.nc')]
    
    if not prediction_files:
        print(f"✗ No prediction files found in: {month_input_dir}")
        continue
    
    print(f"Found {len(prediction_files)} prediction files:")
    for pred_file in prediction_files:
        print(f"  - {pred_file}")
    
    for pred_file in prediction_files:
        model_name = pred_file.replace('predictions_', '').replace('.nc', '')
        print(f"\n  Processing model: {model_name}")
        
        input_file = f'{month_input_dir}/{pred_file}'
        
        month_output_dir = f'{base_output_path}/{month}'
        os.makedirs(month_output_dir, exist_ok=True)
        
        output_file = f'{month_output_dir}/spi_{month}_{model_name}.nc'
        
        try:
            print(f"Loading data from: {input_file}")
            ds = xr.open_dataset(input_file, decode_times=False)
            
            if 'predictions' in ds.data_vars:
                precip_data = ds['predictions']
            else:
                var_name = list(ds.data_vars.keys())[0]
                precip_data = ds[var_name]
            
            print(f"Prediction data shape: {precip_data.shape}")
            print(f"Prediction data dims: {precip_data.dims}")
            
            if len(precip_data.dims) == 2:
                print("Single prediction data detected. Using original data for SPI calculation...")
                
                precip_values = precip_data.values
                
                print("Using spatial distribution of original prediction data for SPI calculation")
                
                spi_result = calculate_spi_from_spatial_distribution(precip_values, precip_data.coords)
                
            else:
                spi_result = calculate_spi_6month(precip_data, window=6)
            
            print(f"Saving SPI result to: {output_file}")
            spi_result.to_netcdf(output_file)
            
            print(f"SPI Statistics for {month} - {model_name}:")
            print(f"  - Min: {spi_result.min().values:.3f}")
            print(f"  - Max: {spi_result.max().values:.3f}")
            print(f"  - Mean: {spi_result.mean().values:.3f}")
            print(f"  - Std: {spi_result.std().values:.3f}")
            print(f"  - NaN count: {spi_result.isnull().sum().values}")
            
            ds.close()
            
        except FileNotFoundError:
            print(f"✗ File not found: {input_file}")
        except Exception as e:
            print(f"✗ Error processing {month} - {model_name}: {str(e)}")

print("\n=== SPI CALCULATION COMPLETED (Using Original Data) ===")
print(f"Results saved to: {base_output_path}")
print("Output organized by month folders:")
for month in cnn_months:
    month_output_dir = f'{base_output_path}/{month}'
    if os.path.exists(month_output_dir):
        files = [f for f in os.listdir(month_output_dir) if f.endswith('.nc')]
        print(f"  {month.upper()}: {len(files)} files in {month_output_dir}")
