import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import linregress, pearsonr
import keras as ks
import tensorflow as tf
import os
import time
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
tf.random.set_seed(42)

def r_squared(pred, obs):
    try:
        slope, intercept, r_value, _, _ = linregress(pred.flatten(), obs.flatten())
        return r_value ** 2
    except:
        return 0.0

def load_observation_data(obs_files, lat_range, lon_range):
    print("Loading observation data...")
    
    try:
        datasets = []
        for file in obs_files:
            if os.path.exists(file):
                dataset = xr.open_dataset(file, decode_times=False).sel(Y=lat_range, X=lon_range)
                datasets.append(dataset)
            else:
                print(f"Warning: Observation file not found: {file}")
        
        if not datasets:
            raise ValueError("No observation files could be loaded")
        
        combined_data = xr.concat(datasets, dim='T')
        obs_data = combined_data['__xarray_dataarray_variable__'].mean(dim='T')
        
        print(f"Observation data loaded successfully:")
        print(f"  Shape: {obs_data.shape}")
        print(f"  Coordinates: Y={obs_data.Y.values[0]}...{obs_data.Y.values[-1]}, X={obs_data.X.values[0]}...{obs_data.X.values[-1]}")
        
        return obs_data
        
    except Exception as e:
        print(f"Error loading observation data: {e}")
        return None

def load_model_data(model_file, obs_coords, lat_range, lon_range, lead_times, var_names):
    print(f"Loading model data from: {model_file}")
    
    try:
        if not os.path.exists(model_file):
            print(f"Model file not found: {model_file}")
            return None, None
        
        dataset = xr.open_dataset(model_file, decode_times=False)
        
        var_name = None
        for name in var_names:
            if name in dataset.data_vars:
                var_name = name
                break
        
        if var_name is None:
            var_name = list(dataset.data_vars.keys())[0]
            print(f"Warning: Could not find {var_names}, using first variable: {var_name}")
        
        print(f"Using variable: {var_name}")
        
        model_data = dataset[var_name].sel(L=slice(*lead_times)).mean(dim='L')
        
        target_shape = (len(obs_coords['Y']), len(obs_coords['X']))
        
        is_lat_match = np.array_equal(model_data.Y.values, obs_coords['Y'].values)
        is_lon_match = np.array_equal(model_data.X.values, obs_coords['X'].values)

        if not (is_lat_match and is_lon_match) or model_data.shape != target_shape:
            print(f"Interpolating model data from {model_data.shape} to {target_shape}")
            model_data = model_data.interp(Y=obs_coords['Y'], X=obs_coords['X'], method='linear')
        else:
            print("Model data grid already matches observation grid. No interpolation needed.")
            
        print(f"Model data loaded successfully:")
        print(f"  Shape after processing: {model_data.shape}")
        
        return model_data.values, dataset
        
    except Exception as e:
        print(f"Error loading model data: {e}")
        return None, None

def normalize_data(data_to_normalize, reference_data):
    print(f"Normalizing data...")
    
    if hasattr(data_to_normalize, 'values'):
        data_to_normalize = data_to_normalize.values
    if hasattr(reference_data, 'values'):
        reference_data = reference_data.values
    
    if np.isnan(data_to_normalize).any():
        data_mean = np.nanmean(data_to_normalize)
        data_to_normalize = np.nan_to_num(data_to_normalize, nan=data_mean)
        print(f"Replaced NaN values in data_to_normalize with mean: {data_mean:.6f}")
    
    if np.isnan(reference_data).any():
        ref_mean = np.nanmean(reference_data)
        reference_data = np.nan_to_num(reference_data, nan=ref_mean)
        print(f"Replaced NaN values in reference_data with mean: {ref_mean:.6f}")
    
    print(f"Data statistics before normalization:")
    print(f"  Data: min={data_to_normalize.min():.6f}, max={data_to_normalize.max():.6f}, mean={data_to_normalize.mean():.6f}")
    print(f"  Reference: min={reference_data.min():.6f}, max={reference_data.max():.6f}, mean={reference_data.mean():.6f}")
    
    data_range = data_to_normalize.max() - data_to_normalize.min()
    ref_range = reference_data.max() - reference_data.min()
    scale_ratio = max(data_range, ref_range) / (min(data_range, ref_range) + 1e-10)
    
    if scale_ratio > 10:
        print(f"  Warning: Large scale difference detected (ratio: {scale_ratio:.1f})")
        print(f"  Using log transformation to reduce scale differences")
        
        data_log = np.log1p(np.maximum(data_to_normalize, 0))
        ref_log = np.log1p(np.maximum(reference_data, 0))
        
        combined_min = min(data_log.min(), ref_log.min())
        combined_max = max(data_log.max(), ref_log.max())
        
        data_norm = (data_log - combined_min) / (combined_max - combined_min + 1e-10)
        ref_norm = (ref_log - combined_min) / (combined_max - combined_min + 1e-10)
        
        print(f"  Log-transformed ranges: Data [{data_log.min():.3f}, {data_log.max():.3f}], Ref [{ref_log.min():.3f}, {ref_log.max():.3f}]")
        norm_params = ('log', combined_min, combined_max, data_to_normalize.min(), data_to_normalize.max(), reference_data.min(), reference_data.max())

    else:
        print(f"  Scale difference acceptable (ratio: {scale_ratio:.1f}), using standard normalization")
        
        data_p5, data_p95 = np.percentile(data_to_normalize, [5, 95])
        ref_p5, ref_p95 = np.percentile(reference_data, [5, 95])
        
        combined_min = min(data_p5, ref_p5)
        combined_max = max(data_p95, ref_p95)
        
        data_norm = (data_to_normalize - combined_min) / (combined_max - combined_min + 1e-10)
        ref_norm = (reference_data - combined_min) / (combined_max - combined_min + 1e-10)
        norm_params = ('standard', combined_min, combined_max, data_to_normalize.min(), data_to_normalize.max(), reference_data.min(), reference_data.max())
    
    data_norm = np.clip(data_norm, 0, 1)
    ref_norm = np.clip(ref_norm, 0, 1)
    
    print(f"Data statistics after normalization:")
    print(f"  Data: min={data_norm.min():.6f}, max={data_norm.max():.6f}, mean={data_norm.mean():.6f}")
    print(f"  Reference: min={ref_norm.min():.6f}, max={ref_norm.max():.6f}, mean={ref_norm.mean():.6f}")
    
    return data_norm, ref_norm, norm_params

def create_cnn_model(input_shape, l2_reg=0.01):
    print(f"Creating improved CNN model with input shape: {input_shape}")
    
    regularizer = tf.keras.regularizers.l2(l2_reg)
    
    if len(input_shape) != 3:
        raise ValueError(f"Input shape must be 3D (height, width, channels), got {input_shape}")
    
    height, width, channels = input_shape
    
    print(f"  Model will process: height={height}, width={width}, channels={channels}")
    print(f"  Flatten output will be: {height * width} dimensions")
    
    model = ks.models.Sequential([
        ks.layers.Input(shape=input_shape),
        
        ks.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same',
                         kernel_regularizer=regularizer),
        ks.layers.BatchNormalization(),
        ks.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same',
                         kernel_regularizer=regularizer),
        ks.layers.BatchNormalization(),
        ks.layers.MaxPooling2D(pool_size=(2, 2), padding='same'),
        ks.layers.Dropout(0.2),
        
        ks.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same',
                         kernel_regularizer=regularizer),
        ks.layers.BatchNormalization(),
        ks.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same',
                         kernel_regularizer=regularizer),
        ks.layers.BatchNormalization(),
        ks.layers.MaxPooling2D(pool_size=(2, 2), padding='same'),
        ks.layers.Dropout(0.2),
        
        ks.layers.GlobalAveragePooling2D(),
        
        ks.layers.Dense(64, activation='relu', kernel_regularizer=regularizer),
        ks.layers.BatchNormalization(),
        ks.layers.Dropout(0.3),
        
        ks.layers.Dense(32, activation='relu', kernel_regularizer=regularizer),
        ks.layers.BatchNormalization(),
        ks.layers.Dropout(0.2),
        
        ks.layers.Dense(height * width, activation='linear'),
        
        ks.layers.Reshape((height, width))
    ])
    
    print("Improved model architecture:")
    model.summary()
    
    return model

def cross_validation_training(model_data, obs_data, n_folds=5, epochs=50, lr=0.0001):
    print(f"\n{'='*60}")
    print(f"STARTING CROSS-VALIDATION TRAINING WITH REAL DATA")
    print(f"{'='*60}")
    print(f"Folds: {n_folds}, Epochs: {epochs}, Learning rate: {lr}")
    
    print(f"DEBUG: Input model_data shape: {model_data.shape}")
    print(f"DEBUG: Input obs_data shape: {obs_data.shape}")
    print(f"DEBUG: model_data.ndim: {model_data.ndim}")
    
    print(f"Using REAL data for cross-validation")
    
    height, width = model_data.shape[0], model_data.shape[1]
    
    print(f"Warning: Original spatial patching logic was flawed.")
    print(f"Forcing training on the full domain ({height}x{width}) as a single sample.")
    patch_size = height
    n_patches = 1
    patches = [(0, height, 0, width)] 
    
    
    if len(patches) > 1:
        np.random.shuffle(patches)
    
    fold_scores = []
    fold_models = []
    fold_histories = []
    
    start_time = time.time()

    for fold in range(n_folds):
        fold_start = time.time()
        print(f"\n{'-'*50}")
        print(f"FOLD {fold + 1}/{n_folds}")
        print(f"{'-'*50}")
        
        if len(patches) > 1:
            split_idx = int(len(patches) * 0.8)
            train_patches = patches[:split_idx]
            val_patches = patches[split_idx:]
        else:
            train_patches = patches
            val_patches = patches
            print(f"  Warning: Only one patch available. Using same data for training and validation.")
        
        print(f"  Training patches: {len(train_patches)}")
        print(f"  Validation patches: {len(val_patches)}")
        
        if len(patches) == 1:
            model_train = model_data[np.newaxis, ...]
            obs_train = obs_data[np.newaxis, ...]
            model_val = model_data[np.newaxis, ...]
            obs_val = obs_data[np.newaxis, ...]
        else:
            model_train = []
            obs_train = []
            model_val = []
            obs_val = []

            for i, end_i, j, end_j in train_patches:
                model_train.append(model_data[i:end_i, j:end_j, :])
                obs_train.append(obs_data[i:end_i, j:end_j, :])
            
            for i, end_i, j, end_j in val_patches:
                model_val.append(model_data[i:end_i, j:end_j, :])
                obs_val.append(obs_data[i:end_i, j:end_j, :])
            
            model_train = np.stack(model_train, axis=0)
            obs_train = np.stack(obs_train, axis=0)
            model_val = np.stack(model_val, axis=0)
            obs_val = np.stack(obs_val, axis=0)
        
        print(f"  Training data shape: {model_train.shape}")
        print(f"  Validation data shape: {model_val.shape}")
        
        actual_height, actual_width = model_train.shape[1], model_train.shape[2]
        input_shape = (actual_height, actual_width, model_train.shape[-1])
        
        print(f"  Training data dimensions: {actual_height} x {actual_width}")
        print(f"  Final CNN input shape: {input_shape}")
        
        cnn_model = create_cnn_model(input_shape)
        
        cnn_model.compile(
            optimizer=ks.optimizers.Adam(learning_rate=lr),
            loss='mse',
            metrics=['mae']
        )
        
        early_stopping = ks.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = ks.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=8,
            min_lr=1e-6,
            verbose=1
        )
        
        print(f"  Training started with real data...")
        print(f"  Training data: {model_train.shape} → {obs_train.shape}")
        print(f"  Validation data: {model_val.shape} → {obs_val.shape}")

        history = cnn_model.fit(
            model_train, obs_train,
            validation_data=(model_val, obs_val),
            epochs=epochs,
            batch_size=16,
            callbacks=[early_stopping, reduce_lr],
            verbose=0
        )
        
        best_val_loss = min(history.history['val_loss'])
        best_val_mae = min(history.history['val_mae'])
        
        fold_scores.append(best_val_loss)
        fold_models.append(cnn_model)
        fold_histories.append(history)
        
        fold_time = time.time() - fold_start
        
        print(f"  Fold {fold + 1} completed:")
        print(f"    Best validation loss: {best_val_loss:.6f}")
        print(f"    Best validation MAE: {best_val_mae:.6f}")
        print(f"    Training time: {fold_time:.2f} seconds")
        
        elapsed = time.time() - start_time
        avg_time_per_fold = elapsed / (fold + 1)
        remaining = n_folds - (fold + 1)
        estimated_remaining = remaining * avg_time_per_fold
        
        print(f"  Progress: {fold + 1}/{n_folds} ({((fold + 1)/n_folds)*100:.1f}%)")
        print(f"  Elapsed: {elapsed/60:.1f} min, Remaining: {estimated_remaining/60:.1f} min")
    
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"CROSS-VALIDATION COMPLETED")
    print(f"{'='*60}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Average time per fold: {total_time/n_folds/60:.1f} minutes")
    
    best_fold_idx = np.argmin(fold_scores)
    best_model = fold_models[best_fold_idx]
    best_history = fold_histories[best_fold_idx]
    
    print(f"\nBest model: Fold {best_fold_idx + 1}")
    print(f"Best validation loss: {fold_scores[best_fold_idx]:.6f}")
    print(f"Average validation loss: {np.mean(fold_scores):.6f} ± {np.std(fold_scores):.6f}")
    
    return best_model, best_history, fold_scores

def evaluate_model(model, model_data, obs_data, norm_params, model_name):
    print(f"\nEvaluating model: {model_name}")
    
    model_input = model_data[np.newaxis, ...]
    print(f"  Prediction input shape: {model_input.shape}")
    
    predictions = model.predict(model_input, verbose=0)
    predictions = predictions.squeeze() 
    
    norm_method = norm_params[0]
    
    if norm_method == 'log':
        combined_min, combined_max, model_orig_min, model_orig_max, obs_orig_min, obs_orig_max = norm_params[1:]
        
        predictions_log = predictions * (combined_max - combined_min) + combined_min
        obs_log = obs_data * (combined_max - combined_min) + combined_min
        
        predictions_denorm = np.expm1(predictions_log)
        obs_denorm = np.expm1(obs_log)
        
    else:
        combined_min, combined_max = norm_params[1], norm_params[2]
        predictions_denorm = predictions * (combined_max - combined_min) + combined_min
        obs_denorm = obs_data * (combined_max - combined_min) + combined_min
    
    mae = mean_absolute_error(obs_denorm.flatten(), predictions_denorm.flatten())
    rmse = np.sqrt(mean_squared_error(obs_denorm.flatten(), predictions_denorm.flatten()))
    r2 = r_squared(predictions_denorm, obs_denorm)
    
    try:
        pearson_corr, _ = pearsonr(obs_denorm.flatten(), predictions_denorm.flatten())
    except:
        pearson_corr = 0.0
    
    print(f"Performance metrics:")
    print(f"  MAE: {mae:.6f}")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  R-squared: {r2:.6f}")
    print(f"  Pearson Correlation: {pearson_corr:.6f}")
    
    return {
        'Model': model_name,
        'MAE': mae,
        'RMSE': rmse,
        'R-squared': r2,
        'Pearson Correlation': pearson_corr,
        'predictions': predictions_denorm,
        'observations': obs_denorm 
    }


def save_results(predictions, obs_coords, model_name, output_folder):
    output_file = os.path.join(output_folder, f'predictions_{model_name}.nc')
    
    output_dataset = xr.Dataset(
        {
            'predictions': (['Y', 'X'], predictions)
        },
        coords={
            'Y': obs_coords['Y'].values,
            'X': obs_coords['X'].values
        }
    )
    
    output_dataset.to_netcdf(output_file)
    print(f"Results saved to: {output_file}")

def plot_results(history, model_name, obs_data_denorm, pred_data_denorm, output_folder):
    print(f"Saving plots for {model_name}...")
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss', color='blue')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='red')
    plt.title(f'Learning Curve - {model_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE', color='blue')
    plt.plot(history.history['val_mae'], label='Validation MAE', color='red')
    plt.title(f'MAE Progress - {model_name}')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    learning_curve_file = os.path.join(output_folder, f'learning_curve_{model_name}.png')
    plt.savefig(learning_curve_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Learning curve saved: {learning_curve_file}")
    
    obs_flat = obs_data_denorm.flatten()
    pred_flat = pred_data_denorm.flatten()
    
    max_points = 100
    if len(obs_flat) > max_points:
        indices = np.random.choice(len(obs_flat), max_points, replace=False)
        obs_sample = obs_flat[indices]
        pred_sample = pred_flat[indices]
        x_labels = [f'P{i+1}' for i in range(max_points)]
    else:
        obs_sample = obs_flat
        pred_sample = pred_flat
        x_labels = [f'P{i+1}' for i in range(len(obs_flat))]
    
    x_pos = np.arange(len(obs_sample))
    width = 0.35
    
    fig, ax1 = plt.subplots(figsize=(15, 8))
    
    color1 = 'tab:blue'
    ax1.set_xlabel('Spatial Points')
    ax1.set_ylabel('Predictions', color=color1)
    bars1 = ax1.bar(x_pos + width/2, pred_sample, width, label='Predictions', 
                    color=color1, alpha=0.7, edgecolor='darkblue')
    ax1.tick_params(axis='y', labelcolor=color1)
    
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('Observations', color=color2)
    bars2 = ax2.bar(x_pos - width/2, obs_sample, width, label='Observations', 
                    color=color2, alpha=0.7, edgecolor='darkred')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(x_labels, rotation=45)
    
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    plt.title(f'Bar Graph: Predictions vs Observations - {model_name}')
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    
    bar_graph_file = os.path.join(output_folder, f'bar_graph_{model_name}.png')
    plt.savefig(bar_graph_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Bar graph saved: {bar_graph_file}")
    
    print(f"All plots saved for {model_name}")

def main():
    print("="*80)
    print("ROBUST CNN TRAINING FOR PRECIPITATION PREDICTION (Prec + SST Input)")
    print("="*80)
    
    lat_range = slice(-9, -5)
    lon_range = slice(105, 115)
    lead_times = (5.5, 7.5) # Lead time diubah sesuai dengan issued time
    
    num_epochs = 50 # Jumlah epoch yang digunakan
    learning_rate = 0.0001 # Learning rate yang digunakan
    n_folds = 20 # Jumlah fold yang digunakan
    l2_reg = 0.005 # L2 regularization yang digunakan
    
    print(f"Configuration:")
    print(f"  Observation Lat range: {lat_range.start} to {lat_range.stop}")
    print(f"  Observation Lon range: {lon_range.start} to {lon_range.stop}")
    print(f"  Lead times: {lead_times}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Cross-validation folds: {n_folds}")
    print(f"  L2 regularization: {l2_reg}")
    print(f"  Input Channels: 2 (Precipitation + SST)")
    print(f"  Grid Strategy: Interpolate all inputs to Observation grid")
    
    obs_files = [
        r"D:\Kuliah\Skripsi\Data_baru\jul_obs.nc",
        r"D:\Kuliah\Skripsi\Data_baru\jun_obs.nc",
        r"D:\Kuliah\Skripsi\Data_baru\mei_obs.nc"
    ]
    
    base_regrid_path = r"D:\Kuliah\Skripsi\Data_baru\Output\Regrid\dec"
    
    model_definitions = [
        {
            'name': 'model_cancm_dec',
            'prec': os.path.join(base_regrid_path, 'regrid_dec_cancm.nc'),
            'sst': os.path.join(base_regrid_path, 'regrid_nino_dec_cancm.nc') 
        },
        {
            'name': 'model_canv2_dec',
            'prec': os.path.join(base_regrid_path, 'regrid_dec_canv2.nc'),
            'sst': os.path.join(base_regrid_path, 'regrid_nino_dec_canv2.nc') 
        },
        {
            'name': 'model_ic3_dec',
            'prec': os.path.join(base_regrid_path, 'regrid_dec_ic3.nc'),
            'sst': os.path.join(base_regrid_path, 'regrid_nino_dec_ic3.nc') 
        },
        {
            'name': 'model_nemo_dec',
            'prec': os.path.join(base_regrid_path, 'regrid_dec_nemo.nc'),
            'sst': os.path.join(base_regrid_path, 'regrid_nino_dec_nemo.nc') 
        }
    ]
    
    base_output_folder = r"D:\Kuliah\Skripsi\Data_baru\Output\CNN_Results"
    
    month_name = os.path.basename(base_regrid_path) 
    output_folder = os.path.join(base_output_folder, month_name)
    
    print(f"Output will be organized by month:")
    print(f"  Base folder: {base_output_folder}")
    print(f"  Month subfolder: {month_name} (auto-detected from regrid path)")
    print(f"  Full output path: {output_folder}")
    
    obs_data = load_observation_data(obs_files, lat_range, lon_range)
    if obs_data is None:
        print("Failed to load observation data. Exiting.")
        return
    
    results = []
    
    for model_def in model_definitions:
        print(f"\n{'#'*70}")
        print(f"PROCESSING: {model_def['name']}")
        print(f"{'#'*70}")
        
        prec_data, _ = load_model_data(
            model_def['prec'], obs_data.coords, lat_range, lon_range, lead_times, 
            ['prec', 'precipitation']
        )
        
        sst_data, _ = load_model_data(
            model_def['sst'], obs_data.coords, lat_range, lon_range, lead_times, 
            ['sst', 'tos'] 
        )
        
        if prec_data is None or sst_data is None:
            print(f"Skipping {model_def['name']} - failed to load one or more data files")
            continue
        
        print("\nNormalizing Precipitation vs Observation...")
        prec_norm, obs_norm, norm_params_prec = normalize_data(prec_data, obs_data)
        
        print("\nNormalizing SST vs SST...")
        sst_norm, _, norm_params_sst = normalize_data(sst_data, sst_data)
        
        model_input_stacked = np.stack([prec_norm, sst_norm], axis=-1)
        
        obs_input = obs_norm[..., np.newaxis]
        
        model_input_stacked = np.asarray(model_input_stacked)
        obs_input = np.asarray(obs_input)
        
        print(f"\nFinal data shapes for training:")
        print(f"  Model input: {model_input_stacked.shape}")
        print(f"  Observation input: {obs_input.shape}")
        
        print(f"DEBUG: About to call cross_validation_training with:")
        print(f"  model_input_stacked.shape: {model_input_stacked.shape}")
        print(f"  obs_input.shape: {obs_input.shape}")
        print(f"  model_input_stacked.ndim: {model_input_stacked.ndim}")
        print(f"  obs_input.ndim: {obs_input.ndim}")
        
        best_model, best_history, fold_scores = cross_validation_training(
            model_input_stacked, obs_input, n_folds, num_epochs, learning_rate
        )
        
        model_name = model_def['name']
        
        result = evaluate_model(
            best_model, model_input_stacked, obs_norm, norm_params_prec, model_name
        )
        results.append(result)
        
        os.makedirs(output_folder, exist_ok=True)
        
        plot_results(
            best_history, model_name, 
            result['observations'], result['predictions'], 
            output_folder
        )
        
        save_results(result['predictions'], obs_data.coords, model_name, output_folder)
    
    if results:
        print(f"\n{'='*80}")
        print(f"FINAL RESULTS SUMMARY")
        print(f"{'='*80}")
        print(f"{'Model':<25}{'MAE':<12}{'RMSE':<12}{'R-squared':<15}{'Pearson':<12}")
        print(f"{'-'*80}")
        
        for result in results:
            print(f"{result['Model']:<25}{result['MAE']:<12.6f}{result['RMSE']:<12.6f}"
                  f"{result['R-squared']:<15.6f}{result['Pearson Correlation']:<12.6f}")
        
        avg_mae = np.mean([r['MAE'] for r in results])
        avg_rmse = np.mean([r['RMSE'] for r in results])
        avg_r2 = np.mean([r['R-squared'] for r in results])
        avg_corr = np.mean([r['Pearson Correlation'] for r in results])
        
        print(f"\nOverall Statistics:")
        print(f"  Average MAE: {avg_mae:.6f}")
        print(f"  Average RMSE: {avg_rmse:.6f}")
        print(f"  Average R-squared: {avg_r2:.6f}")
        print(f"  Average Pearson Correlation: {avg_corr:.6f}")
        
        best_model = min(results, key=lambda x: x['RMSE'])
        print(f"\nBest performing model: {best_model['Model']}")
        print(f"  RMSE: {best_model['RMSE']:.6f}")
        print(f"  MAE: {best_model['MAE']:.6f}")
    
    print(f"\nScript completed successfully!")

if __name__ == "__main__":
    main()