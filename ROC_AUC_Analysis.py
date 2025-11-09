import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
from scipy.interpolate import interp1d
import os
import glob
from tqdm import tqdm
import warnings
import geopandas as gpd
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
    CARTOPY_AVAILABLE = True
except ImportError:
    CARTOPY_AVAILABLE = False
    print("Warning: Cartopy not available. Spatial visualization will use basic matplotlib.")
warnings.filterwarnings('ignore')

def format_model_month_name(comparison_name):
    parts = comparison_name.split('_')
    
    if len(parts) >= 3:
        issued_month = parts[1]
        model = parts[-1]
        
        month_map = {
            'nov': 'November',
            'dec': 'December', 
            'jan': 'January',
            'feb': 'February',
            'mar': 'March',
            'apr': 'April',
            'may': 'May',
            'jun': 'June',
            'jul': 'July',
            'aug': 'August',
            'sep': 'September',
            'oct': 'October'
        }
        
        model_map = {
            'cancm': 'CanCM4i',
            'canv2': 'CanSIPSv2',
            'ic3': 'IC3',
            'nemo': 'NEMO'
        }
        
        month_name = month_map.get(issued_month.lower(), issued_month.title())
        model_name = model_map.get(model, model.upper())
        
        return f"{model_name} {month_name}"
    
    else:
        model = parts[-1]
        
        model_map = {
            'cancm': 'CanCM4i',
            'canv2': 'CanSIPSv2',
            'ic3': 'IC3',
            'nemo': 'NEMO'
        }
        
        model_name = model_map.get(model, model.upper())
        
        return f"{model_name}"

def create_smooth_roc_curve(fpr, tpr, n_points=100):
    """
    Create monotonic ROC curves using linear interpolation.
    
    Uses linear interpolation with monotonicity enforcement to ensure
    mathematically correct ROC curves that never decrease.
    """
    if len(fpr) < 2:
        return fpr, tpr
    
    sort_idx = np.argsort(fpr)
    fpr_sorted = fpr[sort_idx]
    tpr_sorted = tpr[sort_idx]
    
    unique_fpr, unique_idx = np.unique(fpr_sorted, return_index=True)
    unique_tpr = tpr_sorted[unique_idx]
    
    if len(unique_fpr) < 2:
        return fpr, tpr
    
    try:
        fpr_smooth = np.linspace(0, 1, n_points)
        tpr_smooth = np.interp(fpr_smooth, unique_fpr, unique_tpr)
        
        for i in range(1, len(tpr_smooth)):
            if tpr_smooth[i] < tpr_smooth[i-1]:
                tpr_smooth[i] = tpr_smooth[i-1]
        
        if fpr_smooth[0] == 0 and tpr_smooth[0] != 0:
            tpr_smooth[0] = 0
        if fpr_smooth[-1] == 1 and tpr_smooth[-1] != 1:
            tpr_smooth[-1] = 1
            
        return fpr_smooth, tpr_smooth
            
    except Exception as e:
        print(f"      Warning: Interpolation failed, using original points: {e}")
        return fpr, tpr

def load_spi_data(file_path):
    """
    Load SPI data from NetCDF file.
    """
    try:
        ds = xr.open_dataset(file_path, decode_times=False)
        if 'spi' in ds.data_vars:
            spi_data = ds['spi']
        else:
            spi_data = list(ds.data_vars.values())[0]
        
        ds.close()
        return spi_data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def classify_drought_conditions(spi_values, thresholds=None):
    """
    Classify drought conditions based on SPI values.
    
    Drought Classification:
    - Extreme Drought: SPI <= -2.00
    - Severe Drought: -1.99 <= SPI <= -1.50
    - Moderate Drought: -1.49 <= SPI <= -1.00
    - Mild Drought: -0.99 <= SPI <= 0
    - Mildly Wet: 0 <= SPI <= 0.99
    - Moderately Wet: 1.00 <= SPI <= 1.49
    - Severely Wet: 1.50 <= SPI <= 1.99
    - Extremely Wet: SPI >= 2.00
    """
    if thresholds is None:
        thresholds = {
            'extreme_drought': -2.00,
            'severe_drought': -1.50,
            'moderate_drought': -1.00,
            'mild_drought': 0,
            'mildly_wet': 0.99,
            'moderately_wet': 1.49,
            'severely_wet': 1.99
        }
    
    drought_mask = spi_values <= 0
    
    classes = np.zeros_like(spi_values, dtype=int)
    
    classes[spi_values <= thresholds['extreme_drought']] = 0
    classes[(spi_values > thresholds['extreme_drought']) & (spi_values <= thresholds['severe_drought'])] = 1
    classes[(spi_values > thresholds['severe_drought']) & (spi_values <= thresholds['moderate_drought'])] = 2
    classes[(spi_values > thresholds['moderate_drought']) & (spi_values <= thresholds['mild_drought'])] = 3
    classes[(spi_values > thresholds['mild_drought']) & (spi_values <= thresholds['mildly_wet'])] = 4
    classes[(spi_values > thresholds['mildly_wet']) & (spi_values <= thresholds['moderately_wet'])] = 5
    classes[(spi_values > thresholds['moderately_wet']) & (spi_values <= thresholds['severely_wet'])] = 6
    classes[spi_values > thresholds['severely_wet']] = 7
    
    return drought_mask, classes

def calculate_roc_auc_binary(obs_spi, pred_spi, threshold=0):
    """
    Calculate ROC and AUC for binary drought classification.
    """
    obs_flat = obs_spi.values.flatten()
    pred_flat = pred_spi.values.flatten()
    
    valid_mask = ~(np.isnan(obs_flat) | np.isnan(pred_flat))
    obs_valid = obs_flat[valid_mask]
    pred_valid = pred_flat[valid_mask]
    
    print(f"   Data validation:")
    print(f"     - Total grid points: {len(obs_flat)}")
    print(f"     - Valid grid points: {len(obs_valid)}")
    print(f"     - NaN in observations: {np.isnan(obs_flat).sum()}")
    print(f"     - NaN in predictions: {np.isnan(pred_flat).sum()}")
    
    if len(obs_valid) == 0:
        print(f"   Error: No valid data points after removing NaN values")
        return None, None, None
    
    y_true = (obs_valid <= threshold).astype(int)
    y_scores = pred_valid
    
    print(f"   Binary classification debug:")
    print(f"     - Threshold: {threshold}")
    print(f"     - Unique classes in y_true: {np.unique(y_true)}")
    print(f"     - Class counts: {np.bincount(y_true) if len(np.unique(y_true)) > 1 else 'N/A'}")
    print(f"     - y_scores range: [{y_scores.min():.3f}, {y_scores.max():.3f}]")
    print(f"     - y_true range: [{y_true.min()}, {y_true.max()}]")
    
    y_scores_std = np.std(y_scores)
    print(f"     - y_scores std: {y_scores_std:.6f}")
    
    if y_scores_std < 1e-6:
        print(f"   Warning: Predictions have very low variation (std = {y_scores_std:.6f})")
        print(f"   This suggests the model may not be making meaningful predictions")
    
    correlation = np.corrcoef(y_true, y_scores)[0, 1]
    print(f"     - Correlation (true vs scores): {correlation:.6f}")
    
    if abs(correlation) < 0.01:
        print(f"   Warning: Very low correlation between true labels and predictions")
        print(f"   This suggests the model predictions are not informative")
    
    if len(np.unique(y_true)) < 2:
        print(f"   Warning: Only one class found in binary classification. Classes: {np.unique(y_true)}")
        return None, None, None
    

    try:
        if len(y_true) < 10:
            print(f"   Warning: Insufficient data for ROC calculation ({len(y_true)} points)")
            return None, None, None
        
        if not np.all(np.isfinite(y_scores)):
            print(f"   Warning: Non-finite values in prediction scores")
            return None, None, None
        
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        
        
        roc_auc = auc(fpr, tpr)
        
        if hasattr(roc_auc, 'item'):
            roc_auc = roc_auc.item()
        
        if np.isnan(roc_auc) or np.isinf(roc_auc):
            print(f"   Warning: Invalid AUC value: {roc_auc}")
            return None, None, None
        
        print(f"   ROC calculation successful: AUC = {roc_auc:.6f}")
        return fpr, tpr, roc_auc
        
    except Exception as e:
        print(f"   Error calculating ROC: {e}")
        print(f"   y_true shape: {y_true.shape}, y_scores shape: {y_scores.shape}")
        print(f"   y_true unique: {np.unique(y_true)}")
        print(f"   y_scores sample: {y_scores[:10]}")
        return None, None, None

def calculate_roc_auc_multiclass(obs_spi, pred_spi):
    """
    Calculate ROC and AUC for multi-class drought classification using continuous SPI scores.
    """
    obs_flat = obs_spi.values.flatten()
    pred_flat = pred_spi.values.flatten()
    
    valid_mask = ~(np.isnan(obs_flat) | np.isnan(pred_flat))
    obs_valid = obs_flat[valid_mask]
    pred_valid = pred_flat[valid_mask]
    
    if len(obs_valid) == 0:
        return None, None, None
    
    print(f"   Multi-class ROC calculation with continuous SPI scores:")
    print(f"     - Valid data points: {len(obs_valid)}")
    print(f"     - Observed SPI range: [{obs_valid.min():.3f}, {obs_valid.max():.3f}]")
    print(f"     - Predicted SPI range: [{pred_valid.min():.3f}, {pred_valid.max():.3f}]")
    
    _, obs_classes = classify_drought_conditions(obs_valid)
    
    if len(np.unique(obs_classes)) < 2:
        print(f"     Warning: Insufficient class variation for ROC calculation")
        print(f"        - Unique observation classes: {np.unique(obs_classes)}")
        return None, None, None
    
    unique_obs, counts_obs = np.unique(obs_classes, return_counts=True)
    print(f"     - Observation classes: {dict(zip(unique_obs, counts_obs))}")
    
    n_classes = 8
    y_true_bin = label_binarize(obs_classes, classes=range(n_classes))
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        if np.sum(y_true_bin[:, i]) > 0:
            try:
                y_true = y_true_bin[:, i]
                if i <= 3:
                    y_scores = -pred_valid
                else:
                    y_scores = pred_valid
                
                fpr[i], tpr[i], _ = roc_curve(y_true, y_scores)
                
                auc_score = roc_auc_score(y_true, y_scores)
                roc_auc[i] = auc_score
                
                fpr[i], tpr[i] = create_smooth_roc_curve(fpr[i], tpr[i], n_points=100)
                
                print(f"        Class {i}: AUC = {auc_score:.3f}, Points = {len(fpr[i])} (smooth curves)")
                
            except Exception as e:
                print(f"        Class {i}: Error calculating ROC - {e}")
                fpr[i] = np.array([0, 1])
                tpr[i] = np.array([0, 1])
                roc_auc[i] = 0.5
        else:
            print(f"        Class {i}: No samples in data")
            fpr[i] = np.array([0, 1])
            tpr[i] = np.array([0, 1])
            roc_auc[i] = 0.5

    try:
        y_true_micro = y_true_bin.ravel()
        
        y_scores_micro = np.tile(pred_valid, n_classes)
        
        fpr["micro"], tpr["micro"], _ = roc_curve(y_true_micro, y_scores_micro)
        
        auc_micro = roc_auc_score(y_true_micro, y_scores_micro, average='micro')
        roc_auc["micro"] = auc_micro
        
        fpr["micro"], tpr["micro"] = create_smooth_roc_curve(fpr["micro"], tpr["micro"], n_points=100)
        
        print(f"        Micro-average: AUC = {auc_micro:.3f}, Points = {len(fpr['micro'])} (smooth curves)")
        
    except Exception as e:
        print(f"        Micro-average: Error calculating ROC - {e}")
        fpr["micro"] = np.array([0, 1])
        tpr["micro"] = np.array([0, 1])
        roc_auc["micro"] = 0.5
    
    for key in roc_auc:
        if np.isnan(roc_auc[key]) or np.isinf(roc_auc[key]):
            print(f"        Warning: Invalid AUC for {key}: {roc_auc[key]}, setting to 0.5")
            roc_auc[key] = 0.5
    
    return fpr, tpr, roc_auc

def plot_roc_curves(fpr, tpr, roc_auc, title, output_path, is_binary=True):
    """
    Plot ROC curves and save to file with smooth curves.
    """
    plt.figure(figsize=(12, 8))
    
    if is_binary:
        plt.plot(fpr, tpr, color='darkred', lw=2, alpha=0.9,
                 label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.7)
        
    else:
        colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'brown', 'pink']
        class_names = ['Extreme Drought', 'Severe Drought', 'Moderate Drought', 'Mild Drought',
                       'Mildly Wet', 'Moderately Wet', 'Severely Wet', 'Extremely Wet']
        
        for i, color in enumerate(colors):
            if i in fpr and i in tpr and i in roc_auc:
                if len(fpr[i]) > 1:
                    plt.plot(fpr[i], tpr[i], color=color, lw=2.5, alpha=0.9,
                             label=f'{class_names[i]} (AUC = {roc_auc[i]:.3f})')
                else:
                    print(f"        Warning: Class {i} has insufficient points for plotting")
        
        if "micro" in fpr and "micro" in tpr and "micro" in roc_auc:
            if len(fpr["micro"]) > 1:
                plt.plot(fpr["micro"], tpr["micro"],
                         label=f'Micro-average (AUC = {roc_auc["micro"]:.3f})',
                         color='black', linestyle='-', linewidth=4, alpha=0.95)
            else:
                print(f"        Warning: Micro-average has insufficient points for plotting")
        
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves: {title}')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"ROC plot saved to: {output_path}")

def create_grid_roc_plot(results, output_folder):
    """
    Create a 2x2 grid plot showing individual drought class ROC curves for each model.
    Uses actual ROC data from multi-class drought classification.
    """
    if not results:
        print("No results to plot in grid format")
        return
    
    model_data = {}
    
    model_map = {
        'cancm': 'CanCM4i',
        'canv2': 'CanSIPSv2',
        'ic3': 'IC3',
        'nemo': 'NEMO'
    }
    
    class_names = ['Extreme Drought', 'Severe Drought', 'Moderate Drought', 'Mild Drought',
                   'Mildly Wet', 'Moderately Wet', 'Severely Wet', 'Extremely Wet']
    class_colors = ['#d62728', '#ff7f0e', '#ffd700', '#2ca02c', 
                    '#1f77b4', '#9467bd', '#8c564b', '#e377c2']
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('Drought Classification ROC Curves by Model and Issued Month', fontsize=16, fontweight='bold', y=0.95)
    
    axes_flat = axes.flatten()
    
    model_roc_data = {}
    
    for result in results:
        comparison_name = result['comparison_name']
        parts = comparison_name.split('_')
        
        if len(parts) >= 3:
            target_month = parts[0]
            issued_month = parts[1]
            model = parts[-1]
        else:
            continue
        
        if model not in model_roc_data:
            model_roc_data[model] = {}
        
        obs_file = result.get('obs_file')
        pred_file = result.get('pred_file')
        
        if obs_file and pred_file:
            try:
                obs_spi = load_spi_data(obs_file)
                pred_spi = load_spi_data(pred_file)
                
                if obs_spi is not None and pred_spi is not None:
                    if obs_spi.shape != pred_spi.shape:
                        pred_spi = pred_spi.interp(Y=obs_spi.Y, X=obs_spi.X, method='linear')

                    fpr_multiclass, tpr_multiclass, auc_multiclass = calculate_roc_auc_multiclass(obs_spi, pred_spi)
                    
                    if fpr_multiclass and tpr_multiclass and auc_multiclass:
                        model_roc_data[model][issued_month] = {
                            'fpr': fpr_multiclass,
                            'tpr': tpr_multiclass,
                            'auc': auc_multiclass
                        }
                        print(f"    ✓ Loaded ROC data for {model_map.get(model, model)} {issued_month}")
                    else:
                        print(f"    ⚠ Failed to calculate ROC for {model} {issued_month}")
                else:
                    print(f"    ⚠ Failed to load data for {model} {issued_month}")
            except Exception as e:
                print(f"    ⚠ Error processing {model} {issued_month}: {e}")
    
    for idx, (model, data) in enumerate(model_roc_data.items()):
        if idx >= 4:
            break
        
        ax = axes_flat[idx]
        model_name = model_map.get(model, model.upper())
        
        for month in ['nov', 'dec', 'jan', 'feb', 'mar']:
            if month in data:
                roc_data = data[month]
                fpr = roc_data['fpr']
                tpr = roc_data['tpr']
                auc = roc_data['auc']
                
                month_linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]
                month_names = ['Nov', 'Dec', 'Jan', 'Feb', 'Mar']
                month_idx = ['nov', 'dec', 'jan', 'feb', 'mar'].index(month)
                linestyle = month_linestyles[month_idx]
                month_name = month_names[month_idx]
                
                for class_idx, (class_name, color) in enumerate(zip(class_names, class_colors)):
                    if class_idx in fpr and class_idx in tpr and class_idx in auc:
                        if len(fpr[class_idx]) > 1:
                            show_legend = (month == 'nov')
                            label = f'{class_name} (AUC={auc[class_idx]:.3f})' if show_legend else ""
                            
                            ax.plot(fpr[class_idx], tpr[class_idx], 
                                    color=color, lw=2.5, alpha=0.8, linestyle=linestyle,
                                    label=label)
        
        if 'nov' in data:
            ax.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--', alpha=0.8, label='Random')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=11)
        ax.set_ylabel('True Positive Rate', fontsize=11)
        ax.set_title(f'{model_name}', fontsize=14, fontweight='bold', pad=15)
        
        legend1 = ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
        
        from matplotlib.lines import Line2D
        month_legend_elements = [
            Line2D([0], [0], color='black', linestyle='-', lw=2.5, label='Nov'),
            Line2D([0], [0], color='black', linestyle='--', lw=2.5, label='Dec'),
            Line2D([0], [0], color='black', linestyle='-.', lw=2.5, label='Jan'),
            Line2D([0], [0], color='black', linestyle=':', lw=2.5, label='Feb'),
            Line2D([0], [0], color='black', linestyle=(0, (3, 1, 1, 1)), lw=2.5, label='Mar')
        ]
        legend2 = ax.legend(handles=month_legend_elements, loc="upper right", 
                            fontsize=9, framealpha=0.9, title="Issued Month")
        legend2.set_title("Issued Month", prop={'size': 9})
        
        ax.add_artist(legend1)
        
        ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
        
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    
    for idx in range(len(model_roc_data), 4):
        axes_flat[idx].set_visible(False)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    
    grid_plot_path = os.path.join(output_folder, 'roc_grid_drought_classes.png')
    plt.savefig(grid_plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Grid ROC plot (drought classes) saved to: {grid_plot_path}")

def create_comprehensive_5x4_plot(results, output_folder):
    """
    Create a comprehensive 5x4 grid plot showing ROC curves.
    Rows = Issued Months (Nov, Dec, Jan, Feb, Mar) - 5 rows
    Columns = Models (CanCM4i, CanSIPSv2, IC3, NEMO) - 4 columns
    """
    if not results:
        print("No results to plot in comprehensive format")
        return
    
    model_map = {
        'cancm': 'CanCM4i',
        'canv2': 'CanSIPSv2',
        'ic3': 'IC3',
        'nemo': 'NEMO'
    }
    
    class_names = ['Extreme Drought', 'Severe Drought', 'Moderate Drought', 'Mild Drought',
                   'Mildly Wet', 'Moderately Wet', 'Severely Wet', 'Extremely Wet']
    class_colors = ['#d62728', '#ff7f0e', '#ffd700', '#2ca02c', 
                    '#1f77b4', '#9467bd', '#8c564b', '#e377c2']
    
    fig, axes = plt.subplots(5, 4, figsize=(20, 16))
    fig.suptitle('Comprehensive ROC Analysis: Issued Month vs Model', fontsize=18, fontweight='bold', y=0.95)
    
    issued_months = ['nov', 'dec', 'jan', 'feb', 'mar']
    month_names = ['November', 'December', 'January', 'February', 'March']
    models = ['cancm', 'canv2', 'ic3', 'nemo']
    
    model_roc_data = {}
    
    for result in results:
        comparison_name = result['comparison_name']
        parts = comparison_name.split('_')
        
        if len(parts) >= 3:
            target_month = parts[0]
            issued_month = parts[1]
            model = parts[-1]
        else:
            continue
        
        if model not in model_roc_data:
            model_roc_data[model] = {}
        
        obs_file = result.get('obs_file')
        pred_file = result.get('pred_file')
        
        if obs_file and pred_file:
            try:
                obs_spi = load_spi_data(obs_file)
                pred_spi = load_spi_data(pred_file)
                
                if obs_spi is not None and pred_spi is not None:
                    if obs_spi.shape != pred_spi.shape:
                        pred_spi = pred_spi.interp(Y=obs_spi.Y, X=obs_spi.X, method='linear')
                    
                    fpr_multiclass, tpr_multiclass, auc_multiclass = calculate_roc_auc_multiclass(obs_spi, pred_spi)
                    
                    if fpr_multiclass and tpr_multiclass and auc_multiclass:
                        model_roc_data[model][issued_month] = {
                            'fpr': fpr_multiclass,
                            'tpr': tpr_multiclass,
                            'auc': auc_multiclass
                        }
                        print(f"    ✓ Loaded ROC data for {model_map.get(model, model)} {issued_month}")
                    else:
                        print(f"    ⚠ Failed to calculate ROC for {model} {issued_month}")
                else:
                    print(f"    ⚠ Failed to load data for {model} {issued_month}")
            except Exception as e:
                print(f"    ⚠ Error processing {model} {issued_month}: {e}")
    
    for row, issued_month in enumerate(issued_months):
        for col, model in enumerate(models):
            ax = axes[row, col]
            
            if row == 0:
                ax.set_title(f'{model_map.get(model, model.upper())}', fontsize=12, fontweight='bold', pad=10)
            
            if col == 0:
                ax.set_ylabel(f'{month_names[row]}', fontsize=11, fontweight='bold', rotation=90, ha='center', va='center')
            
            if model in model_roc_data and issued_month in model_roc_data[model]:
                roc_data = model_roc_data[model][issued_month]
                fpr = roc_data['fpr']
                tpr = roc_data['tpr']
                auc = roc_data['auc']
                
                for class_idx, (class_name, color) in enumerate(zip(class_names, class_colors)):
                    if class_idx in fpr and class_idx in tpr and class_idx in auc:
                        if len(fpr[class_idx]) > 1:
                            ax.plot(fpr[class_idx], tpr[class_idx], 
                                    color=color, lw=2, alpha=0.8,
                                    label=f'{class_name} (AUC={auc[class_idx]:.3f})')
                
                ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', alpha=0.7, label='Random')
                
                if row == 0 and col == 0:
                    ax.legend(loc="lower right", fontsize=6, framealpha=0.9)
            
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
            ax.tick_params(axis='both', which='major', labelsize=8)
            
            if row == 4:
                ax.set_xlabel('False Positive Rate', fontsize=9)
            if col == 0:
                ax.set_ylabel('True Positive Rate', fontsize=9, rotation=90, ha='center', va='center')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    
    comprehensive_plot_path = os.path.join(output_folder, 'roc_comprehensive_5x4.png')
    plt.savefig(comprehensive_plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Comprehensive 5x4 ROC plot saved to: {comprehensive_plot_path}")

def create_individual_roc_plots(results, output_folder):
    """
    Create individual ROC plots for each model-month combination.
    Layout: 4 rows (models) x 5 columns (issued months) = 20 plots
    Each plot shows actual ROC curves for that specific model-month combination.
    """
    if not results:
        print("No results to plot in individual format")
        return
    
    model_map = {
        'cancm': 'CanCM4i',
        'canv2': 'CanSIPSv2', 
        'ic3': 'IC3',
        'nemo': 'NEMO'
    }
    
    issued_months = ['nov', 'dec', 'jan', 'feb', 'mar']
    month_names = ['Nov', 'Dec', 'Jan', 'Feb', 'Mar']
    models = ['canv2', 'cancm', 'ic3', 'nemo']
    
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    fig.suptitle('Individual ROC Curves: Model vs Issued Month', fontsize=18, fontweight='bold', y=0.95)
    
    model_roc_data = {}
    
    for result in results:
        comparison_name = result['comparison_name']
        parts = comparison_name.split('_')
        
        if len(parts) >= 3:
            target_month = parts[0]
            issued_month = parts[1]
            model = parts[-1]
        else:
            continue
        
        if model not in model_roc_data:
            model_roc_data[model] = {}
        
        obs_file = result.get('obs_file')
        pred_file = result.get('pred_file')
        
        if obs_file and pred_file:
            try:
                obs_spi = load_spi_data(obs_file)
                pred_spi = load_spi_data(pred_file)
                
                if obs_spi is not None and pred_spi is not None:
                    if obs_spi.shape != pred_spi.shape:
                        pred_spi = pred_spi.interp(Y=obs_spi.Y, X=obs_spi.X, method='linear')
                    
                    fpr_binary, tpr_binary, auc_binary = calculate_roc_auc_binary(obs_spi, pred_spi)
                    
                    if fpr_binary is not None and tpr_binary is not None and auc_binary is not None:
                        model_roc_data[model][issued_month] = {
                            'fpr': fpr_binary,
                            'tpr': tpr_binary,
                            'auc': auc_binary
                        }
                        print(f"    ✓ Loaded ROC data for {model_map.get(model, model)} {issued_month}")
                    else:
                        print(f"    ⚠ Failed to calculate ROC for {model} {issued_month}")
                else:
                    print(f"    ⚠ Failed to load data for {model} {issued_month}")
            except Exception as e:
                print(f"    ⚠ Error processing {model} {issued_month}: {e}")
    
    for row, model in enumerate(models):
        for col, issued_month in enumerate(issued_months):
            
            ax = axes[row, col]
            
            if row == 0:
                ax.set_title(f'{month_names[col]}', fontsize=12, fontweight='bold', pad=10)
            
            if col == 0:
                ax.set_ylabel(f'{model_map.get(model, model.upper())}', 
                              fontsize=11, fontweight='bold', rotation=90, ha='center', va='center')
            
            if model in model_roc_data and issued_month in model_roc_data[model]:
                roc_data = model_roc_data[model][issued_month]
                fpr = roc_data['fpr']
                tpr = roc_data['tpr']
                auc = roc_data['auc']
                
                ax.plot(fpr, tpr, color='darkred', lw=2.5, alpha=0.9,
                        label=f'ROC curve (AUC = {auc:.3f})')
                
                ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.7, label='Random')
                
                if row == 0 and col == 0:
                    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
                
                ax.text(0.6, 0.2, f'AUC: {auc:.3f}', 
                        fontsize=10, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
            else:
                ax.text(0.5, 0.5, 'No Data', 
                        transform=ax.transAxes, ha='center', va='center',
                        fontsize=12, fontweight='bold', color='red',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.8))
            
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
            ax.tick_params(axis='both', which='major', labelsize=8)
            
            if row == 3:
                ax.set_xlabel('False Positive Rate', fontsize=9)
            if col == 0:
                ax.set_ylabel('True Positive Rate', fontsize=9, rotation=90, ha='center', va='center')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    
    individual_plot_path = os.path.join(output_folder, 'individual_roc_curves_4x5.png')
    plt.savefig(individual_plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Individual ROC curves plot saved to: {individual_plot_path}")

def create_spatial_roc_visualization(results, output_folder, shapefile_path):
    """
    Create a spatial ROC visualization showing ROC curves per grid cell across Java region.
    Layout: 4 rows (models) x 5 columns (issued months)
    Each subplot shows spatial distribution of ROC performance across grid cells.
    """
    if not results:
        print("No results to plot in spatial format")
        return
    
    try:
        java_gdf = gpd.read_file(shapefile_path)
        print(f"✓ Loaded Java shapefile: {shapefile_path}")
    except Exception as e:
        print(f"✗ Error loading shapefile: {e}")
        return
    
    model_map = {
        'cancm': 'CanCM4i',
        'canv2': 'CanSIPSv2', 
        'ic3': 'IC3',
        'nemo': 'NEMO'
    }
    
    issued_months = ['nov', 'dec', 'jan', 'feb', 'mar']
    month_names = ['Nov', 'Dec', 'Jan', 'Feb', 'Mar']
    models = ['canv2', 'cancm', 'ic3', 'nemo']
    
    if CARTOPY_AVAILABLE:
        fig = plt.figure(figsize=(20, 16))
        fig.suptitle('Spatial ROC Analysis: Per-Grid Performance Across Java Region', fontsize=18, fontweight='bold', y=0.95)
    else:
        fig, axes = plt.subplots(4, 5, figsize=(20, 16))
        fig.suptitle('Spatial ROC Analysis: Per-Grid Performance Across Java Region', fontsize=18, fontweight='bold', y=0.95)
    
    spatial_roc_data = {}
    
    for result in results:
        comparison_name = result['comparison_name']
        parts = comparison_name.split('_')
        
        if len(parts) >= 3:
            target_month = parts[0]
            issued_month = parts[1]
            model = parts[-1]
        else:
            continue
        
        if model not in spatial_roc_data:
            spatial_roc_data[model] = {}
        
        obs_file = result.get('obs_file')
        pred_file = result.get('pred_file')
        
        if obs_file and pred_file:
            try:
                obs_spi = load_spi_data(obs_file)
                pred_spi = load_spi_data(pred_file)
                
                if obs_spi is not None and pred_spi is not None:
                    if obs_spi.shape != pred_spi.shape:
                        pred_spi = pred_spi.interp(Y=obs_spi.Y, X=obs_spi.X, method='linear')
                    
                    spatial_auc = calculate_spatial_roc_auc(obs_spi, pred_spi)
                    
                    if spatial_auc is not None:
                        dim_names_local = list(obs_spi.dims)
                        lat_candidates_local = ['Y', 'y', 'lat', 'latitude']
                        lon_candidates_local = ['X', 'x', 'lon', 'longitude']
                        lat_dim_local = next((d for d in dim_names_local if d in lat_candidates_local), None)
                        lon_dim_local = next((d for d in dim_names_local if d in lon_candidates_local), None)
                        lons_vals = obs_spi[lon_dim_local].values if lon_dim_local in obs_spi.coords else np.arange(spatial_auc.shape[1])
                        lats_vals = obs_spi[lat_dim_local].values if lat_dim_local in obs_spi.coords else np.arange(spatial_auc.shape[0])
                        spatial_roc_data[model][issued_month] = {
                            'spatial_auc': spatial_auc,
                            'lons': lons_vals,
                            'lats': lats_vals
                        }
                        print(f"    ✓ Loaded spatial ROC data for {model_map.get(model, model)} {issued_month}")
                    else:
                        print(f"    ⚠ Failed to calculate spatial ROC for {model} {issued_month}")
                else:
                    print(f"    ⚠ Failed to load data for {model} {issued_month}")
            except Exception as e:
                print(f"    ⚠ Error processing {model} {issued_month}: {e}")
    
    for row, model in enumerate(models):
        for col, issued_month in enumerate(issued_months):
            if CARTOPY_AVAILABLE:
                ax = fig.add_subplot(4, 5, row * 5 + col + 1, projection=ccrs.PlateCarree())
            else:
                ax = axes[row, col]
            
            if row == 0:
                ax.set_title(f'{month_names[col]}', fontsize=12, fontweight='bold', pad=10)
            
            if col == 0:
                ax.set_ylabel(f'{model_map.get(model, model.upper())}', 
                              fontsize=11, fontweight='bold', rotation=90, ha='center', va='center')
            
            java_gdf.plot(ax=ax, color='lightgray', edgecolor='black', linewidth=0.5, alpha=0.3)
            
            if model in spatial_roc_data and issued_month in spatial_roc_data[model]:
                spatial_data = spatial_roc_data[model][issued_month]
                spatial_auc = spatial_data['spatial_auc']
                lons = spatial_data['lons']
                lats = spatial_data['lats']
                
                lon2d, lat2d = np.meshgrid(lons, lats)
                data2d = np.asarray(spatial_auc)
                data2d_masked = np.ma.masked_invalid(data2d)
                if CARTOPY_AVAILABLE:
                    img = ax.pcolormesh(lon2d, lat2d, data2d_masked,
                                        cmap='RdYlGn', vmin=0.4, vmax=1.0,
                                        shading='auto', transform=ccrs.PlateCarree())
                else:
                    img = ax.pcolormesh(lon2d, lat2d, data2d_masked,
                                        cmap='RdYlGn', vmin=0.4, vmax=1.0,
                                        shading='auto')
                scatter = img
                
                avg_auc = float(np.nanmean(data2d)) if np.isfinite(np.nanmean(data2d)) else 0.5
                ax.text(0.5, 0.5, f'Avg AUC: {avg_auc:.3f}', 
                        transform=ax.transAxes, ha='center', va='center',
                        fontsize=10, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
            else:
                ax.text(0.5, 0.5, 'No Data', 
                        transform=ax.transAxes, ha='center', va='center',
                        fontsize=12, fontweight='bold', color='red',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.8))
            
            if CARTOPY_AVAILABLE:
                ax.set_extent([105, 115, -9, -5], crs=ccrs.PlateCarree())
                
                ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
                ax.add_feature(cfeature.BORDERS, linewidth=0.5)
                
                gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                                  linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
                
                if row == 3:
                    gl.bottom_labels = True
                    gl.xformatter = LONGITUDE_FORMATTER
                if col == 0:
                    gl.left_labels = True
                    gl.yformatter = LATITUDE_FORMATTER
            else:
                ax.set_xlim(105, 115)
                ax.set_ylim(-9, -5)
                ax.grid(True, alpha=0.3)
                ax.set_aspect('equal')
                
                if row == 3:
                    ax.set_xlabel('Longitude', fontsize=9)
                if col == 0:
                    ax.set_ylabel('Latitude', fontsize=9)
    
    cbar_ax = fig.add_axes([0.15, 0.02, 0.7, 0.02])
    import matplotlib as mpl
    mappable = scatter
    if mappable is None:
        from matplotlib.cm import get_cmap
        cmap = get_cmap('RdYlGn')
        norm = mpl.colors.Normalize(vmin=0.4, vmax=1.0)
        mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(mappable, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('AUC Values (Per Grid Cell)', fontsize=12, fontweight='bold')
    cbar.set_ticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    
    plt.tight_layout(rect=[0, 0.08, 1, 0.93])
    
    spatial_plot_path = os.path.join(output_folder, 'spatial_roc_per_grid_java.png')
    plt.savefig(spatial_plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Spatial ROC per-grid analysis plot saved to: {spatial_plot_path}")

def calculate_spatial_roc_auc(obs_spi, pred_spi):
    """
    Calculate ROC AUC per grid cell by using the time/sample dimension as observations.
    Returns a 2D array (lat, lon) of AUC values.
    If no sample dimension (e.g., time) exists, returns None.
    """
    try:
        dim_names = list(obs_spi.dims)
        lat_candidates = ['Y', 'y', 'lat', 'latitude']
        lon_candidates = ['X', 'x', 'lon', 'longitude']
        sample_candidates = ['time', 'T', 't', 'month', 'date', 'year']
        
        lat_dim = next((d for d in dim_names if d in lat_candidates), None)
        lon_dim = next((d for d in dim_names if d in lon_candidates), None)
        sample_dim = next((d for d in dim_names if d in sample_candidates), None)
        
        if lat_dim is None or lon_dim is None:
            print("   Warning: Could not identify lat/lon dimensions; returning None")
            return None
        if sample_dim is None:
            print("   Warning: No time/sample dimension found; cannot compute per-grid ROC. Returning None.")
            return None
        
        obs_da = obs_spi.transpose(sample_dim, lat_dim, lon_dim)
        pred_da = pred_spi.transpose(sample_dim, lat_dim, lon_dim)
        
        obs_arr = np.asarray(obs_da.values)
        pred_arr = np.asarray(pred_da.values)
        
        valid_mask = ~(np.isnan(obs_arr) | np.isnan(pred_arr))
        
        ny, nx = obs_da.sizes[lat_dim], obs_da.sizes[lon_dim]
        spatial_auc = np.full((ny, nx), 0.5, dtype=float)
        
        for i in range(ny):
            for j in range(nx):
                vmask = valid_mask[:, i, j]
                if np.sum(vmask) < 3:
                    spatial_auc[i, j] = 0.5
                    continue
                y_true = (obs_arr[vmask, i, j] <= 0).astype(int)
                y_scores = pred_arr[vmask, i, j]
                if len(np.unique(y_true)) < 2:
                    spatial_auc[i, j] = 0.5
                    continue
                try:
                    auc_val = roc_auc_score(y_true, y_scores)
                    if np.isnan(auc_val) or np.isinf(auc_val):
                        spatial_auc[i, j] = 0.5
                    else:
                        spatial_auc[i, j] = float(auc_val)
                except Exception:
                    spatial_auc[i, j] = 0.5
        
        return spatial_auc
    
    except Exception as e:
        print(f"Error calculating spatial ROC AUC: {e}")
        return None

def analyze_spi_comparison(obs_spi_path, pred_spi_path, output_folder, comparison_name):
    """
    Analyze SPI comparison between observations and predictions.
    """
    print(f"\n--- Analyzing {comparison_name} ---")
    
    obs_spi = load_spi_data(obs_spi_path)
    pred_spi = load_spi_data(pred_spi_path)
    
    if obs_spi is None or pred_spi is None:
        print(f"✗ Failed to load data for {comparison_name}")
        return None
    
    print(f"Observation SPI shape: {obs_spi.shape}")
    print(f"Prediction SPI shape: {pred_spi.shape}")
    
    print(f"Observation SPI range: [{obs_spi.min().values:.3f}, {obs_spi.max().values:.3f}]")
    print(f"Prediction SPI range: [{pred_spi.min().values:.3f}, {pred_spi.max().values:.3f}]")
    
    if obs_spi.shape != pred_spi.shape:
        print(f"Warning: Shape mismatch. Interpolating prediction to observation grid...")
        pred_spi = pred_spi.interp(Y=obs_spi.Y, X=obs_spi.X, method='linear')
    
    print("Calculating binary drought classification ROC...")
    fpr_binary, tpr_binary, auc_binary = calculate_roc_auc_binary(obs_spi, pred_spi)
    
    if fpr_binary is not None:
        print(f"Binary Drought Classification AUC: {auc_binary:.3f}")
        
        binary_plot_path = os.path.join(output_folder, f'roc_binary_{comparison_name}.png')
        clean_title = format_model_month_name(comparison_name)
        plot_roc_curves(fpr_binary, tpr_binary, auc_binary, 
                        f'Binary Drought Classification - {clean_title}', 
                        binary_plot_path, is_binary=True)
    else:
        print("✗ Failed to calculate binary ROC")
    
    print("Calculating multi-class drought classification ROC...")
    fpr_multiclass, tpr_multiclass, auc_multiclass = calculate_roc_auc_multiclass(obs_spi, pred_spi)
    
    if fpr_multiclass is not None:
        print(f"Multi-class Classification Micro-average AUC: {auc_multiclass['micro']:.3f}")
        
        multiclass_plot_path = os.path.join(output_folder, f'roc_multiclass_{comparison_name}.png')
        clean_title = format_model_month_name(comparison_name)
        plot_roc_curves(fpr_multiclass, tpr_multiclass, auc_multiclass, 
                        f'Multi-class Drought Classification - {clean_title}', 
                        multiclass_plot_path, is_binary=False)
    else:
        print("✗ Failed to calculate multi-class ROC")
    
    return {
        'binary_auc': auc_binary,
        'multiclass_auc': auc_multiclass['micro'] if auc_multiclass else None,
        'comparison_name': comparison_name,
        'obs_file': obs_spi_path,
        'pred_file': pred_spi_path
    }

def main():
    """
    Main function to run ROC and AUC analysis.
    """
    print("=== ROC AND AUC ANALYSIS FOR SPI COMPARISON ===")
    print("Training months: NOV, DEC, JAN, FEB, MAR")
    print("Target period: MJJ (May, June, July)")
    print("Comparing: CNN predictions for MJJ vs Observations for MJJ")
    print()
    
    obs_spi_path = 'D:/Kuliah/Skripsi/Data_baru/Output/SPI_Obs'
    pred_spi_path = 'D:/Kuliah/Skripsi/Data_baru/Output/SPI_CNN'
    output_folder = 'D:/Kuliah/Skripsi/Data_baru/Output/ROC_AUC_Analysis'
    
    os.makedirs(output_folder, exist_ok=True)
    
    obs_files = glob.glob(os.path.join(obs_spi_path, '*.nc'))
    print(f"Found {len(obs_files)} observation SPI files")
    
    pred_files = glob.glob(os.path.join(pred_spi_path, '**/*.nc'), recursive=True)
    print(f"Found {len(pred_files)} prediction SPI files")
    
    print("\nDebug - Found prediction files:")
    for f in pred_files[:10]:
        print(f"  {f}")
    if len(pred_files) > 10:
        print(f"  ... and {len(pred_files) - 10} more files")
    
    if not obs_files or not pred_files:
        print("✗ No SPI files found. Please run SPI calculation scripts first.")
        return
    
    results = []
    
    target_months = ['may', 'jun', 'jul']
    issued_months = ['nov', 'dec', 'jan', 'feb', 'mar']
    
    print(f"Target Months: {', '.join([m.upper() for m in target_months])}")
    print(f"Issued Months: {', '.join([m.upper() for m in issued_months])}")
    
    for obs_file in obs_files:
        obs_month = os.path.basename(obs_file).split('_')[1].lower()
        
        if obs_month not in target_months:
            print(f"⚠ Skipping {obs_month.upper()} - not in target period MJJ")
            continue
        
        print(f"\n--- Processing target month: {obs_month.upper()} ---")
        
        for issued_month in issued_months:
            print(f"\n  📆 Processing issued month: {issued_month.upper()}")
            
            issued_pred_files = [f for f in pred_files if f'/{issued_month}/' in f or f'\\{issued_month}\\' in f]
            
            print(f"     Looking for files containing: /{issued_month}/ or \\{issued_month}\\")
            print(f"     Found {len(issued_pred_files)} files for {issued_month.upper()}")
            if issued_pred_files:
                for f in issued_pred_files:
                    print(f"        ✓ {f}")
            
            if not issued_pred_files:
                print(f"     ⚠ No prediction files found for {issued_month.upper()}")
                continue
            
            for pred_file in issued_pred_files:
                filename = os.path.basename(pred_file)
                
                if 'spi_' in filename:
                    model_name = filename.replace('spi_', '').replace('.nc', '')
                else:
                    model_name = filename.replace('.nc', '')
                
                comparison_name = f"{obs_month.upper()}_{issued_month}_{model_name}"
                
                print(f"     🔍 Analyzing: {model_name} ({issued_month.upper()})")
                
                result = analyze_spi_comparison(obs_file, pred_file, output_folder, comparison_name)
                
                if result:
                    results.append(result)
    
    print("\n" + "="*60)
    print("ROC AND AUC ANALYSIS SUMMARY")
    print("="*60)
    
    if results:
        model_averages = {}
        month_averages = {}
        
        for result in results:
            comparison_name = result['comparison_name']
            parts = comparison_name.split('_')
            
            if len(parts) >= 3:
                target_month = parts[0]
                issued_month = parts[1]
                model = parts[-1]
            else:
                target_month = parts[0]
                issued_month = target_month
                model = parts[-1]
            
            if model not in model_averages:
                model_averages[model] = {'binary_aucs': [], 'multiclass_aucs': []}
            if result['binary_auc'] is not None:
                model_averages[model]['binary_aucs'].append(result['binary_auc'])
            if result['multiclass_auc'] is not None:
                model_averages[model]['multiclass_aucs'].append(result['multiclass_auc'])
            
            if issued_month not in month_averages:
                month_averages[issued_month] = {'binary_aucs': [], 'multiclass_aucs': []}
            if result['binary_auc'] is not None:
                month_averages[issued_month]['binary_aucs'].append(result['binary_auc'])
            if result['multiclass_auc'] is not None:
                month_averages[issued_month]['multiclass_aucs'].append(result['multiclass_auc'])
        
        print("\n📊 AVERAGE PERFORMANCE BY MODEL:")
        print(f"{'Model':<15} {'Binary AUC':<12} {'Multi-class AUC':<15} {'Samples':<8}")
        print("-" * 60)
        
        for model, data in model_averages.items():
            binary_avg = np.mean(data['binary_aucs']) if data['binary_aucs'] else 0
            multiclass_avg = np.mean(data['multiclass_aucs']) if data['multiclass_aucs'] else 0
            samples = len(data['binary_aucs'])
            print(f"{model:<15} {binary_avg:.3f}        {multiclass_avg:.3f}              {samples}")
        
        print("\n📅 AVERAGE PERFORMANCE BY ISSUED MONTH:")
        print(f"{'Month':<12} {'Binary AUC':<12} {'Multi-class AUC':<15} {'Samples':<8}")
        print("-" * 55)
        
        month_order = ['nov', 'dec', 'jan', 'feb', 'mar']
        for month in month_order:
            if month in month_averages:
                data = month_averages[month]
                binary_avg = np.mean(data['binary_aucs']) if data['binary_aucs'] else 0
                multiclass_avg = np.mean(data['multiclass_aucs']) if data['multiclass_aucs'] else 0
                samples = len(data['binary_aucs'])
                month_name = month.upper()
                print(f"{month_name:<12} {binary_avg:.3f}        {multiclass_avg:.3f}              {samples}")
        
        all_binary_aucs = [r['binary_auc'] for r in results if r['binary_auc'] is not None]
        all_multiclass_aucs = [r['multiclass_auc'] for r in results if r['multiclass_auc'] is not None]
        
        print("\n🎯 OVERALL AVERAGE PERFORMANCE:")
        if all_binary_aucs:
            overall_binary = np.mean(all_binary_aucs)
            print(f"Overall Binary Drought Classification AUC: {overall_binary:.3f}")
        if all_multiclass_aucs:
            overall_multiclass = np.mean(all_multiclass_aucs)
            print(f"Overall Multi-class Classification AUC: {overall_multiclass:.3f}")
        
        best_model = max(model_averages.keys(), 
                         key=lambda m: np.mean(model_averages[m]['binary_aucs']) if model_averages[m]['binary_aucs'] else 0)
        best_month = max(month_averages.keys(), 
                         key=lambda m: np.mean(month_averages[m]['binary_aucs']) if month_averages[m]['binary_aucs'] else 0)
        
        print(f"\n🏆 BEST PERFORMING:")
        print(f"Model: {best_model} (Binary AUC: {np.mean(model_averages[best_model]['binary_aucs']):.3f})")
        print(f"Month: {best_month} (Binary AUC: {np.mean(month_averages[best_month]['binary_aucs']):.3f})")
        
        results_file = os.path.join(output_folder, 'roc_auc_consolidated_results.txt')
        with open(results_file, 'w') as f:
            f.write("ROC AND AUC ANALYSIS - CONSOLIDATED RESULTS\n")
            f.write("="*50 + "\n\n")
            
            f.write("AVERAGE PERFORMANCE BY MODEL:\n")
            f.write("-" * 30 + "\n")
            for model, data in model_averages.items():
                binary_avg = np.mean(data['binary_aucs']) if data['binary_aucs'] else 0
                multiclass_avg = np.mean(data['multiclass_aucs']) if data['multiclass_aucs'] else 0
                f.write(f"{model}: Binary AUC = {binary_avg:.3f}, Multi-class AUC = {multiclass_avg:.3f}\n")
            
            f.write("\nAVERAGE PERFORMANCE BY MONTH:\n")
            f.write("-" * 30 + "\n")
            for month, data in month_averages.items():
                binary_avg = np.mean(data['binary_aucs']) if data['binary_aucs'] else 0
                multiclass_avg = np.mean(data['multiclass_aucs']) if data['multiclass_aucs'] else 0
                f.write(f"{month}: Binary AUC = {binary_avg:.3f}, Multi-class AUC = {multiclass_avg:.3f}\n")
            
            f.write(f"\nOVERALL AVERAGES:\n")
            f.write("-" * 20 + "\n")
            if all_binary_aucs:
                f.write(f"Binary AUC: {np.mean(all_binary_aucs):.3f}\n")
            if all_multiclass_aucs:
                f.write(f"Multi-class AUC: {np.mean(all_multiclass_aucs):.3f}\n")
        
        print(f"\n📁 Consolidated results saved to: {results_file}")
        
        print("\n📊 Creating 2x2 grid ROC plot...")
        create_grid_roc_plot(results, output_folder)
        
        print("\n📊 Creating comprehensive 5x4 ROC plot...")
        create_comprehensive_5x4_plot(results, output_folder)
        
        print("\n📊 Creating individual ROC curves (4x5 grid)...")
        create_individual_roc_plots(results, output_folder)
        
        print("\n🗺️ Creating spatial ROC visualization...")
        shapefile_path = 'D:/Kuliah/Skripsi/shp/jawa.shp'
        create_spatial_roc_visualization(results, output_folder, shapefile_path)
        
    else:
        print("✗ No successful comparisons completed.")
    
    print(f"\nAll outputs saved to: {output_folder}")
    print("=== ANALYSIS COMPLETED ===")

if __name__ == "__main__":
    main()