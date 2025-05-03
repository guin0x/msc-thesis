import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import kruskal
from statsmodels.stats.multitest import multipletests

def read_sar_data(filepath):
    """Read SAR data from a given filepath."""
    try:
        ds = xr.open_dataset(filepath, engine='h5netcdf')
        return ds
    except Exception as e:
        print(f"Error reading SAR data: {e}")
        return None

def coupled_perturbation(wspd, wdir, seed=None, factor=1.0):
    """
    Perturb wind speed and direction using a coupled error model.
    
    Implements:
    speed_error = 0.1*wspd + 0.25*max(0,wspd-15)
    dir_error = max(5, 20-wspd/2)
    
    Parameters:
    -----------
    wspd : float
        Wind speed to perturb
    wdir : float
        Wind direction to perturb
    seed : int, optional
        Random seed for reproducibility
    factor : float, optional
        Scaling factor for perturbation magnitude
    
    Returns:
    --------
    wspd_perturbed, wdir_perturbed : tuple of float
        Perturbed wind speed and direction
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Compute error magnitudes based on wind speed
    speed_error = factor * (0.1 * wspd + 0.25 * np.maximum(0, wspd - 15))
    dir_error = factor * np.maximum(5, 20 - wspd / 2)
    
    # Generate random errors with appropriate standard deviations
    wspd_noise = np.random.normal(0, speed_error)
    wdir_noise = np.random.normal(0, dir_error)
    
    # Apply perturbations
    wspd_perturbed = wspd + wspd_noise
    wspd_perturbed = np.maximum(0, wspd_perturbed)  # Ensure non-negative wind speed
    wdir_perturbed = np.mod(wdir + wdir_noise, 360)  # Keep direction in [0, 360)
    
    return wspd_perturbed, wdir_perturbed

def compute_phi(wdir, azimuth_look):
    """Compute phi (relative wind direction) from wind direction and azimuth look angle."""
    phi = wdir - azimuth_look
    # Wrap to [-180, 180]
    phi = np.mod(phi + 180, 360) - 180
    return phi

def cmod5n_forward(wspd, phi, incidence):
    """CMOD5N forward model to compute sigma0 from wind parameters."""
    try:
        from utils.cmod5n import cmod5n_forward
        return cmod5n_forward(np.full(phi.shape, wspd), phi, incidence)
    except ImportError:
        try:
            from utils.cmod5n import cmod5n_forward
            return cmod5n_forward(np.full(phi.shape, wspd), phi, incidence)
        except ImportError:
            print("Warning: No CMOD5N module found.")
            # Simplified approximation
            phi_rad = np.deg2rad(phi)
            inc_rad = np.deg2rad(incidence)
            return 0.1 * (1 + np.cos(phi_rad)) * wspd**0.5 * np.exp(-0.1 * inc_rad)

def cmod5n_inverse(sigma0, phi, incidence):
    """CMOD5N inverse model to compute wind speed from sigma0."""
    try:
        from utils.cmod5n import cmod5n_inverse
        return cmod5n_inverse(sigma0, phi, incidence)
    except ImportError:
        try:
            from utils.cmod5n import cmod5n_inverse
            return cmod5n_inverse(sigma0, phi, incidence,)
        except ImportError:
            print("Warning: No CMOD5N module found.")
            # Simplified approximation
            phi_rad = np.deg2rad(phi)
            inc_rad = np.deg2rad(incidence)
            return np.sqrt(sigma0 / (0.1 * (1 + np.cos(phi_rad)) * np.exp(-0.1 * inc_rad)))

def compute_2d_fft(sigma0):
    """Compute 2D FFT of sigma0 values."""
    # Clean NaN values
    sigma0_clean = sigma0.copy()
    if np.isnan(sigma0_clean).any():
        sigma0_clean[np.isnan(sigma0_clean)] = 0
    
    # Compute 2D FFT
    fft_data = np.fft.fft2(sigma0_clean)
    psd_2d = np.abs(fft_data)**2
    
    # Compute wavenumbers
    freqx = np.fft.fftfreq(sigma0.shape[1])
    freqy = np.fft.fftfreq(sigma0.shape[0])
    kx, ky = np.meshgrid(freqx, freqy)
    kmagnitude = np.sqrt(kx**2 + ky**2)
    
    return fft_data, psd_2d, kx, ky, kmagnitude

def band_filter(fft_data, kmagnitude, kmin, kmax):
    """Apply band-pass filter to FFT data."""
    # Create mask for band-pass filter
    mask = (kmagnitude >= kmin) & (kmagnitude < kmax)
    
    # Apply filter in frequency domain
    fft_filtered = np.zeros_like(fft_data, dtype=complex)
    fft_filtered[mask] = fft_data[mask]
    
    # Invert back to spatial domain
    filtered_sigma0 = np.real(np.fft.ifft2(fft_filtered))
    
    return filtered_sigma0

def calculate_error_metrics(retrieved_wspd, true_wspd):
    """Calculate error metrics between retrieved and true wind speeds."""
    # Handle NaN values
    
    if hasattr(retrieved_wspd, 'flatten'):
        retrieved_wspd = retrieved_wspd.flatten()
    
    mask = ~np.isnan(retrieved_wspd)
    retrieved_wspd_clean = retrieved_wspd[mask]
    
    if isinstance(true_wspd, (int, float)):
        true_wspd_clean = np.full_like(retrieved_wspd_clean, true_wspd)
    else:
        if hasattr(true_wspd, 'flatten'):
            true_wspd = true_wspd.flatten()
        true_wspd_clean = true_wspd[mask]
    
    # Calculate errors
    error = retrieved_wspd_clean - true_wspd_clean
    abs_error = np.abs(error)
    rel_error = error / true_wspd_clean
    
    # Calculate metrics
    bias = np.mean(error)
    rmse = np.sqrt(np.mean(error**2))
    mean_abs_error = np.mean(abs_error)
    mean_rel_error = np.mean(rel_error)
    
    return {
        'bias': bias,
        'rmse': rmse,
        'abs_error': mean_abs_error,
        'rel_error': mean_rel_error
    }

def calculate_sigma0_comparison_metrics(obs_sigma0, model_sigma0):
    """Calculate comparison metrics between observed and modeled sigma0."""
    # Handle NaN values
    if hasattr(obs_sigma0, 'flatten'):
        obs_sigma0 = obs_sigma0.flatten()
    if hasattr(model_sigma0, 'flatten'):
        model_sigma0 = model_sigma0.flatten()
    
    mask = ~(np.isnan(obs_sigma0) | np.isnan(model_sigma0))
    obs_sigma0_clean = obs_sigma0[mask]
    model_sigma0_clean = model_sigma0[mask]
    
    # Calculate metrics
    diff = obs_sigma0_clean - model_sigma0_clean
    
    # Handle potential division by zero in ratio calculation
    ratio = np.zeros_like(obs_sigma0_clean)
    nonzero_mask = obs_sigma0_clean != 0
    ratio[nonzero_mask] = model_sigma0_clean[nonzero_mask] / obs_sigma0_clean[nonzero_mask]
    
    # Filter out inf and nan in ratio
    ratio = ratio[~np.isinf(ratio) & ~np.isnan(ratio)]
    
    # Calculate statistics
    bias = np.mean(diff)
    rmse = np.sqrt(np.mean(diff**2))
    mean_ratio = np.mean(ratio)
    
    return {
        'bias': bias,
        'rmse': rmse,
        'mean_ratio': mean_ratio
    }

def calculate_spectral_coherence(obs_sigma0, model_sigma0, kmagnitude, band_ranges):
    """Calculate spectral coherence between observed and modeled sigma0 for different bands."""
    # Compute FFTs
    fft_obs = np.fft.fft2(np.nan_to_num(obs_sigma0))
    fft_model = np.fft.fft2(np.nan_to_num(model_sigma0))
    
    # Compute coherence
    coherence = np.abs(fft_obs * np.conj(fft_model))
    coherence_norm = np.sqrt(np.abs(fft_obs)**2 * np.abs(fft_model)**2)
    
    # Avoid division by zero
    mask = coherence_norm > 0
    normalized_coherence = np.zeros_like(coherence)
    normalized_coherence[mask] = coherence[mask] / coherence_norm[mask]
    
    # Calculate band-specific coherence
    band_coherence = {}
    for band_name, (kmin, kmax) in band_ranges.items():
        band_mask = (kmagnitude >= kmin) & (kmagnitude < kmax)
        band_coherence[band_name] = np.mean(normalized_coherence[band_mask])
    
    return band_coherence

def calculate_scale_dependent_sensitivity(band0_cmod, band1_cmod, band2_cmod, 
                                         band0_cmod_strong, band1_cmod_strong, band2_cmod_strong,
                                         wspd_perturbed, wspd_perturbed_strong):
    """Calculate scale-dependent sensitivity of modeled sigma0 to wind changes."""
    # Calculate wind speed difference
    wspd_diff = wspd_perturbed_strong - wspd_perturbed
    
    # Calculate sensitivity for each band
    sensitivity_band0 = np.nanmean((band0_cmod_strong - band0_cmod)) / wspd_diff
    sensitivity_band1 = np.nanmean((band1_cmod_strong - band1_cmod)) / wspd_diff
    sensitivity_band2 = np.nanmean((band2_cmod_strong - band2_cmod)) / wspd_diff
    
    return {
        'band0': sensitivity_band0,
        'band1': sensitivity_band1,
        'band2': sensitivity_band2
    }

def kruskal_wallis_test(errors_band0, errors_band1, errors_band2):
    """Perform Kruskal-Wallis test to detect significant differences across bands."""
    # Remove NaN values
    errors_band0 = np.array(errors_band0)[~np.isnan(errors_band0)]
    errors_band1 = np.array(errors_band1)[~np.isnan(errors_band1)]
    errors_band2 = np.array(errors_band2)[~np.isnan(errors_band2)]
    
    # Perform Kruskal-Wallis test
    statistic, p_value = kruskal(errors_band0, errors_band1, errors_band2)
    
    return statistic, p_value, p_value < 0.05

def process_sar_file(sar_filepath, era5_wspd, era5_wdir, seed=None):
    """Process a single SAR file according to the workflow."""
    try:
        # Read SAR data
        sar_ds = read_sar_data(sar_filepath)
        if sar_ds is None:
            return None
        
        # Extract SAR data
        sigma_sar = sar_ds.sigma0.values
        incidence = sar_ds.incidence.values
        ground_heading = sar_ds.ground_heading.values

        if sigma_sar.ndim == 3:
            sigma_sar = sigma_sar[0]  # Take first slice if 3D
        
        # Clean NaN rows/columns
        if np.isnan(sigma_sar[-1, :]).all():
            sigma_sar = sigma_sar[:-1, :]
            incidence = incidence[:-1, :]
            ground_heading = ground_heading[:-1, :]

        if np.isnan(sigma_sar[:, -1]).all():
            sigma_sar = sigma_sar[:, :-1]
            incidence = incidence[:, :-1]
            ground_heading = ground_heading[:, :-1]
                
        # Normalize ground_heading to 0-360
        ground_heading = np.mod(ground_heading, 360)
        
        # Calculate azimuth look angle (Sentinel-1 is right-looking)
        azimuth_look = np.mod(ground_heading + 90, 360)
        
        # Perturb wind speed and direction
        wspd_perturbed, wdir_perturbed = coupled_perturbation(era5_wspd, era5_wdir, seed)
        
        # Create additional stronger perturbation for sensitivity analysis
        wspd_perturbed_strong, wdir_perturbed_strong = coupled_perturbation(era5_wspd, era5_wdir, seed, factor=2.0)
        
        # Compute phi values
        phi_perturbed = compute_phi(wdir_perturbed, azimuth_look)
        phi_perturbed_strong = compute_phi(wdir_perturbed_strong, azimuth_look)
        phi_nominal = compute_phi(era5_wdir, azimuth_look)
        
        # Forward modeling with CMOD5N
        sigma_cmod = cmod5n_forward(wspd_perturbed, phi_perturbed, incidence)
        sigma_cmod_strong = cmod5n_forward(wspd_perturbed_strong, phi_perturbed_strong, incidence)
        
        # Spectral processing
        fft_sar, psd_sar, kx_sar, ky_sar, kmag_sar = compute_2d_fft(sigma_sar)
        fft_cmod, psd_cmod, kx_cmod, ky_cmod, kmag_cmod = compute_2d_fft(sigma_cmod)
        fft_cmod_strong, _, _, _, kmag_cmod_strong = compute_2d_fft(sigma_cmod_strong)
        
        # Band filtering for SAR
        band0_sar = band_filter(fft_sar, kmag_sar, 0, 0.1)
        band1_sar = band_filter(fft_sar, kmag_sar, 0.1, 0.3)
        band2_sar = band_filter(fft_sar, kmag_sar, 0.3, np.inf)
        
        # Band filtering for CMOD
        band0_cmod = band_filter(fft_cmod, kmag_cmod, 0, 0.1)
        band1_cmod = band_filter(fft_cmod, kmag_cmod, 0.1, 0.3)
        band2_cmod = band_filter(fft_cmod, kmag_cmod, 0.3, np.inf)
        
        # Band filtering for strong perturbation CMOD
        band0_cmod_strong = band_filter(fft_cmod_strong, kmag_cmod_strong, 0, 0.1)
        band1_cmod_strong = band_filter(fft_cmod_strong, kmag_cmod_strong, 0.1, 0.3)
        band2_cmod_strong = band_filter(fft_cmod_strong, kmag_cmod_strong, 0.3, np.inf)
        
        # CMOD inversion for each band
        wspd_band0 = cmod5n_inverse(band0_sar, phi_nominal, incidence)
        wspd_band1 = cmod5n_inverse(band1_sar, phi_nominal, incidence)
        wspd_band2 = cmod5n_inverse(band2_sar, phi_nominal, incidence)
        
        # Calculate error metrics for each band
        errors_band0 = calculate_error_metrics(wspd_band0, era5_wspd)
        errors_band1 = calculate_error_metrics(wspd_band1, era5_wspd)
        errors_band2 = calculate_error_metrics(wspd_band2, era5_wspd)
        
        # Calculate direct sigma0 comparison metrics
        sigma0_diff_band0 = calculate_sigma0_comparison_metrics(band0_sar, band0_cmod)
        sigma0_diff_band1 = calculate_sigma0_comparison_metrics(band1_sar, band1_cmod)
        sigma0_diff_band2 = calculate_sigma0_comparison_metrics(band2_sar, band2_cmod)
        
        # Calculate transfer function ratios
        ratio_band0 = np.nanmean(band0_cmod / np.where(band0_sar == 0, np.nan, band0_sar))
        ratio_band1 = np.nanmean(band1_cmod / np.where(band1_sar == 0, np.nan, band1_sar))
        ratio_band2 = np.nanmean(band2_cmod / np.where(band2_sar == 0, np.nan, band2_sar))
        
        # Calculate spectral coherence
        band_ranges = {
            'band0': (0, 0.1),
            'band1': (0.1, 0.3),
            'band2': (0.3, np.inf)
        }
        coherence_metrics = calculate_spectral_coherence(sigma_sar, sigma_cmod, kmag_sar, band_ranges)
        
        # Calculate scale-dependent sensitivity
        sensitivity_metrics = calculate_scale_dependent_sensitivity(
            band0_cmod, band1_cmod, band2_cmod,
            band0_cmod_strong, band1_cmod_strong, band2_cmod_strong,
            wspd_perturbed, wspd_perturbed_strong
        )
        
        # Cross-scale impact analysis
        # Use band0 from model and bands 1&2 from observations
        mixed_sigma0_model0_obs12 = band0_cmod + band1_sar + band2_sar
        wspd_model0_obs12 = cmod5n_inverse(mixed_sigma0_model0_obs12, phi_nominal, incidence)
        errors_cross_model0_obs12 = calculate_error_metrics(wspd_model0_obs12, era5_wspd)
        
        # Use band0 from observations and bands 1&2 from model
        mixed_sigma0_obs0_model12 = band0_sar + band1_cmod + band2_cmod
        wspd_obs0_model12 = cmod5n_inverse(mixed_sigma0_obs0_model12, phi_nominal, incidence)
        errors_cross_obs0_model12 = calculate_error_metrics(wspd_obs0_model12, era5_wspd)
        
        # Statistical testing for each file
        band0_errors = wspd_band0.flatten() - era5_wspd
        band1_errors = wspd_band1.flatten() - era5_wspd
        band2_errors = wspd_band2.flatten() - era5_wspd
        
        statistic, p_value, is_significant = kruskal_wallis_test(
            band0_errors[~np.isnan(band0_errors)],
            band1_errors[~np.isnan(band1_errors)],
            band2_errors[~np.isnan(band2_errors)]
        )
        
        # Return results
        return {
            'sar_filepath': sar_filepath,
            'era5_wspd': era5_wspd,
            'era5_wdir': era5_wdir,
            'wspd_perturbed': wspd_perturbed,
            'wdir_perturbed': wdir_perturbed,
            'phi_perturbed': np.median(phi_perturbed),
            'phi_nominal': np.median(phi_nominal),
            'sigma_sar_median': np.median(sigma_sar),
            'sigma_cmod_median': np.median(sigma_cmod),
            'band0_wspd_mean': np.nanmean(wspd_band0),
            'band0_wspd_median': np.nanmedian(wspd_band0),
            'band1_wspd_mean': np.nanmean(wspd_band1),
            'band1_wspd_median': np.nanmedian(wspd_band1),
            'band2_wspd_mean': np.nanmean(wspd_band2),
            'band2_wspd_median': np.nanmedian(wspd_band2),
            'errors_band0': errors_band0,
            'errors_band1': errors_band1,
            'errors_band2': errors_band2,
            # New scale dependency analysis metrics
            'sigma0_diff_band0': sigma0_diff_band0,
            'sigma0_diff_band1': sigma0_diff_band1,
            'sigma0_diff_band2': sigma0_diff_band2,
            'ratio_band0': ratio_band0,
            'ratio_band1': ratio_band1,
            'ratio_band2': ratio_band2,
            'coherence_metrics': coherence_metrics,
            'sensitivity_metrics': sensitivity_metrics,
            'errors_cross_model0_obs12': errors_cross_model0_obs12,
            'errors_cross_obs0_model12': errors_cross_obs0_model12,
            'kw_statistic': statistic,
            'kw_p_value': p_value,
            'is_scale_dependent': is_significant
        }
        
    except Exception as e:
        print(f"Error processing SAR file {sar_filepath}: {e}")
        return None

def plot_error_distributions(df_results, output_dir):
    """Plot error distributions for different bands."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Extracting error metrics for each band
    bias0 = df_results['errors_band0'].apply(lambda x: x['bias'])
    bias1 = df_results['errors_band1'].apply(lambda x: x['bias'])
    bias2 = df_results['errors_band2'].apply(lambda x: x['bias'])
    
    rmse0 = df_results['errors_band0'].apply(lambda x: x['rmse'])
    rmse1 = df_results['errors_band1'].apply(lambda x: x['rmse'])
    rmse2 = df_results['errors_band2'].apply(lambda x: x['rmse'])
    
    rel0 = df_results['errors_band0'].apply(lambda x: x['rel_error'])
    rel1 = df_results['errors_band1'].apply(lambda x: x['rel_error'])
    rel2 = df_results['errors_band2'].apply(lambda x: x['rel_error'])
    
    # Plot bias
    plt.figure(figsize=(10, 6))
    plt.hist(bias0, bins=30, alpha=0.7, label='Band 0 (k < 0.1)')
    plt.hist(bias1, bins=30, alpha=0.7, label='Band 1 (0.1 ≤ k < 0.3)')
    plt.hist(bias2, bins=30, alpha=0.7, label='Band 2 (k ≥ 0.3)')
    plt.xlabel('Bias (m/s)')
    plt.ylabel('Frequency')
    plt.title('Bias Distribution by Wavenumber Band')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(output_dir / 'bias_distribution.png', dpi=300)
    plt.close()
    
    # Plot RMSE
    plt.figure(figsize=(10, 6))
    plt.hist(rmse0, bins=30, alpha=0.7, label='Band 0 (k < 0.1)')
    plt.hist(rmse1, bins=30, alpha=0.7, label='Band 1 (0.1 ≤ k < 0.3)')
    plt.hist(rmse2, bins=30, alpha=0.7, label='Band 2 (k ≥ 0.3)')
    plt.xlabel('RMSE (m/s)')
    plt.ylabel('Frequency')
    plt.title('RMSE Distribution by Wavenumber Band')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(output_dir / 'rmse_distribution.png', dpi=300)
    plt.close()
    
    # Boxplot for all metrics
    fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
    
    # Bias boxplot
    axes[0].boxplot([bias0, bias1, bias2], labels=['Band 0', 'Band 1', 'Band 2'])
    axes[0].set_title('Bias by Wavenumber Band')
    axes[0].set_ylabel('Bias (m/s)')
    axes[0].grid(alpha=0.3)
    
    # RMSE boxplot
    axes[1].boxplot([rmse0, rmse1, rmse2], labels=['Band 0', 'Band 1', 'Band 2'])
    axes[1].set_title('RMSE by Wavenumber Band')
    axes[1].set_ylabel('RMSE (m/s)')
    axes[1].grid(alpha=0.3)
    
    # Relative error boxplot
    axes[2].boxplot([rel0, rel1, rel2], labels=['Band 0', 'Band 1', 'Band 2'])
    axes[2].set_title('Relative Error by Wavenumber Band')
    axes[2].set_xlabel('Wavenumber Band')
    axes[2].set_ylabel('Relative Error')
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'error_metrics_boxplot.png', dpi=300)
    plt.close()
    
    # Plot new enhanced metrics for scale dependency analysis
    
    # Direct sigma0 comparison
    sigma0_bias0 = df_results['sigma0_diff_band0'].apply(lambda x: x['bias'])
    sigma0_bias1 = df_results['sigma0_diff_band1'].apply(lambda x: x['bias'])
    sigma0_bias2 = df_results['sigma0_diff_band2'].apply(lambda x: x['bias'])
    
    plt.figure(figsize=(10, 6))
    plt.boxplot([sigma0_bias0, sigma0_bias1, sigma0_bias2], labels=['Band 0', 'Band 1', 'Band 2'])
    plt.title('Direct Sigma0 Bias by Wavenumber Band')
    plt.ylabel('Sigma0 Bias')
    plt.grid(alpha=0.3)
    plt.savefig(output_dir / 'sigma0_bias_boxplot.png', dpi=300)
    plt.close()
    
    # Model-to-Observed Ratio
    ratio0 = df_results['ratio_band0']
    ratio1 = df_results['ratio_band1']
    ratio2 = df_results['ratio_band2']
    
    plt.figure(figsize=(10, 6))
    plt.boxplot([ratio0, ratio1, ratio2], labels=['Band 0', 'Band 1', 'Band 2'])
    plt.title('Model-to-Observed Sigma0 Ratio by Wavenumber Band')
    plt.ylabel('Ratio')
    plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.5)  # Reference line at ratio=1
    plt.grid(alpha=0.3)
    plt.savefig(output_dir / 'sigma0_ratio_boxplot.png', dpi=300)
    plt.close()
    
    # Spectral Coherence
    coherence0 = df_results['coherence_metrics'].apply(lambda x: x['band0'])
    coherence1 = df_results['coherence_metrics'].apply(lambda x: x['band1'])
    coherence2 = df_results['coherence_metrics'].apply(lambda x: x['band2'])
    
    plt.figure(figsize=(10, 6))
    plt.boxplot([coherence0, coherence1, coherence2], labels=['Band 0', 'Band 1', 'Band 2'])
    plt.title('Spectral Coherence by Wavenumber Band')
    plt.ylabel('Coherence')
    plt.grid(alpha=0.3)
    plt.savefig(output_dir / 'spectral_coherence_boxplot.png', dpi=300)
    plt.close()
    
    # Scale-Dependent Sensitivity
    sensitivity0 = df_results['sensitivity_metrics'].apply(lambda x: x['band0'])
    sensitivity1 = df_results['sensitivity_metrics'].apply(lambda x: x['band1'])
    sensitivity2 = df_results['sensitivity_metrics'].apply(lambda x: x['band2'])
    
    plt.figure(figsize=(10, 6))
    plt.boxplot([sensitivity0, sensitivity1, sensitivity2], labels=['Band 0', 'Band 1', 'Band 2'])
    plt.title('Scale-Dependent Sensitivity by Wavenumber Band')
    plt.ylabel('Sensitivity (∆sigma0/∆wspd)')
    plt.grid(alpha=0.3)
    plt.savefig(output_dir / 'scale_sensitivity_boxplot.png', dpi=300)
    plt.close()
    
    # Cross-Scale Impact Analysis
    cross_bias_model0_obs12 = df_results['errors_cross_model0_obs12'].apply(lambda x: x['bias'])
    cross_bias_obs0_model12 = df_results['errors_cross_obs0_model12'].apply(lambda x: x['bias'])
    cross_rmse_model0_obs12 = df_results['errors_cross_model0_obs12'].apply(lambda x: x['rmse'])
    cross_rmse_obs0_model12 = df_results['errors_cross_obs0_model12'].apply(lambda x: x['rmse'])
    
    plt.figure(figsize=(12, 8))
    labels = ['Regular Band 0', 'Regular Band 1', 'Regular Band 2', 'Model0+Obs12', 'Obs0+Model12']
    plt.boxplot([bias0, bias1, bias2, cross_bias_model0_obs12, cross_bias_obs0_model12], labels=labels)
    plt.title('Cross-Scale Impact Analysis: Bias')
    plt.ylabel('Bias (m/s)')
    plt.grid(alpha=0.3)
    plt.savefig(output_dir / 'cross_scale_bias_boxplot.png', dpi=300)
    plt.close()
    
    plt.figure(figsize=(12, 8))
    plt.boxplot([rmse0, rmse1, rmse2, cross_rmse_model0_obs12, cross_rmse_obs0_model12], labels=labels)
    plt.title('Cross-Scale Impact Analysis: RMSE')
    plt.ylabel('RMSE (m/s)')
    plt.grid(alpha=0.3)
    plt.savefig(output_dir / 'cross_scale_rmse_boxplot.png', dpi=300)
    plt.close()
    
    # Summary plot of all scale-dependent metrics
    # Create a summary figure with multiple subplots
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    
    # Direct sigma0 comparison
    axs[0, 0].boxplot([sigma0_bias0, sigma0_bias1, sigma0_bias2], labels=['Band 0', 'Band 1', 'Band 2'])
    axs[0, 0].set_title('Direct Sigma0 Bias')
    axs[0, 0].set_ylabel('Bias')
    axs[0, 0].grid(alpha=0.3)
    
    # Model-to-Observed Ratio
    axs[0, 1].boxplot([ratio0, ratio1, ratio2], labels=['Band 0', 'Band 1', 'Band 2'])
    axs[0, 1].set_title('Model-to-Observed Ratio')
    axs[0, 1].set_ylabel('Ratio')
    axs[0, 1].axhline(y=1.0, color='r', linestyle='--', alpha=0.5)  # Reference line at ratio=1
    axs[0, 1].grid(alpha=0.3)
    
    # Spectral Coherence
    axs[1, 0].boxplot([coherence0, coherence1, coherence2], labels=['Band 0', 'Band 1', 'Band 2'])
    axs[1, 0].set_title('Spectral Coherence')
    axs[1, 0].set_ylabel('Coherence')
    axs[1, 0].grid(alpha=0.3)
    
    # Cross-Scale Analysis
    axs[1, 1].boxplot([bias0, bias1, bias2, cross_bias_model0_obs12, cross_bias_obs0_model12], labels=labels)
    axs[1, 1].set_title('Wind Retrieval Bias')
    axs[1, 1].set_ylabel('Bias (m/s)')
    axs[1, 1].grid(alpha=0.3)
    axs[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'scale_dependency_summary.png', dpi=300)
    plt.close()