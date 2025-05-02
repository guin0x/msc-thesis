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

def coupled_perturbation(wspd, wdir, seed=None):
    """
    Perturb wind speed and direction using a coupled error model.
    
    Implements:
    speed_error = 0.1*wspd + 0.25*max(0,wspd-15)
    dir_error = max(5, 20-wspd/2)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Compute error magnitudes based on wind speed
    speed_error = 0.1 * wspd + 0.25 * np.maximum(0, wspd - 15)
    dir_error = np.maximum(5, 20 - wspd / 2)
    
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
        
        # Compute phi values
        phi_perturbed = compute_phi(wdir_perturbed, azimuth_look)
        phi_nominal = compute_phi(era5_wdir, azimuth_look)
        
        # Forward modeling with CMOD5N
        sigma_cmod = cmod5n_forward(wspd_perturbed, phi_perturbed, incidence)
        
        # Spectral processing
        fft_sar, psd_sar, kx_sar, ky_sar, kmag_sar = compute_2d_fft(sigma_sar)
        fft_cmod, psd_cmod, kx_cmod, ky_cmod, kmag_cmod = compute_2d_fft(sigma_cmod)
        
        # Band filtering for SAR
        band0_sar = band_filter(fft_sar, kmag_sar, 0, 0.1)
        band1_sar = band_filter(fft_sar, kmag_sar, 0.1, 0.3)
        band2_sar = band_filter(fft_sar, kmag_sar, 0.3, np.inf)
        
        # Band filtering for CMOD
        band0_cmod = band_filter(fft_cmod, kmag_cmod, 0, 0.1)
        band1_cmod = band_filter(fft_cmod, kmag_cmod, 0.1, 0.3)
        band2_cmod = band_filter(fft_cmod, kmag_cmod, 0.3, np.inf)
        
        
        # CMOD inversion for each band
        wspd_band0 = cmod5n_inverse(band0_sar, phi_nominal, incidence)
        wspd_band1 = cmod5n_inverse(band1_sar, phi_nominal, incidence)
        wspd_band2 = cmod5n_inverse(band2_sar, phi_nominal, incidence)

        # Calculate error metrics for each band
        errors_band0 = calculate_error_metrics(wspd_band0, era5_wspd)
        errors_band1 = calculate_error_metrics(wspd_band1, era5_wspd)
        errors_band2 = calculate_error_metrics(wspd_band2, era5_wspd)
        
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
