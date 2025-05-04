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
    # Break signals into segments and apply windowing
    # For demonstration, assuming segmentation has been applied
    
    # Calculate cross-spectral density
    cross_spectrum = np.mean(fft_obs * np.conj(fft_model), axis=0)
    
    # Calculate auto-spectral densities
    power_spectrum_obs = np.mean(np.abs(fft_obs)**2, axis=0)
    power_spectrum_model = np.mean(np.abs(fft_model)**2, axis=0)
    
    # Calculate coherence
    coherence_squared = np.abs(cross_spectrum)**2 / (power_spectrum_obs * power_spectrum_model)
    
    # Calculate band-specific coherence
    band_coherence = {}
    for band_name, (kmin, kmax) in band_ranges.items():
        band_mask = (kmagnitude >= kmin) & (kmagnitude < kmax)
        band_coherence[band_name] = np.mean(coherence_squared[band_mask])
    
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

def perform_statistical_tests(df_results):
    """
    Perform statistical tests to determine if there are significant differences
    across wavenumber bands for various metrics.
    
    Parameters:
    -----------
    df_results : pandas.DataFrame
        DataFrame containing the analysis results
        
    Returns:
    --------
    dict
        Dictionary with test results for different metrics
    """
    # Initialize results dictionary
    test_results = {}
    
    # 1. Transfer function ratios
    ratio0 = df_results['ratio_band0']
    ratio1 = df_results['ratio_band1']
    ratio2 = df_results['ratio_band2']
    
    # Check if we have at least two different values before running Kruskal-Wallis
    ratio_values = np.concatenate([ratio0.dropna().values, ratio1.dropna().values, ratio2.dropna().values])
    if len(np.unique(ratio_values)) <= 1:
        # All values are identical, can't run Kruskal-Wallis
        test_results['transfer_ratio'] = {
            'statistic': np.nan,
            'p_value': 1.0,
            'significant': False
        }
    else:
        # Kruskal-Wallis test for ratios
        try:
            ratio_statistic, ratio_p_value = kruskal(
                ratio0.dropna(), 
                ratio1.dropna(), 
                ratio2.dropna()
            )
            
            test_results['transfer_ratio'] = {
                'statistic': ratio_statistic,
                'p_value': ratio_p_value,
                'significant': ratio_p_value < 0.05
            }
        except ValueError as e:
            print(f"Warning: Error in transfer_ratio Kruskal-Wallis test: {e}")
            test_results['transfer_ratio'] = {
                'statistic': np.nan,
                'p_value': 1.0,
                'significant': False
            }
    
    # Pairwise tests for ratios using Mann-Whitney U
    ratio_pairs = []
    test_results['transfer_ratio_pairwise'] = {}
    
    # Only do pairwise tests if we have different values
    if len(np.unique(ratio_values)) > 1:
        try:
            ratio_pairs = [
                ('band0_band1', stats.mannwhitneyu(ratio0.dropna(), ratio1.dropna())),
                ('band0_band2', stats.mannwhitneyu(ratio0.dropna(), ratio2.dropna())),
                ('band1_band2', stats.mannwhitneyu(ratio1.dropna(), ratio2.dropna()))
            ]
            
            # Apply Bonferroni correction for multiple comparisons
            p_values = [pair[1].pvalue for pair in ratio_pairs]
            reject, p_adjusted, _, _ = multipletests(p_values, method='bonferroni')
            
            test_results['transfer_ratio_pairwise'] = {
                pair[0]: {
                    'statistic': pair[1].statistic,
                    'p_value': pair[1].pvalue,
                    'p_adjusted': p_adj,
                    'significant': rej
                }
                for pair, p_adj, rej in zip(ratio_pairs, p_adjusted, reject)
            }
        except Exception as e:
            print(f"Warning: Error in transfer_ratio pairwise tests: {e}")
            for pair_name in ['band0_band1', 'band0_band2', 'band1_band2']:
                test_results['transfer_ratio_pairwise'][pair_name] = {
                    'statistic': np.nan,
                    'p_value': 1.0,
                    'p_adjusted': 1.0,
                    'significant': False
                }
    else:
        for pair_name in ['band0_band1', 'band0_band2', 'band1_band2']:
            test_results['transfer_ratio_pairwise'][pair_name] = {
                'statistic': np.nan,
                'p_value': 1.0,
                'p_adjusted': 1.0,
                'significant': False
            }
    
    # 2. Spectral coherence
    coherence0 = df_results['coherence_metrics'].apply(lambda x: x['band0'])
    coherence1 = df_results['coherence_metrics'].apply(lambda x: x['band1'])
    coherence2 = df_results['coherence_metrics'].apply(lambda x: x['band2'])
    
    # Check if we have at least two different values before running Kruskal-Wallis
    coherence_values = np.concatenate([coherence0.dropna().values, coherence1.dropna().values, coherence2.dropna().values])
    if len(np.unique(coherence_values)) <= 1:
        # All values are identical, can't run Kruskal-Wallis
        test_results['spectral_coherence'] = {
            'statistic': np.nan,
            'p_value': 1.0,
            'significant': False
        }
    else:
        # Kruskal-Wallis test for coherence
        try:
            coherence_statistic, coherence_p_value = kruskal(
                coherence0.dropna(),
                coherence1.dropna(),
                coherence2.dropna()
            )
            
            test_results['spectral_coherence'] = {
                'statistic': coherence_statistic,
                'p_value': coherence_p_value,
                'significant': coherence_p_value < 0.05
            }
        except ValueError as e:
            print(f"Warning: Error in spectral_coherence Kruskal-Wallis test: {e}")
            test_results['spectral_coherence'] = {
                'statistic': np.nan,
                'p_value': 1.0,
                'significant': False
            }
    
    # Pairwise tests for coherence
    test_results['spectral_coherence_pairwise'] = {}
    
    # Only do pairwise tests if we have different values
    if len(np.unique(coherence_values)) > 1:
        try:
            coherence_pairs = [
                ('band0_band1', stats.mannwhitneyu(coherence0.dropna(), coherence1.dropna())),
                ('band0_band2', stats.mannwhitneyu(coherence0.dropna(), coherence2.dropna())),
                ('band1_band2', stats.mannwhitneyu(coherence1.dropna(), coherence2.dropna()))
            ]
            
            p_values = [pair[1].pvalue for pair in coherence_pairs]
            reject, p_adjusted, _, _ = multipletests(p_values, method='bonferroni')
            
            test_results['spectral_coherence_pairwise'] = {
                pair[0]: {
                    'statistic': pair[1].statistic,
                    'p_value': pair[1].pvalue,
                    'p_adjusted': p_adj,
                    'significant': rej
                }
                for pair, p_adj, rej in zip(coherence_pairs, p_adjusted, reject)
            }
        except Exception as e:
            print(f"Warning: Error in spectral_coherence pairwise tests: {e}")
            for pair_name in ['band0_band1', 'band0_band2', 'band1_band2']:
                test_results['spectral_coherence_pairwise'][pair_name] = {
                    'statistic': np.nan,
                    'p_value': 1.0,
                    'p_adjusted': 1.0,
                    'significant': False
                }
    else:
        for pair_name in ['band0_band1', 'band0_band2', 'band1_band2']:
            test_results['spectral_coherence_pairwise'][pair_name] = {
                'statistic': np.nan,
                'p_value': 1.0,
                'p_adjusted': 1.0,
                'significant': False
            }
    
    # 3. Sensitivity metrics
    sensitivity0 = df_results['sensitivity_metrics'].apply(lambda x: x['band0'])
    sensitivity1 = df_results['sensitivity_metrics'].apply(lambda x: x['band1'])
    sensitivity2 = df_results['sensitivity_metrics'].apply(lambda x: x['band2'])
    
    # Check if we have at least two different values before running Kruskal-Wallis
    sensitivity_values = np.concatenate([sensitivity0.dropna().values, sensitivity1.dropna().values, sensitivity2.dropna().values])
    if len(np.unique(sensitivity_values)) <= 1:
        # All values are identical, can't run Kruskal-Wallis
        test_results['sensitivity'] = {
            'statistic': np.nan,
            'p_value': 1.0,
            'significant': False
        }
    else:
        # Kruskal-Wallis test for sensitivity
        try:
            sensitivity_statistic, sensitivity_p_value = kruskal(
                sensitivity0.dropna(),
                sensitivity1.dropna(),
                sensitivity2.dropna()
            )
            
            test_results['sensitivity'] = {
                'statistic': sensitivity_statistic,
                'p_value': sensitivity_p_value,
                'significant': sensitivity_p_value < 0.05
            }
        except ValueError as e:
            print(f"Warning: Error in sensitivity Kruskal-Wallis test: {e}")
            test_results['sensitivity'] = {
                'statistic': np.nan,
                'p_value': 1.0,
                'significant': False
            }
    
    # Pairwise tests for sensitivity
    test_results['sensitivity_pairwise'] = {}
    
    # Only do pairwise tests if we have different values
    if len(np.unique(sensitivity_values)) > 1:
        try:
            sensitivity_pairs = [
                ('band0_band1', stats.mannwhitneyu(sensitivity0.dropna(), sensitivity1.dropna())),
                ('band0_band2', stats.mannwhitneyu(sensitivity0.dropna(), sensitivity2.dropna())),
                ('band1_band2', stats.mannwhitneyu(sensitivity1.dropna(), sensitivity2.dropna()))
            ]
            
            p_values = [pair[1].pvalue for pair in sensitivity_pairs]
            reject, p_adjusted, _, _ = multipletests(p_values, method='bonferroni')
            
            test_results['sensitivity_pairwise'] = {
                pair[0]: {
                    'statistic': pair[1].statistic,
                    'p_value': pair[1].pvalue,
                    'p_adjusted': p_adj,
                    'significant': rej
                }
                for pair, p_adj, rej in zip(sensitivity_pairs, p_adjusted, reject)
            }
        except Exception as e:
            print(f"Warning: Error in sensitivity pairwise tests: {e}")
            for pair_name in ['band0_band1', 'band0_band2', 'band1_band2']:
                test_results['sensitivity_pairwise'][pair_name] = {
                    'statistic': np.nan,
                    'p_value': 1.0,
                    'p_adjusted': 1.0,
                    'significant': False
                }
    else:
        for pair_name in ['band0_band1', 'band0_band2', 'band1_band2']:
            test_results['sensitivity_pairwise'][pair_name] = {
                'statistic': np.nan,
                'p_value': 1.0,
                'p_adjusted': 1.0,
                'significant': False
            }
    
    # 4. Cross-scale analysis
    bias0 = df_results['errors_band0'].apply(lambda x: x['bias'])
    bias1 = df_results['errors_band1'].apply(lambda x: x['bias'])
    bias2 = df_results['errors_band2'].apply(lambda x: x['bias'])
    cross_bias_model0_obs12 = df_results['errors_cross_model0_obs12'].apply(lambda x: x['bias'])
    cross_bias_obs0_model12 = df_results['errors_cross_obs0_model12'].apply(lambda x: x['bias'])
    
    # Check if we have at least two different values before running Kruskal-Wallis
    cross_scale_values = np.concatenate([
        bias0.dropna().values, 
        bias1.dropna().values, 
        bias2.dropna().values,
        cross_bias_model0_obs12.dropna().values,
        cross_bias_obs0_model12.dropna().values
    ])
    
    if len(np.unique(cross_scale_values)) <= 1:
        # All values are identical, can't run Kruskal-Wallis
        test_results['cross_scale'] = {
            'statistic': np.nan,
            'p_value': 1.0,
            'significant': False
        }
    else:
        # Kruskal-Wallis test for cross-scale analysis
        try:
            cross_scale_statistic, cross_scale_p_value = kruskal(
                bias0.dropna(),
                bias1.dropna(),
                bias2.dropna(),
                cross_bias_model0_obs12.dropna(),
                cross_bias_obs0_model12.dropna()
            )
            
            test_results['cross_scale'] = {
                'statistic': cross_scale_statistic,
                'p_value': cross_scale_p_value,
                'significant': cross_scale_p_value < 0.05
            }
        except ValueError as e:
            print(f"Warning: Error in cross_scale Kruskal-Wallis test: {e}")
            test_results['cross_scale'] = {
                'statistic': np.nan,
                'p_value': 1.0,
                'significant': False
            }
    
    # Pairwise tests focusing on cross-scale combinations vs regular bands
    test_results['cross_scale_pairwise'] = {}
    
    # Only do pairwise tests if we have different values
    if len(np.unique(cross_scale_values)) > 1:
        try:
            cross_scale_pairs = [
                ('band0_model0_obs12', stats.mannwhitneyu(bias0.dropna(), cross_bias_model0_obs12.dropna())),
                ('band0_obs0_model12', stats.mannwhitneyu(bias0.dropna(), cross_bias_obs0_model12.dropna())),
                ('model0_obs12_obs0_model12', stats.mannwhitneyu(cross_bias_model0_obs12.dropna(), cross_bias_obs0_model12.dropna()))
            ]
            
            p_values = [pair[1].pvalue for pair in cross_scale_pairs]
            reject, p_adjusted, _, _ = multipletests(p_values, method='bonferroni')
            
            test_results['cross_scale_pairwise'] = {
                pair[0]: {
                    'statistic': pair[1].statistic,
                    'p_value': pair[1].pvalue,
                    'p_adjusted': p_adj,
                    'significant': rej
                }
                for pair, p_adj, rej in zip(cross_scale_pairs, p_adjusted, reject)
            }
        except Exception as e:
            print(f"Warning: Error in cross_scale pairwise tests: {e}")
            for pair_name in ['band0_model0_obs12', 'band0_obs0_model12', 'model0_obs12_obs0_model12']:
                test_results['cross_scale_pairwise'][pair_name] = {
                    'statistic': np.nan,
                    'p_value': 1.0,
                    'p_adjusted': 1.0,
                    'significant': False
                }
    else:
        for pair_name in ['band0_model0_obs12', 'band0_obs0_model12', 'model0_obs12_obs0_model12']:
            test_results['cross_scale_pairwise'][pair_name] = {
                'statistic': np.nan,
                'p_value': 1.0,
                'p_adjusted': 1.0,
                'significant': False
            }
    
    return test_results

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
    
def generate_statistical_report(test_results, output_path):
    """
    Generate a report of statistical test results.
    
    Parameters:
    -----------
    test_results : dict
        Dictionary with test results from perform_statistical_tests
    output_path : Path
        Path to save the report
    """
    with open(output_path / "statistical_tests_report.txt", "w") as f:
        f.write("Statistical Tests for Scale-Dependent Analysis\n")
        f.write("=" * 60 + "\n\n")
        
        # 1. Transfer function ratios
        f.write("1. Transfer Function Ratios\n")
        f.write("-" * 40 + "\n")
        ratio_result = test_results['transfer_ratio']
        f.write(f"Kruskal-Wallis Test: H={ratio_result['statistic']:.4f}, p={ratio_result['p_value']:.6f}\n")
        
        if ratio_result['significant']:
            f.write("SIGNIFICANT: There are statistically significant differences in the transfer function ratios across bands.\n\n")
        else:
            f.write("NOT SIGNIFICANT: No statistically significant differences in the transfer function ratios across bands.\n\n")
        
        f.write("Pairwise comparisons (Mann-Whitney U with Bonferroni correction):\n")
        for pair, result in test_results['transfer_ratio_pairwise'].items():
            f.write(f"  {pair}: U={result['statistic']:.4f}, p={result['p_value']:.6f}, adj. p={result['p_adjusted']:.6f}, {'SIGNIFICANT' if result['significant'] else 'not significant'}\n")
        f.write("\n")
        
        # 2. Spectral coherence
        f.write("2. Spectral Coherence\n")
        f.write("-" * 40 + "\n")
        coherence_result = test_results['spectral_coherence']
        f.write(f"Kruskal-Wallis Test: H={coherence_result['statistic']:.4f}, p={coherence_result['p_value']:.6f}\n")
        
        if coherence_result['significant']:
            f.write("SIGNIFICANT: There are statistically significant differences in the spectral coherence across bands.\n\n")
        else:
            f.write("NOT SIGNIFICANT: No statistically significant differences in the spectral coherence across bands.\n\n")
        
        f.write("Pairwise comparisons (Mann-Whitney U with Bonferroni correction):\n")
        for pair, result in test_results['spectral_coherence_pairwise'].items():
            f.write(f"  {pair}: U={result['statistic']:.4f}, p={result['p_value']:.6f}, adj. p={result['p_adjusted']:.6f}, {'SIGNIFICANT' if result['significant'] else 'not significant'}\n")
        f.write("\n")
        
        # 3. Sensitivity metrics
        f.write("3. Scale-Dependent Sensitivity\n")
        f.write("-" * 40 + "\n")
        sensitivity_result = test_results['sensitivity']
        f.write(f"Kruskal-Wallis Test: H={sensitivity_result['statistic']:.4f}, p={sensitivity_result['p_value']:.6f}\n")
        
        if sensitivity_result['significant']:
            f.write("SIGNIFICANT: There are statistically significant differences in the sensitivity metrics across bands.\n\n")
        else:
            f.write("NOT SIGNIFICANT: No statistically significant differences in the sensitivity metrics across bands.\n\n")
        
        f.write("Pairwise comparisons (Mann-Whitney U with Bonferroni correction):\n")
        for pair, result in test_results['sensitivity_pairwise'].items():
            f.write(f"  {pair}: U={result['statistic']:.4f}, p={result['p_value']:.6f}, adj. p={result['p_adjusted']:.6f}, {'SIGNIFICANT' if result['significant'] else 'not significant'}\n")
        f.write("\n")
        
        # 4. Cross-scale analysis
        f.write("4. Cross-Scale Analysis\n")
        f.write("-" * 40 + "\n")
        cross_scale_result = test_results['cross_scale']
        f.write(f"Kruskal-Wallis Test: H={cross_scale_result['statistic']:.4f}, p={cross_scale_result['p_value']:.6f}\n")
        
        if cross_scale_result['significant']:
            f.write("SIGNIFICANT: There are statistically significant differences in the bias between regular bands and cross-scale combinations.\n\n")
        else:
            f.write("NOT SIGNIFICANT: No statistically significant differences in the bias between regular bands and cross-scale combinations.\n\n")
        
        f.write("Pairwise comparisons (Mann-Whitney U with Bonferroni correction):\n")
        for pair, result in test_results['cross_scale_pairwise'].items():
            f.write(f"  {pair}: U={result['statistic']:.4f}, p={result['p_value']:.6f}, adj. p={result['p_adjusted']:.6f}, {'SIGNIFICANT' if result['significant'] else 'not significant'}\n")
        f.write("\n")
        
        # Summary of findings
        f.write("Summary of Statistical Findings\n")
        f.write("-" * 40 + "\n")
        significant_findings = []
        
        if test_results['transfer_ratio']['significant']:
            significant_findings.append("Transfer function ratios show scale dependence")
        
        if test_results['spectral_coherence']['significant']:
            significant_findings.append("Spectral coherence shows scale dependence")
        
        if test_results['sensitivity']['significant']:
            significant_findings.append("Sensitivity metrics show scale dependence")
        
        if test_results['cross_scale']['significant']:
            significant_findings.append("Cross-scale analysis shows significant differences between bands")
        
        if significant_findings:
            f.write("The following metrics show statistically significant scale dependence:\n")
            for finding in significant_findings:
                f.write(f"- {finding}\n")
        else:
            f.write("None of the analyzed metrics showed statistically significant scale dependence.\n")

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


def plot_statistical_test_results(test_results, output_path):
    """
    Generate visualization of statistical test results.
    
    Parameters:
    -----------
    test_results : dict
        Dictionary with test results from perform_statistical_tests
    output_path : Path
        Path to save the visualization
    """
    # Extract p-values for each test
    tests = ['Transfer Function', 'Spectral Coherence', 'Sensitivity', 'Cross-Scale']
    p_values = [
        test_results['transfer_ratio']['p_value'],
        test_results['spectral_coherence']['p_value'],
        test_results['sensitivity']['p_value'],
        test_results['cross_scale']['p_value']
    ]
    
    # Check for NaN values
    valid_tests = [not np.isnan(p) for p in p_values]
    
    if not any(valid_tests):
        print("Warning: No valid statistical tests to plot")
        return
    
    # Filter out NaN values
    valid_tests_names = [test for test, valid in zip(tests, valid_tests) if valid]
    valid_p_values = [p for p, valid in zip(p_values, valid_tests) if valid]
    significant = [p < 0.05 for p in valid_p_values]
    
    # Create bar chart
    plt.figure(figsize=(12, 8))
    bars = plt.bar(valid_tests_names, valid_p_values, color=['green' if sig else 'red' for sig in significant])
    
    # Add reference line for significance threshold
    plt.axhline(y=0.05, color='black', linestyle='--', alpha=0.7, label='Significance Threshold (p=0.05)')
    
    # Add labels and title
    plt.ylabel('p-value')
    plt.title('Statistical Test Results for Scale Dependency')
    plt.grid(alpha=0.3)
    plt.yscale('log')  # Use log scale for better visualization
    plt.legend()
    
    # Add p-values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'p={height:.4f}', ha='center', va='bottom', 
                 rotation=0 if height > 0.1 else 0)
    
    # Add significance indicators
    for i, sig in enumerate(significant):
        plt.text(i, 0.001, 'SIGNIFICANT' if sig else 'not significant',
                ha='center', va='bottom', rotation=90, alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_path / 'statistical_test_results.png', dpi=300)
    plt.close()
    
    # Also create a pairwise comparisons visualization
    # Plot pairwise comparison results for each test that was significant
    for test_name, test_key, valid in zip(
        tests, 
        ['transfer_ratio', 'spectral_coherence', 'sensitivity', 'cross_scale'],
        valid_tests
    ):
        if valid and test_results[test_key]['significant']:
            pairwise_key = f"{test_key}_pairwise"
            if pairwise_key in test_results:
                pairs = list(test_results[pairwise_key].keys())
                p_adj_values = [test_results[pairwise_key][pair]['p_adjusted'] for pair in pairs]
                
                # Check for NaN values in pairwise tests
                valid_pairs = [not np.isnan(p) for p in p_adj_values]
                if not any(valid_pairs):
                    continue
                
                valid_pair_names = [pair for pair, valid in zip(pairs, valid_pairs) if valid]
                valid_p_adj_values = [p for p, valid in zip(p_adj_values, valid_pairs) if valid]
                significant_pairs = [p < 0.05 for p in valid_p_adj_values]
                
                plt.figure(figsize=(10, 6))
                bars = plt.bar(valid_pair_names, valid_p_adj_values, 
                              color=['green' if sig else 'red' for sig in significant_pairs])
                
                plt.axhline(y=0.05, color='black', linestyle='--', alpha=0.7, 
                           label='Significance Threshold (p=0.05)')
                plt.ylabel('Adjusted p-value')
                plt.title(f'Pairwise Comparisons for {test_name}')
                plt.grid(alpha=0.3)
                plt.yscale('log')
                plt.legend()
                
                # Add p-values on top of bars
                for bar in bars:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                            f'p={height:.4f}', ha='center', va='bottom')
                
                plt.tight_layout()
                plt.savefig(output_path / f'{test_key}_pairwise_tests.png', dpi=300)
                plt.close()