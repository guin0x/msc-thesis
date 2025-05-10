import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import kruskal
from statsmodels.stats.multitest import multipletests
import matplotlib as mpl
from matplotlib.ticker import LogLocator

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
    
    speed_error = factor * (0.1 * wspd + 0.25 * np.maximum(0, wspd - 15))
    dir_error = factor * np.maximum(5, 20 - wspd / 2)
    
    wspd_noise = np.random.normal(0, speed_error)
    wdir_noise = np.random.normal(0, dir_error)
    
    wspd_perturbed = wspd + wspd_noise
    wspd_perturbed = np.maximum(0, wspd_perturbed) 
    wdir_perturbed = np.mod(wdir + wdir_noise, 360)
    
    return wspd_perturbed, wdir_perturbed

def compute_phi(wdir, azimuth_look):
    """Compute phi (relative wind direction) from wind direction and azimuth look angle."""
    phi = wdir - azimuth_look
    phi = np.mod(phi + 180, 360) - 180
    return phi

def cmod5n_forward(wspd, phi, incidence):
    """CMOD5N forward model to compute sigma0 from wind parameters."""
    from utils.cmod5n import cmod5n_forward
    return cmod5n_forward(np.full(phi.shape, wspd), phi, incidence)

def cmod5n_inverse(sigma0, phi, incidence):
    """CMOD5N inverse model to compute wind speed from sigma0."""
    from utils.cmod5n import cmod5n_inverse
    return cmod5n_inverse(sigma0, phi, incidence)

def compute_2d_fft(sigma0):
    """Compute 2D FFT of sigma0 values."""

    sigma0_clean = sigma0.copy()
    if np.isnan(sigma0_clean).any():
        sigma0_clean[np.isnan(sigma0_clean)] = 0
    
    fft_data = np.fft.fft2(sigma0_clean)
    psd_2d = np.abs(fft_data)**2
    
    freqx = np.fft.fftfreq(sigma0.shape[1])
    freqy = np.fft.fftfreq(sigma0.shape[0])
    kx, ky = np.meshgrid(freqx, freqy)
    kmagnitude = np.sqrt(kx**2 + ky**2)
    
    return fft_data, psd_2d, kx, ky, kmagnitude

def band_filter(fft_data, kmagnitude, kmin, kmax):
    """Apply band-pass filter to FFT data."""
    
    mask = (kmagnitude >= kmin) & (kmagnitude < kmax)
    
    fft_filtered = np.zeros_like(fft_data, dtype=complex)
    fft_filtered[mask] = fft_data[mask]
    
    filtered_sigma0 = np.real(np.fft.ifft2(fft_filtered))
    
    return filtered_sigma0

def calculate_error_metrics(retrieved_wspd, true_wspd):
    """Calculate error metrics between retrieved and true wind speeds."""

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
    
    error = retrieved_wspd_clean - true_wspd_clean
    abs_error = np.abs(error)
    rel_error = error / true_wspd_clean
    
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
    if hasattr(obs_sigma0, 'flatten'):
        obs_sigma0 = obs_sigma0.flatten()
    if hasattr(model_sigma0, 'flatten'):
        model_sigma0 = model_sigma0.flatten()
    
    mask = ~(np.isnan(obs_sigma0) | np.isnan(model_sigma0))
    obs_sigma0_clean = obs_sigma0[mask]
    model_sigma0_clean = model_sigma0[mask]
    
    diff = obs_sigma0_clean - model_sigma0_clean
    
    ratio = np.zeros_like(obs_sigma0_clean)
    nonzero_mask = obs_sigma0_clean != 0
    ratio[nonzero_mask] = model_sigma0_clean[nonzero_mask] / obs_sigma0_clean[nonzero_mask]
    
    ratio = ratio[~np.isinf(ratio) & ~np.isnan(ratio)]
    
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
    fft_obs = np.fft.fft2(np.nan_to_num(obs_sigma0))
    fft_model = np.fft.fft2(np.nan_to_num(model_sigma0))
    
    coherence = np.abs(fft_obs * np.conj(fft_model))
    coherence_norm = np.sqrt(np.abs(fft_obs)**2 * np.abs(fft_model)**2)
    
    mask = coherence_norm > 0
    normalized_coherence = np.zeros_like(coherence)
    normalized_coherence[mask] = coherence[mask] / coherence_norm[mask]
    
    band_coherence = {}
    for band_name, (kmin, kmax) in band_ranges.items():
        band_mask = (kmagnitude >= kmin) & (kmagnitude < kmax)
        band_coherence[band_name] = np.mean(normalized_coherence[band_mask])
    
    return band_coherence

def calculate_scale_dependent_sensitivity(bands_cmod, wspd_perturbed, wspd_perturbed_strong):
    """Calculate scale-dependent sensitivity of modeled sigma0 to wind changes."""
    wspd_diff = wspd_perturbed_strong - wspd_perturbed
    
    sensitivities = {}
    
    for band_name, (cmod, cmod_strong) in bands_cmod.items():
        sensitivity = np.nanmean((cmod_strong - cmod)) / wspd_diff
        sensitivities[band_name] = sensitivity
    
    return sensitivities

def kruskal_wallis_test(*error_bands):
    """Perform Kruskal-Wallis test to detect significant differences across bands."""
    cleaned_bands = []
    for band in error_bands:
        cleaned_band = np.array(band)[~np.isnan(band)]
        cleaned_bands.append(cleaned_band)
    
    statistic, p_value = kruskal(*cleaned_bands)
    
    return statistic, p_value, p_value < 0.05

def perform_statistical_tests(df_results, band_names):
    """
    Perform statistical tests to determine if there are significant differences
    across wavenumber bands for various metrics.
    
    Parameters:
    -----------
    df_results : pandas.DataFrame
        DataFrame containing the analysis results
    band_names : list
        List of band names
        
    Returns:
    --------
    dict
        Dictionary with test results for different metrics
    """
    test_results = {}
    
    ratios = [df_results[f'ratio_{band_name}'] for band_name in band_names]
    
    ratio_values = np.concatenate([ratio.dropna().values for ratio in ratios])
    if len(np.unique(ratio_values)) <= 1:
        test_results['transfer_ratio'] = {
            'statistic': np.nan,
            'p_value': 1.0,
            'significant': False
        }
    else:
        try:
            ratio_statistic, ratio_p_value = kruskal(*[ratio.dropna() for ratio in ratios])
            
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
    
    ratio_pairs = []
    test_results['transfer_ratio_pairwise'] = {}
    
    if len(np.unique(ratio_values)) > 1:
        try:
            ratio_pairs = []
            for i in range(len(band_names)):
                for j in range(i+1, len(band_names)):
                    pair_name = f'{band_names[i]}_{band_names[j]}'
                    pair_result = stats.mannwhitneyu(ratios[i].dropna(), ratios[j].dropna())
                    ratio_pairs.append((pair_name, pair_result))
            
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
            for i in range(len(band_names)):
                for j in range(i+1, len(band_names)):
                    pair_name = f'{band_names[i]}_{band_names[j]}'
                    test_results['transfer_ratio_pairwise'][pair_name] = {
                        'statistic': np.nan,
                        'p_value': 1.0,
                        'p_adjusted': 1.0,
                        'significant': False
                    }
    else:
        for i in range(len(band_names)):
            for j in range(i+1, len(band_names)):
                pair_name = f'{band_names[i]}_{band_names[j]}'
                test_results['transfer_ratio_pairwise'][pair_name] = {
                    'statistic': np.nan,
                    'p_value': 1.0,
                    'p_adjusted': 1.0,
                    'significant': False
                }
    
    sensitivities = [df_results['sensitivity_metrics'].apply(lambda x: x[band_name]) for band_name in band_names]
    
    sensitivity_values = np.concatenate([sens.dropna().values for sens in sensitivities])
    if len(np.unique(sensitivity_values)) <= 1:
        test_results['sensitivity'] = {
            'statistic': np.nan,
            'p_value': 1.0,
            'significant': False
        }
    else:
        try:
            sensitivity_statistic, sensitivity_p_value = kruskal(*[sens.dropna() for sens in sensitivities])
            
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
    
    test_results['sensitivity_pairwise'] = {}
    
    if len(np.unique(sensitivity_values)) > 1:
        try:
            sensitivity_pairs = []
            for i in range(len(band_names)):
                for j in range(i+1, len(band_names)):
                    pair_name = f'{band_names[i]}_{band_names[j]}'
                    pair_result = stats.mannwhitneyu(sensitivities[i].dropna(), sensitivities[j].dropna())
                    sensitivity_pairs.append((pair_name, pair_result))
            
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
            for i in range(len(band_names)):
                for j in range(i+1, len(band_names)):
                    pair_name = f'{band_names[i]}_{band_names[j]}'
                    test_results['sensitivity_pairwise'][pair_name] = {
                        'statistic': np.nan,
                        'p_value': 1.0,
                        'p_adjusted': 1.0,
                        'significant': False
                    }
    else:
        for i in range(len(band_names)):
            for j in range(i+1, len(band_names)):
                pair_name = f'{band_names[i]}_{band_names[j]}'
                test_results['sensitivity_pairwise'][pair_name] = {
                    'statistic': np.nan,
                    'p_value': 1.0,
                    'p_adjusted': 1.0,
                    'significant': False
                }
    
    biases = [df_results[f'errors_{band_name}'].apply(lambda x: x['bias']) for band_name in band_names]

    mix1_bias = df_results['errors_mix1'].apply(lambda x: x['bias'])
    mix2_bias = df_results['errors_mix2'].apply(lambda x: x['bias'])
    mix3_bias = df_results['errors_mix3'].apply(lambda x: x['bias'])
    mix4_bias = df_results['errors_mix4'].apply(lambda x: x['bias'])

    cross_scale_values = np.concatenate([
        *[bias.dropna().values for bias in biases],
        mix1_bias.dropna().values,
        mix2_bias.dropna().values,
        mix3_bias.dropna().values,
        mix4_bias.dropna().values
    ])
    
    if len(np.unique(cross_scale_values)) <= 1:
        test_results['cross_scale'] = {
            'statistic': np.nan,
            'p_value': 1.0,
            'significant': False
        }
    else:
        try:
            cross_scale_statistic, cross_scale_p_value = kruskal(
            *[bias.dropna() for bias in biases],
            mix1_bias.dropna(),
            mix2_bias.dropna(),
            mix3_bias.dropna(),
            mix4_bias.dropna()
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
    
    return test_results

def plot_focused_analysis(df_results, output_dir):
    """
    Plot the most relevant analyses for scale dependency: transfer function ratios,
    scale-dependent sensitivity, and cross-scale impact.
    
    Parameters:
    -----------
    df_results : pandas.DataFrame
        DataFrame containing the analysis results
    output_dir : Path
        Path to save the plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    band_labels = ['Band 0a (k < 0.0333)', 'Band 0b (0.0333 ≤ k < 0.0666)', 
                   'Band 0c (0.0666 ≤ k < 0.1)', 'Band 1 (0.1 ≤ k < 0.3)', 
                   'Band 2 (k ≥ 0.3)']
    
    ratio0a = df_results['ratio_band0a']
    ratio0b = df_results['ratio_band0b']
    ratio0c = df_results['ratio_band0c']
    ratio1 = df_results['ratio_band1']
    ratio2 = df_results['ratio_band2']
    
    plt.figure(figsize=(12, 6))
    plt.boxplot([ratio0a, ratio0b, ratio0c, ratio1, ratio2], labels=band_labels)
    plt.title('Transfer Function Ratios by Wavenumber Band')
    plt.ylabel('Model-to-Observed Ratio')
    plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.5)  # Reference line at ratio=1
    plt.grid(alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / 'transfer_function_ratios.png', dpi=300)
    plt.close()
    
    sensitivity0a = df_results['sensitivity_metrics'].apply(lambda x: x['band0a'])
    sensitivity0b = df_results['sensitivity_metrics'].apply(lambda x: x['band0b'])
    sensitivity0c = df_results['sensitivity_metrics'].apply(lambda x: x['band0c'])
    sensitivity1 = df_results['sensitivity_metrics'].apply(lambda x: x['band1'])
    sensitivity2 = df_results['sensitivity_metrics'].apply(lambda x: x['band2'])
    
    plt.figure(figsize=(12, 6))
    plt.boxplot([sensitivity0a, sensitivity0b, sensitivity0c, sensitivity1, sensitivity2], labels=band_labels)
    plt.title('Scale-Dependent Sensitivity by Wavenumber Band')
    plt.ylabel('Sensitivity (∆sigma0/∆wspd)')
    plt.grid(alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / 'scale_sensitivity.png', dpi=300)
    plt.close()
    
    # 3. Cross-Scale Impact Analysis
    bias0a = df_results['errors_band0a'].apply(lambda x: x['bias'])
    bias0b = df_results['errors_band0b'].apply(lambda x: x['bias'])
    bias0c = df_results['errors_band0c'].apply(lambda x: x['bias'])
    bias1 = df_results['errors_band1'].apply(lambda x: x['bias'])
    bias2 = df_results['errors_band2'].apply(lambda x: x['bias'])
    
    mix1_bias = df_results['errors_mix1'].apply(lambda x: x['bias'])
    mix2_bias = df_results['errors_mix2'].apply(lambda x: x['bias'])
    mix3_bias = df_results['errors_mix3'].apply(lambda x: x['bias'])
    mix4_bias = df_results['errors_mix4'].apply(lambda x: x['bias'])

    no_mix_bias = df_results['errors_no_mix'].apply(lambda x: x['bias'])    
    
    plt.figure(figsize=(14, 8))
    mix_labels = ['Model(0a,0b)+Obs(0c,1,2)', 'Obs(0a,0b)+Model(0c,1,2)',
                  'Model(0a,0b,0c)+Obs(1,2)', 'Obs(0a,0b,0c)+Model(1,2)']
    all_labels = [*band_labels, *mix_labels]
    
    plt.boxplot([bias0a, bias0b, bias0c, bias1, bias2, mix1_bias, mix2_bias, mix3_bias, mix4_bias], 
                labels=all_labels)
    plt.title('Cross-Scale Impact Analysis: Bias')
    plt.ylabel('Bias (m/s)')
    plt.grid(alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / 'cross_scale_impact.png', dpi=300)
    plt.close()
    
    fig, axs = plt.subplots(3, 1, figsize=(14, 20))
    
    axs[0].boxplot([ratio0a, ratio0b, ratio0c, ratio1, ratio2], labels=band_labels)
    axs[0].set_title('Transfer Function Ratios')
    axs[0].set_ylabel('Model-to-Observed Ratio')
    axs[0].axhline(y=1.0, color='r', linestyle='--', alpha=0.5)
    axs[0].grid(alpha=0.3)
    axs[0].tick_params(axis='x', rotation=45)
    
    axs[1].boxplot([sensitivity0a, sensitivity0b, sensitivity0c, sensitivity1, sensitivity2], labels=band_labels)
    axs[1].set_title('Scale-Dependent Sensitivity')
    axs[1].set_ylabel('Sensitivity (∆sigma0/∆wspd)')
    axs[1].grid(alpha=0.3)
    axs[1].tick_params(axis='x', rotation=45)
    
    axs[2].boxplot([bias0a, bias0b, bias0c, bias1, bias2, mix1_bias, mix2_bias, mix3_bias, mix4_bias], 
                  labels=all_labels)
    axs[2].set_title('Wind Retrieval Bias')
    axs[2].set_ylabel('Bias (m/s)')
    axs[2].grid(alpha=0.3)
    axs[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'scale_dependency_summary.png', dpi=300)
    plt.close()
    
    plt.figure(figsize=(10, 6))
    mix_data = [mix1_bias, mix2_bias, mix3_bias, mix4_bias, no_mix_bias]
    plt.boxplot(mix_data, labels=mix_labels)
    plt.title('Comparison of Different Mix Strategies')
    plt.ylabel('Bias (m/s)')
    plt.grid(alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / 'mix_strategies_comparison.png', dpi=300)
    plt.close()

def plot_statistical_test_results(test_results, output_path):
    """
    Generate visualization of key statistical test results.
    
    Parameters:
    -----------
    test_results : dict
        Dictionary with test results from perform_statistical_tests
    output_path : Path
        Path to save the visualization
    """
    tests = ['Transfer Function', 'Sensitivity', 'Cross-Scale']
    p_values = [
        test_results['transfer_ratio']['p_value'],
        test_results['sensitivity']['p_value'],
        test_results['cross_scale']['p_value']
    ]
    
    valid_tests = [not np.isnan(p) for p in p_values]
    
    if not any(valid_tests):
        print("Warning: No valid statistical tests to plot")
        return
    
    valid_tests_names = [test for test, valid in zip(tests, valid_tests) if valid]
    valid_p_values = [p for p, valid in zip(p_values, valid_tests) if valid]
    significant = [p < 0.05 for p in valid_p_values]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(valid_tests_names, valid_p_values, color=['green' if sig else 'red' for sig in significant])
    
    plt.axhline(y=0.05, color='black', linestyle='--', alpha=0.7, label='Significance Threshold (p=0.05)')
    
    plt.ylabel('p-value')
    plt.title('Statistical Test Results for Scale Dependency')
    plt.grid(alpha=0.3)
    plt.yscale('log')
    plt.legend()
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'p={height:.4f}', ha='center', va='bottom', 
                 rotation=0 if height > 0.1 else 0)
    
    for i, sig in enumerate(significant):
        plt.text(i, 0.001, 'SIGNIFICANT' if sig else 'not significant',
                ha='center', va='bottom', rotation=90, alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_path / 'statistical_test_results.png', dpi=300)
    plt.close()

def process_sar_file(sar_filepath, era5_wspd, era5_wdir, seed=None):
    """Process a single SAR file according to the workflow."""
    try:
        sar_ds = read_sar_data(sar_filepath)
        if sar_ds is None:
            return None
        
        sigma_sar = sar_ds.sigma0.values
        incidence = sar_ds.incidence.values
        ground_heading = sar_ds.ground_heading.values

        if sigma_sar.ndim == 3:
            sigma_sar = sigma_sar[0]  
        
        if np.isnan(sigma_sar[-1, :]).all():
            sigma_sar = sigma_sar[:-1, :]
            incidence = incidence[:-1, :]
            ground_heading = ground_heading[:-1, :]

        if np.isnan(sigma_sar[:, -1]).all():
            sigma_sar = sigma_sar[:, :-1]
            incidence = incidence[:, :-1]
            ground_heading = ground_heading[:, :-1]
                
        ground_heading = np.mod(ground_heading, 360)
        
        azimuth_look = np.mod(ground_heading + 90, 360)
        
        wspd_perturbed, wdir_perturbed = coupled_perturbation(era5_wspd, era5_wdir, seed)
        
        wspd_perturbed_strong, wdir_perturbed_strong = coupled_perturbation(era5_wspd, era5_wdir, seed, factor=2.0)
        
        phi_perturbed = compute_phi(wdir_perturbed, azimuth_look)
        phi_perturbed_strong = compute_phi(wdir_perturbed_strong, azimuth_look)
        phi_nominal = compute_phi(era5_wdir, azimuth_look)
        
        sigma_cmod = cmod5n_forward(wspd_perturbed, phi_perturbed, incidence)
        sigma_cmod_strong = cmod5n_forward(wspd_perturbed_strong, phi_perturbed_strong, incidence)
        
        fft_sar, psd_sar, kx_sar, ky_sar, kmag_sar = compute_2d_fft(sigma_sar)
        fft_cmod, psd_cmod, kx_cmod, ky_cmod, kmag_cmod = compute_2d_fft(sigma_cmod)
        fft_cmod_strong, _, _, _, kmag_cmod_strong = compute_2d_fft(sigma_cmod_strong)
        
        band0a_sar = band_filter(fft_sar, kmag_sar, 0, 0.0333)
        band0b_sar = band_filter(fft_sar, kmag_sar, 0.0333, 0.0666)
        band0c_sar = band_filter(fft_sar, kmag_sar, 0.0666, 0.1)
        band1_sar = band_filter(fft_sar, kmag_sar, 0.1, 0.3)
        band2_sar = band_filter(fft_sar, kmag_sar, 0.3, np.inf)

        band0a_cmod = band_filter(fft_cmod, kmag_cmod, 0, 0.0333)
        band0b_cmod = band_filter(fft_cmod, kmag_cmod, 0.0333, 0.0666)
        band0c_cmod = band_filter(fft_cmod, kmag_cmod, 0.0666, 0.1)
        band1_cmod = band_filter(fft_cmod, kmag_cmod, 0.1, 0.3)
        band2_cmod = band_filter(fft_cmod, kmag_cmod, 0.3, np.inf)

        band0a_cmod_strong = band_filter(fft_cmod_strong, kmag_cmod_strong, 0, 0.0333)
        band0b_cmod_strong = band_filter(fft_cmod_strong, kmag_cmod_strong, 0.0333, 0.0666)
        band0c_cmod_strong = band_filter(fft_cmod_strong, kmag_cmod_strong, 0.0666, 0.1)
        band1_cmod_strong = band_filter(fft_cmod_strong, kmag_cmod_strong, 0.1, 0.3)
        band2_cmod_strong = band_filter(fft_cmod_strong, kmag_cmod_strong, 0.3, np.inf)
        
        wspd_band0a = cmod5n_inverse(band0a_sar, phi_nominal, incidence)
        wspd_band0b = cmod5n_inverse(band0b_sar, phi_nominal, incidence)
        wspd_band0c = cmod5n_inverse(band0c_sar, phi_nominal, incidence)
        wspd_band1 = cmod5n_inverse(band1_sar, phi_nominal, incidence)
        wspd_band2 = cmod5n_inverse(band2_sar, phi_nominal, incidence)
        
        errors_band0a = calculate_error_metrics(wspd_band0a, era5_wspd)
        errors_band0b = calculate_error_metrics(wspd_band0b, era5_wspd)
        errors_band0c = calculate_error_metrics(wspd_band0c, era5_wspd)
        errors_band1 = calculate_error_metrics(wspd_band1, era5_wspd)
        errors_band2 = calculate_error_metrics(wspd_band2, era5_wspd)
        
        ratio_band0a = np.nanmean(band0a_cmod / np.where(band0a_sar == 0, np.nan, band0a_sar))
        ratio_band0b = np.nanmean(band0b_cmod / np.where(band0b_sar == 0, np.nan, band0b_sar))
        ratio_band0c = np.nanmean(band0c_cmod / np.where(band0c_sar == 0, np.nan, band0c_sar))
        ratio_band1 = np.nanmean(band1_cmod / np.where(band1_sar == 0, np.nan, band1_sar))
        ratio_band2 = np.nanmean(band2_cmod / np.where(band2_sar == 0, np.nan, band2_sar))
        
        band_ranges = {
            'band0a': (0, 0.0333),
            'band0b': (0.0333, 0.0666),
            'band0c': (0.0666, 0.1),
            'band1': (0.1, 0.3),
            'band2': (0.3, np.inf)
        }
        coherence_metrics = calculate_spectral_coherence(sigma_sar, sigma_cmod, kmag_sar, band_ranges)
        
        bands_cmod = {
            'band0a': (band0a_cmod, band0a_cmod_strong),
            'band0b': (band0b_cmod, band0b_cmod_strong),
            'band0c': (band0c_cmod, band0c_cmod_strong),
            'band1': (band1_cmod, band1_cmod_strong),
            'band2': (band2_cmod, band2_cmod_strong)
        }
        
        sensitivity_metrics = calculate_scale_dependent_sensitivity(
            bands_cmod, wspd_perturbed, wspd_perturbed_strong
        )
        
        mix1_sigma0 = band0a_cmod + band0b_cmod + band0c_sar + band1_sar + band2_sar
        wspd_mix1 = cmod5n_inverse(mix1_sigma0, phi_nominal, incidence)
        errors_mix1 = calculate_error_metrics(wspd_mix1, era5_wspd)
        
        mix2_sigma0 = band0a_sar + band0b_sar + band0c_cmod + band1_cmod + band2_cmod
        wspd_mix2 = cmod5n_inverse(mix2_sigma0, phi_nominal, incidence)
        errors_mix2 = calculate_error_metrics(wspd_mix2, era5_wspd)
        
        mix3_sigma0 = band0a_cmod + band0b_cmod + band0c_cmod + band1_sar + band2_sar
        wspd_mix3 = cmod5n_inverse(mix3_sigma0, phi_nominal, incidence)
        errors_mix3 = calculate_error_metrics(wspd_mix3, era5_wspd)
        
        mix4_sigma0 = band0a_sar + band0b_sar + band0c_sar + band1_cmod + band2_cmod
        wspd_mix4 = cmod5n_inverse(mix4_sigma0, phi_nominal, incidence)
        errors_mix4 = calculate_error_metrics(wspd_mix4, era5_wspd)

        mix_sigma0 = band0a_sar + band0b_sar + band0c_sar + band1_sar + band2_sar
        wspd_no_mix = cmod5n_inverse(mix_sigma0, phi_nominal, incidence)
        errors_no_mix = calculate_error_metrics(wspd_no_mix, era5_wspd)
        
        band0a_errors = wspd_band0a.flatten() - era5_wspd
        band0b_errors = wspd_band0b.flatten() - era5_wspd
        band0c_errors = wspd_band0c.flatten() - era5_wspd
        band1_errors = wspd_band1.flatten() - era5_wspd
        band2_errors = wspd_band2.flatten() - era5_wspd
        
        statistic, p_value, is_significant = kruskal_wallis_test(
            band0a_errors[~np.isnan(band0a_errors)],
            band0b_errors[~np.isnan(band0b_errors)],
            band0c_errors[~np.isnan(band0c_errors)],
            band1_errors[~np.isnan(band1_errors)],
            band2_errors[~np.isnan(band2_errors)]
        )
        
        return {
            'sar_filepath': sar_filepath,
            'era5_wspd': era5_wspd,
            'era5_wdir': era5_wdir,
            'wspd_perturbed': wspd_perturbed,
            'wdir_perturbed': wdir_perturbed,
            'phi_perturbed': np.median(phi_perturbed),
            'phi_nominal': np.median(phi_nominal),
            'wspd_perturbed_strong': wspd_perturbed_strong,
            'wdir_perturbed_strong': wdir_perturbed_strong,
            'phi_perturbed_strong': np.median(phi_perturbed_strong),
            'sigma_sar_median': np.median(sigma_sar),
            'sigma_cmod_median': np.median(sigma_cmod),
            'errors_band0a': errors_band0a,
            'errors_band0b': errors_band0b,
            'errors_band0c': errors_band0c,
            'errors_band1': errors_band1,
            'errors_band2': errors_band2,
            'ratio_band0a': ratio_band0a,
            'ratio_band0b': ratio_band0b,
            'ratio_band0c': ratio_band0c,
            'ratio_band1': ratio_band1,
            'ratio_band2': ratio_band2,
            'coherence_metrics': coherence_metrics,
            'sensitivity_metrics': sensitivity_metrics,
            'errors_mix1': errors_mix1,
            'errors_mix2': errors_mix2,
            'errors_mix3': errors_mix3,
            'errors_mix4': errors_mix4,
            'errors_no_mix': errors_no_mix,
            'kw_statistic': statistic,
            'kw_p_value': p_value,
            'is_scale_dependent': is_significant
        }
        
    except Exception as e:
        print(f"Error processing SAR file {sar_filepath}: {e}")
        return None
    
def radial_profile(data, center=None):
    y, x = np.indices(data.shape)
    if center is None:
        center = np.array([(x.max() - x.min()) / 2.0, (y.max() - y.min()) / 2.0])
    
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.astype(int)
    
    tbin = np.bincount(r.ravel(), weights=data.ravel())
    nr = np.bincount(r.ravel())
    
    radial_profile = tbin / nr
    
    return np.arange(len(radial_profile)), radial_profile

def process_sar_file_v2(sar_filepath, era5_wspd, era5_wdir, seed=None):
    """Process a single SAR file to calculate radial PSD."""
    try:
        sar_ds = read_sar_data(sar_filepath)
        if sar_ds is None:
            return None
        
        sigma_sar = sar_ds.sigma0.values
        incidence = sar_ds.incidence.values
        ground_heading = sar_ds.ground_heading.values

        if sigma_sar.ndim == 3:
            sigma_sar = sigma_sar[0] 
        
        if np.isnan(sigma_sar[-1, :]).all():
            sigma_sar = sigma_sar[:-1, :]
            incidence = incidence[:-1, :]
            ground_heading = ground_heading[:-1, :]

        if np.isnan(sigma_sar[:, -1]).all():
            sigma_sar = sigma_sar[:, :-1]
            incidence = incidence[:, :-1]
            ground_heading = ground_heading[:, :-1]

        _, psd, _, _, _ = compute_2d_fft(sigma_sar)

        psd_centered = np.fft.fftshift(psd)
        
        def radial_profile(data, center=None):
            y, x = np.indices(data.shape)
            if center is None:
                center = np.array([(x.max() - x.min()) / 2.0, (y.max() - y.min()) / 2.0])
            
            r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
            r = r.astype(int)
            
            tbin = np.bincount(r.ravel(), weights=data.ravel())
            nr = np.bincount(r.ravel())
            nr[nr == 0] = 1  
            
            return np.arange(len(tbin)), tbin / nr
        
        distances, radial_psd = radial_profile(psd_centered)
        min_length = min(len(distances), len(radial_psd))
        distances = distances[:min_length]
        radial_psd = radial_psd[:min_length]

        pixel_size = 5  
        k_values = distances * (1.0 / (pixel_size * max(psd.shape)))

        return {
            'sar_filepath': sar_filepath,
            'radial_psd': radial_psd,
            'k_values': k_values,
        }
    except Exception as e:
        print(f"Error processing {sar_filepath} for radial PSD: {e}")
        return None
    
def process_sar_file_v3(sar_filepath, era5_wspd, era5_wdir, seed=None):
    """Process a single SAR file to calculate radial wind PSD."""
    try:
        sar_ds = read_sar_data(sar_filepath)
        if sar_ds is None:
            return None
        
        sigma_sar = sar_ds.sigma0.values
        incidence = sar_ds.incidence.values
        ground_heading = sar_ds.ground_heading.values
        

        if sigma_sar.ndim == 3:
            sigma_sar = sigma_sar[0] 
        
        if np.isnan(sigma_sar[-1, :]).all():
            sigma_sar = sigma_sar[:-1, :]
            incidence = incidence[:-1, :]
            ground_heading = ground_heading[:-1, :]

        if np.isnan(sigma_sar[:, -1]).all():
            sigma_sar = sigma_sar[:, :-1]
            incidence = incidence[:, :-1]
            ground_heading = ground_heading[:, :-1]
        
        azimuth_look = np.mod(ground_heading + 90, 360)
        
        
        phi = compute_phi(era5_wdir, azimuth_look)
        
        wind_field = cmod5n_inverse(sigma_sar, phi, incidence)

        def radial_profile(data, center=None):
            y, x = np.indices(data.shape)
            if center is None:
                center = np.array([(x.max() - x.min()) / 2.0, (y.max() - y.min()) / 2.0])
            
            r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
            r = r.astype(int)
            
            tbin = np.bincount(r.ravel(), weights=data.ravel())
            nr = np.bincount(r.ravel())
            nr[nr == 0] = 1  
            
            return np.arange(len(tbin)), tbin / nr
        
        def process_wind_field(wind_field):
            fft_wind = np.fft.fft2(wind_field)
            
            fft_shifted = np.fft.fftshift(fft_wind)
            
            psd = np.abs(fft_shifted)**2
            
            distances, radial_psd = radial_profile(psd)
            
            return radial_psd, distances, psd

        radial_wind_psd, distances, psd_wind = process_wind_field(wind_field)
        
        min_length = min(len(distances), len(radial_wind_psd))
        distances = distances[:min_length]
        radial_wind_psd = radial_wind_psd[:min_length]

        pixel_size = 5  
        k_values_wind = distances * (1.0 / (pixel_size * max(psd_wind.shape)))

        return {
            'sar_filepath': sar_filepath,
            'radial_wind_psd': radial_wind_psd,
            'k_values_wind': k_values_wind,
        }
    
    except Exception as e:
        print(f"Error processing {sar_filepath} for radial wind: {e}")
        return None
    
def plot_avg_spectral_density(k_values, df_list, title_list, suptitle, confidence=0.95, 
                               figsize=(12, 8), x_range=None, y_range=None, use_log_scale=True, wavelength=False):
    """
    Creates a scientific publication-quality plot of average spectral density with confidence intervals.
    
    Parameters:
    -----------
    k_values : array-like
        The wavenumber values for the x-axis
    df_list : list of DataFrames
        List of DataFrames containing the spectral density data
    title_list : list of str
        List of titles corresponding to each DataFrame
    suptitle : str
        The main title for the plot
    confidence : float, optional
        Confidence level (default: 0.95 for 95% confidence interval)
    figsize : tuple, optional
        Figure size (width, height) in inches (default: (12, 8))
    x_range : tuple or None, optional
        Tuple of (min_x, max_x) to explicitly set the x-axis range
        If None, the full range of k_values will be shown (default: None)
    y_range : tuple or None, optional
        Tuple of (min_y, max_y) to explicitly set the y-axis range
        If None, an appropriate range will be calculated (default: None)
    use_log_scale : bool, optional
        Whether to use logarithmic scaling (default: True)
        Set to False for linear scaling
    wavelength : bool, optional
        If True, plot with wavelength (1/k) on x-axis instead of wavenumber
        (default: False)
    """
    
    plt.style.use('seaborn-v0_8-whitegrid')
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    fig, ax = plt.subplots(figsize=figsize, dpi=120)
    
    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'legend.frameon': True,
        'legend.framealpha': 0.7,
        'legend.edgecolor': 'lightgray'
    })
    
    if wavelength:
        x_values = np.array([1.0/k if k != 0 else float('inf') for k in k_values])
        if x_range:
            if x_range[0] != 0 and x_range[1] != 0:
                x_range = (1.0/x_range[1], 1.0/x_range[0])
    else:
        x_values = k_values
    
    alpha = 1 - confidence
    
    for i, (df, title) in enumerate(zip(df_list, title_list)):
        if "Wind" in title:
            column_name = 'radial_wind_psd_padded'
        elif "PSD" in title:
            column_name = 'radial_psd_padded'
        else:
            column_name = 'radial_psd_padded'  
        
        try:
            if column_name not in df.columns:
                print(f"Warning: Column '{column_name}' not found in dataframe. Available columns: {df.columns.tolist()}")
                continue
                
            values_list = []
            
            if df[column_name].apply(lambda x: isinstance(x, (list, np.ndarray))).all():
                for arr in df[column_name].values:
                    if len(arr) == len(k_values): 
                        values_list.append(arr)
            else:
                raise ValueError("Expected array-like values in column")
            all_values = np.vstack(values_list)
            
            mean_values = np.nanmean(all_values, axis=0)
            sem_values = stats.sem(all_values, axis=0, nan_policy='omit')
            
            df_stats = all_values.shape[0] - 1  
            
            t_critical = stats.t.ppf(1 - alpha/2, df_stats)
            ci_lower = mean_values - t_critical * sem_values
            ci_upper = mean_values + t_critical * sem_values
            
            ci_lower = np.maximum(ci_lower, np.min(mean_values) * 0.01)
            
            total_range_observations = len(values_list)
            
            if use_log_scale:
                main_line = ax.loglog(x_values, mean_values, 
                         color=colors[i % len(colors)], 
                         linewidth=2.5, 
                         label=f'{title} (n={total_range_observations})')
                
                confidence_pct = int(confidence * 100)
                lower_line = ax.loglog(x_values, ci_lower, 
                        color=colors[i % len(colors)], 
                        linestyle='--', 
                        linewidth=1.5, 
                        alpha=0.7,
                    )
                        
                upper_line = ax.loglog(x_values, ci_upper, 
                        color=colors[i % len(colors)], 
                        linestyle='--', 
                        linewidth=1.5, 
                        alpha=0.7,
                    )
            else:
                main_line = ax.plot(x_values, mean_values, 
                         color=colors[i % len(colors)], 
                         linewidth=2.5, 
                         label=f'{title} (n={total_range_observations})')
                
                confidence_pct = int(confidence * 100)
                lower_line = ax.plot(x_values, ci_lower, 
                        color=colors[i % len(colors)], 
                        linestyle='--', 
                        linewidth=1.5, 
                        alpha=0.7,
                )
                        
                upper_line = ax.plot(x_values, ci_upper, 
                        color=colors[i % len(colors)], 
                        linestyle='--', 
                        linewidth=1.5, 
                        alpha=0.7,
                )
            
            ax.fill_between(x_values, 
                           ci_lower, 
                           ci_upper, 
                           color=colors[i % len(colors)], 
                           alpha=0.15)
                               
        except Exception as e:
            print(f"Error processing {title}: {str(e)}")
            continue
    
    ax.grid(True, which="major", ls="-", alpha=0.2)
    ax.grid(True, which="minor", ls=":", alpha=0.1)
    
    if wavelength:
        ax.set_xlabel(r'Wavelength [m]', fontweight='bold')
    else:
        ax.set_xlabel(r'Wavenumber [m$^-1$]', fontweight='bold')
    ax.set_ylabel(r'Power Spectral Density (PSD)', fontweight='bold')
    
    range_info = ""
    ax.set_title(f"{suptitle}{range_info}", fontweight='bold', fontsize=16)
    
    if x_range:
        ax.set_xlim(x_range)
    if y_range:
        ax.set_ylim(y_range)
    
    ax.tick_params(axis='both', which='major', direction='in', length=6, width=1)
    ax.tick_params(axis='both', which='minor', direction='in', length=3, width=0.5)
    
    if use_log_scale:
        ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
        ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
    else:
    
        import matplotlib.ticker as ticker
        ax.xaxis.set_major_locator(ticker.AutoLocator())
        ax.yaxis.set_major_locator(ticker.AutoLocator())
    
    legend = ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, frameon=True)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.9)
    
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
    
    text_x = 0.48
    text_y = 0.95
    scale_type = "Linear scale" if not use_log_scale else "Log scale"

    x_label = "Wavelength" if wavelength else "Wavenumber"

    ax.text(text_x, text_y, 
            f"Dashed lines represent {int(confidence*100)}% confidence intervals • {scale_type}", 
            transform=ax.transAxes, fontsize=10, 
            bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5'))
    
    plt.tight_layout()
    
    return fig, ax

def pad_arrays_to_max_length(df, column_name, size=None):
    max_length = df[column_name].apply(lambda x: len(x)).max()
    
    def pad_array(arr, size=size):
        if not size:
            padding_size = max_length - len(arr)
        else:
            padding_size = size - len(arr)

        if padding_size > 0:
            return np.pad(arr, (0, padding_size), 'constant', constant_values=0)
        else:
            return arr
    
    df[column_name + '_padded'] = df[column_name].apply(pad_array)
    
    return df

def get_k_values(df, column_name):
    max_length_idx = df[column_name].apply(lambda x: len(x)).argmax()
    
    k_values = df[column_name].iloc[max_length_idx]
    
    return k_values

def create_phi_bins_columns(df):
    df['phi_bins'] = pd.cut(
    df['phi_nominal_median'], 
    bins=np.arange(-180, 181, 1),
    right=False, 
    include_lowest=True
    )

    df["phi_bins"] = df["phi_bins"].astype(str)

    return df

def create_dfs_from_phi_interval(phi_bin, df_complete, df_results_updated, df_results_wind):
    if not isinstance(phi_bin, str):
        raise ValueError("phi_bin should be a string")
    
    if phi_bin not in df_complete.phi_bins.unique():
        raise ValueError(f"phi_bin {phi_bin} not found in df_complete")
    
    df_phi_interval  = df_complete[df_complete.phi_bins == phi_bin].copy()
    dfr_phi_interval = df_results_updated[df_results_updated.renamed_filename.isin(df_phi_interval.renamed_filename)].copy()
    dfw_phi_interval = df_results_wind[df_results_wind.renamed_filename.isin(df_phi_interval.renamed_filename)].copy()

    return df_phi_interval, dfr_phi_interval, dfw_phi_interval

def filter_similar_atmospheric_conditions(df, air_sea_temp_diff_range=(-3, -1), 
                                       blh_range=(1400, 1600), rh_range=(70, 80)):
   """
   Filter a dataframe to include only rows with similar atmospheric conditions.
   
   Parameters:
   -----------
   df : pandas.DataFrame
       Input dataframe containing meteorological variables
   air_sea_temp_diff_range : tuple
       Range (min, max) of acceptable air-sea temperature differences in °C
   blh_range : tuple
       Range (min, max) of acceptable boundary layer heights in meters
   rh_range : tuple
       Range (min, max) of acceptable relative humidity values in percent
       
   Returns:
   --------
   pandas.DataFrame
       Filtered dataframe with only rows meeting the specified atmospheric conditions
   """
   df['air_sea_diff'] = df['airt'] - df['sst']
   
   filtered_df = df[
       (df['air_sea_diff'] >= air_sea_temp_diff_range[0]) & 
       (df['air_sea_diff'] <= air_sea_temp_diff_range[1]) &
       
       (df['blh'] >= blh_range[0]) & 
       (df['blh'] <= blh_range[1]) &
       
       (df['rh'] >= rh_range[0]) & 
       (df['rh'] <= rh_range[1]) &
    
       (df['wspd'] > 15) &
       (df['L'] < 0)
   ]
   
   original_count = len(df)
   filtered_count = len(filtered_df)
   print(f"Filtered from {original_count} to {filtered_count} observations ({filtered_count/original_count:.1%} retained)")
   
   print("\nMean values before filtering:")
   print(f"Air-Sea Diff: {df['air_sea_diff'].mean():.2f}°C, BLH: {df['blh'].mean():.0f}m, RH: {df['rh'].mean():.1f}%")
   
   print("\nMean values after filtering:")
   print(f"Air-Sea Diff: {filtered_df['air_sea_diff'].mean():.2f}°C, BLH: {filtered_df['blh'].mean():.0f}m, RH: {filtered_df['rh'].mean():.1f}%")
   
   return filtered_df