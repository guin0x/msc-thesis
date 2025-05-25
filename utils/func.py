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
        
        sigma_cmod, b0_cmod, b1_cmod, b2_cmod = cmod5n_forward(wspd_perturbed, phi_perturbed, incidence)
        sigma_cmod_strong, b0_cmod_strong, b1_cmod_strong, b2_cmod_strong = cmod5n_forward(wspd_perturbed_strong, phi_perturbed_strong, incidence)
        
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
        
        wspd_band0a, _, _, _ = cmod5n_inverse(band0a_sar, phi_nominal, incidence)
        wspd_band0b, _, _, _ = cmod5n_inverse(band0b_sar, phi_nominal, incidence)
        wspd_band0c, _, _, _ = cmod5n_inverse(band0c_sar, phi_nominal, incidence)
        wspd_band1, _, _, _ = cmod5n_inverse(band1_sar, phi_nominal, incidence)
        wspd_band2, _, _, _ = cmod5n_inverse(band2_sar, phi_nominal, incidence)
        
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
        wspd_mix1, _, _, _ = cmod5n_inverse(mix1_sigma0, phi_nominal, incidence)
        errors_mix1 = calculate_error_metrics(wspd_mix1, era5_wspd)
        
        mix2_sigma0 = band0a_sar + band0b_sar + band0c_cmod + band1_cmod + band2_cmod
        wspd_mix2, _, _, _ = cmod5n_inverse(mix2_sigma0, phi_nominal, incidence)
        errors_mix2 = calculate_error_metrics(wspd_mix2, era5_wspd)
        
        mix3_sigma0 = band0a_cmod + band0b_cmod + band0c_cmod + band1_sar + band2_sar
        wspd_mix3, _, _, _ = cmod5n_inverse(mix3_sigma0, phi_nominal, incidence)
        errors_mix3 = calculate_error_metrics(wspd_mix3, era5_wspd)
        
        mix4_sigma0 = band0a_sar + band0b_sar + band0c_sar + band1_cmod + band2_cmod
        wspd_mix4, _, _, _ = cmod5n_inverse(mix4_sigma0, phi_nominal, incidence)
        errors_mix4 = calculate_error_metrics(wspd_mix4, era5_wspd)

        mix_sigma0 = band0a_sar + band0b_sar + band0c_sar + band1_sar + band2_sar
        wspd_no_mix, _, _, _ = cmod5n_inverse(mix_sigma0, phi_nominal, incidence)
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

        pixel_size = 100
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
        wind_field, b0, b1, b2 = cmod5n_inverse(sigma_sar, phi, incidence)

        b0_stats = {"mean": float(np.nanmean(b0)), 
                    "median": float(np.nanmedian(b0)), 
                    "std": float(np.nanstd(b0))}
        
        b1_stats = {"mean": float(np.nanmean(b1)),
                    "median": float(np.nanmedian(b1)), 
                    "std": float(np.nanstd(b1))}
        
        b2_stats = {"mean": float(np.nanmean(b2)),
                    "median": float(np.nanmedian(b2)), 
                    "std": float(np.nanstd(b2))}

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

        pixel_size = 100
        k_values_wind = distances * (1.0 / (pixel_size * max(psd_wind.shape)))

        return {
            'sar_filepath': sar_filepath,
            'radial_wind_psd': radial_wind_psd.tolist(),  # Convert to regular Python list
            'k_values_wind': k_values_wind.tolist(),      # Convert to regular Python list
            'b0': b0_stats,  # Already JSON-serializable
            'b1': b1_stats,  # Already JSON-serializable
            'b2': b2_stats,  # Already JSON-serializable
        }
    
    except Exception as e:
        print(f"Error processing {sar_filepath} for radial wind: {e}")
        return None