import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from scipy.stats import ttest_ind, levene, ks_2samp, shapiro, bartlett, f_oneway, kruskal
from statsmodels.stats.multitest import multipletests

def check_normality(data, alpha=0.05):
    """
    Test normality of data using Shapiro-Wilk test
    
    Parameters:
    -----------
    data : array-like
        Data to test
    alpha : float
        Significance level
    
    Returns:
    --------
    bool
        True if data appears normally distributed
    dict
        Test results
    """
    # Shapiro-Wilk test (most powerful normality test)
    stat, p_value = shapiro(data)
    
    # QQ plot data for visualization
    qq_x = np.linspace(0, 1, len(data))
    qq_y = np.sort(data)
    
    return p_value > alpha, {
        'test': 'Shapiro-Wilk',
        'statistic': stat,
        'p_value': p_value,
        'normal': p_value > alpha,
        'qq_data': (qq_x, qq_y)
    }

def bootstrap_confidence_intervals(data, n_bootstrap=1000, confidence=0.99):
    """
    Calculate bootstrap confidence intervals for the median, IQR, and skewness
    
    Parameters:
    -----------
    data : array-like
        Data to bootstrap
    n_bootstrap : int
        Number of bootstrap samples
    confidence : float
        Confidence level (between 0 and 1)
    
    Returns:
    --------
    dict
        Dictionary with confidence intervals for each statistic
    """
    bootstrap_medians = []
    bootstrap_iqrs = []
    bootstrap_skews = []
    
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_medians.append(np.median(sample))
        # Calculate IQR (Q3 - Q1) as a robust measure of dispersion
        q1, q3 = np.percentile(sample, [25, 75])
        bootstrap_iqrs.append(q3 - q1)
        bootstrap_skews.append(stats.skew(sample))
    
    lower_quantile = (1 - confidence) / 2
    upper_quantile = 1 - lower_quantile
    
    return {
        'median': (np.quantile(bootstrap_medians, lower_quantile), 
                   np.quantile(bootstrap_medians, upper_quantile)),
        'iqr': (np.quantile(bootstrap_iqrs, lower_quantile), 
                np.quantile(bootstrap_iqrs, upper_quantile)),
        'skewness': (np.quantile(bootstrap_skews, lower_quantile), 
                     np.quantile(bootstrap_skews, upper_quantile))
    }

def test_scale_dependence(df, band1_col, band2_col, alpha=0.05):
    """
    Test for scale dependence between two frequency bands
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the band data
    band1_col : str
        Column name for the first band
    band2_col : str
        Column name for the second band
    alpha : float
        Significance level
    
    Returns:
    --------
    dict
        Dictionary with test results
    """
    # Remove NaN values
    valid_data = df[[band1_col, band2_col]].dropna()
    
    # Check normality of each band
    is_normal_band1, norm_test_band1 = check_normality(valid_data[band1_col])
    is_normal_band2, norm_test_band2 = check_normality(valid_data[band2_col])
    both_normal = is_normal_band1 and is_normal_band2
    
    # Test for equality of variances using both Levene (robust) and Bartlett (more power if normal)
    levene_stat, levene_pvalue = levene(valid_data[band1_col], valid_data[band2_col])
    bartlett_stat, bartlett_pvalue = bartlett(valid_data[band1_col], valid_data[band2_col])
    
    # Use more robust Levene's test by default, but consider Bartlett if data are normal
    equal_var_test = "Bartlett's test" if both_normal else "Levene's test"
    equal_var_pvalue = bartlett_pvalue if both_normal else levene_pvalue
    equal_var = equal_var_pvalue > alpha
    
    # We'll use non-parametric tests since we're focusing on medians
    # Non-parametric test for location (Mann-Whitney U test)
    u_stat, u_pvalue = stats.mannwhitneyu(valid_data[band1_col], valid_data[band2_col])
    first_moment_test = "Mann-Whitney U test"
    first_moment_stat = u_stat
    first_moment_pvalue = u_pvalue
    
    # Non-parametric test for scale (Mood's median test)
    # mood_stat, mood_pvalue = stats.median_test(valid_data[band1_col], valid_data[band2_col])
    mood_stat, mood_pvalue, _, _ = stats.median_test(valid_data[band1_col], valid_data[band2_col])
    second_moment_test = "Mood's median test"
    second_moment_stat = mood_stat
    second_moment_pvalue = mood_pvalue
    
    # Distribution test (Kolmogorov-Smirnov)
    ks_stat, ks_pvalue = ks_2samp(valid_data[band1_col], valid_data[band2_col])
    
    return {
        'normality': {
            'band1': {
                'test': norm_test_band1['test'],
                'p_value': norm_test_band1['p_value'],
                'normal': norm_test_band1['normal']
            },
            'band2': {
                'test': norm_test_band2['test'],
                'p_value': norm_test_band2['p_value'],
                'normal': norm_test_band2['normal']
            },
            'both_normal': both_normal
        },
        'first_moment': {
            'test': first_moment_test,
            'statistic': first_moment_stat,
            'p_value': first_moment_pvalue,
            'significant': first_moment_pvalue < alpha
        },
        'second_moment': {
            'test': second_moment_test,
            'statistic': second_moment_stat,
            'p_value': second_moment_pvalue,
            'significant': second_moment_pvalue < alpha,
            'levene': {'statistic': levene_stat, 'p_value': levene_pvalue},
            'bartlett': {'statistic': bartlett_stat, 'p_value': bartlett_pvalue}
        },
        'distribution': {
            'test': 'Kolmogorov-Smirnov test',
            'statistic': ks_stat,
            'p_value': ks_pvalue,
            'significant': ks_pvalue < alpha
        }
    }

def perform_omnibus_tests(df, band_columns, alpha=0.05):
    """
    Perform omnibus tests across all bands before pairwise comparisons
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the band data
    band_columns : list
        List of column names for the bands
    alpha : float
        Significance level
    
    Returns:
    --------
    dict
        Dictionary with test results
    """
    # Prepare data for testing
    data_for_test = [df[band].dropna().values for band in band_columns]
    
    # Check normality of all bands
    normality_results = [check_normality(band_data)[0] for band_data in data_for_test]
    all_normal = all(normality_results)
    
    # For non-parametric analysis (focusing on medians), we'll use Kruskal-Wallis
    # Kruskal-Wallis H-test (non-parametric alternative to one-way ANOVA)
    h_stat, h_pvalue = kruskal(*data_for_test)
    test_name = "Kruskal-Wallis H-test"
    statistic = h_stat
    p_value = h_pvalue
    
    return {
        'test': test_name,
        'statistic': statistic,
        'p_value': p_value,
        'significant': p_value < alpha,
        'all_normal': all_normal
    }

# Adjust p-values for multiple comparisons
def adjust_pvalues(p_values, method='bonferroni'):
    """
    Adjust p-values for multiple comparisons
    
    Parameters:
    -----------
    p_values : array-like
        Array of p-values
    method : str
        Method for adjustment ('bonferroni', 'holm', 'fdr_bh')
    
    Returns:
    --------
    array
        Adjusted p-values
    """
    return multipletests(p_values, method=method)[1]

# Main analysis function
def analyze_scale_dependence(df_wv1, df_wv2, bootstrap_samples=1000, confidence=0.99, alpha=0.05):
    """
    Comprehensive analysis of scale dependence using median-based methods
    
    Parameters:
    -----------
    df_wv1, df_wv2 : pandas.DataFrame
        DataFrames containing band data for WV1 and WV2
    bootstrap_samples : int
        Number of bootstrap samples
    confidence : float
        Confidence level for bootstrap intervals
    alpha : float
        Significance level for hypothesis tests
    
    Returns:
    --------
    dict
        Dictionary with complete analysis results
    """
    print("Performing comprehensive scale dependence analysis using median-based methods...")
    
    # Define band columns and pairs for comparison
    band_columns = ['mean_psd_band0', 'mean_psd_band1', 'mean_psd_band2']
    band_pairs = [
        ('mean_psd_band0', 'mean_psd_band1'),
        ('mean_psd_band1', 'mean_psd_band2'),
        ('mean_psd_band0', 'mean_psd_band2')
    ]
    
    # Results dictionary
    results = {
        'WV1': {'bootstrap': {}, 'omnibus': None, 'pairwise': {}},
        'WV2': {'bootstrap': {}, 'omnibus': None, 'pairwise': {}}
    }
    
    # 1. Normality check and visualization
    print("\nChecking normality of distributions...")
    
    # Create QQ plots for visual inspection of normality
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('QQ Plots for Normality Assessment', fontsize=16)
    
    for i, (dataset_name, df) in enumerate([('WV1', df_wv1), ('WV2', df_wv2)]):
        for j, band in enumerate(band_columns):
            data = df[band].dropna().values
            is_normal, norm_test = check_normality(data)
            
            # Plot QQ plot
            ax = axes[i, j]
            stats.probplot(data, dist="norm", plot=ax)
            ax.set_title(f"{dataset_name} - {band}\nNormal: {is_normal}")
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('images/normality_qq_plots.png', dpi=300)
    
    # 2. Bootstrap confidence intervals for medians
    print("\nComputing bootstrap confidence intervals for medians...")
    
    for dataset_name, df in [('WV1', df_wv1), ('WV2', df_wv2)]:
        for band in band_columns:
            valid_data = df[band].dropna().values
            results[dataset_name]['bootstrap'][band] = bootstrap_confidence_intervals(
                valid_data, n_bootstrap=bootstrap_samples, confidence=confidence
            )
    
    # 3. Omnibus tests
    print("\nPerforming omnibus tests across all bands...")
    
    for dataset_name, df in [('WV1', df_wv1), ('WV2', df_wv2)]:
        results[dataset_name]['omnibus'] = perform_omnibus_tests(df, band_columns, alpha=alpha)
        
        is_significant = results[dataset_name]['omnibus']['significant']
        test_name = results[dataset_name]['omnibus']['test']
        p_value = results[dataset_name]['omnibus']['p_value']
        
        print(f"  {dataset_name} - {test_name}: p-value = {p_value:.6f} "
              f"({'SIGNIFICANT' if is_significant else 'not significant'})")
    
    # 4. Pairwise tests (only if omnibus test is significant)
    print("\nPerforming pairwise comparisons between bands...")
    
    for dataset_name, df in [('WV1', df_wv1), ('WV2', df_wv2)]:
        if results[dataset_name]['omnibus']['significant']:
            # Collect p-values for adjustment
            first_moment_pvalues = []
            second_moment_pvalues = []
            distribution_pvalues = []
            
            # Perform pairwise tests
            for band1, band2 in band_pairs:
                test_result = test_scale_dependence(df, band1, band2, alpha=alpha)
                results[dataset_name]['pairwise'][(band1, band2)] = test_result
                
                first_moment_pvalues.append(test_result['first_moment']['p_value'])
                second_moment_pvalues.append(test_result['second_moment']['p_value'])
                distribution_pvalues.append(test_result['distribution']['p_value'])
            
            # Adjust p-values for multiple comparisons
            adjusted_first = adjust_pvalues(first_moment_pvalues)
            adjusted_second = adjust_pvalues(second_moment_pvalues)
            adjusted_distribution = adjust_pvalues(distribution_pvalues)
            
            # Store adjusted p-values
            for i, (band1, band2) in enumerate(band_pairs):
                results[dataset_name]['pairwise'][(band1, band2)]['first_moment']['adjusted_p_value'] = adjusted_first[i]
                results[dataset_name]['pairwise'][(band1, band2)]['second_moment']['adjusted_p_value'] = adjusted_second[i]
                results[dataset_name]['pairwise'][(band1, band2)]['distribution']['adjusted_p_value'] = adjusted_distribution[i]
    
    # 5. Create visualization
    print("\nGenerating visualizations...")

    # Create a larger figure with more subplots
    fig = plt.figure(figsize=(18, 15))
    fig.suptitle('Scale Dependence Analysis (Median-Based)', fontsize=18)

    # Create a grid layout
    gs = fig.add_gridspec(3, 3)

    # 1. Log scale plot of all bands
    ax1 = fig.add_subplot(gs[0, :2])
    x_labels = ['Band0\n(Low k)', 'Band1\n(Medium k)', 'Band2\n(High k)']

    median_values_wv1 = [np.median(df_wv1[band].dropna()) for band in band_columns]
    median_values_wv2 = [np.median(df_wv2[band].dropna()) for band in band_columns]

    ci_lower_wv1 = [results['WV1']['bootstrap'][band]['median'][0] for band in band_columns]
    ci_upper_wv1 = [results['WV1']['bootstrap'][band]['median'][1] for band in band_columns]
    ci_lower_wv2 = [results['WV2']['bootstrap'][band]['median'][0] for band in band_columns]
    ci_upper_wv2 = [results['WV2']['bootstrap'][band]['median'][1] for band in band_columns]

    x = np.arange(len(x_labels))
    width = 0.35

    # Create log scale bar plot
    bars1 = ax1.bar(x - width/2, median_values_wv1, width, label='WV1', alpha=0.7)
    bars2 = ax1.bar(x + width/2, median_values_wv2, width, label='WV2', alpha=0.7)

    # Add error bars
    ax1.errorbar(x - width/2, median_values_wv1, 
                yerr=[np.array(median_values_wv1) - np.array(ci_lower_wv1), 
                    np.array(ci_upper_wv1) - np.array(median_values_wv1)],
                fmt='none', color='k', capsize=5)

    ax1.errorbar(x + width/2, median_values_wv2, 
                yerr=[np.array(median_values_wv2) - np.array(ci_lower_wv2), 
                    np.array(ci_upper_wv2) - np.array(median_values_wv2)],
                fmt='none', color='k', capsize=5)

    # Set to log scale to see all bands
    ax1.set_yscale('log')
    ax1.set_xticks(x)
    ax1.set_xticklabels(x_labels)
    ax1.set_ylabel('Median PSD (log scale)')
    ax1.set_title('Median Power Spectral Density Across Scales\n(99% Confidence Intervals, Log Scale)')
    ax1.legend()

    # Add value annotations to the bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height*1.1,
                f'{height:.1f}', ha='center', va='bottom', rotation=0, fontsize=9)
                
    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height*1.1,
                f'{height:.1f}', ha='center', va='bottom', rotation=0, fontsize=9)

    # 2. Linear scale plot focusing on bands 1 and 2 only
    ax2 = fig.add_subplot(gs[0, 2])
    x_labels_small = ['Band1\n(Medium k)', 'Band2\n(High k)']
    x_small = np.arange(len(x_labels_small))

    # Extract just bands 1 and 2
    median_values_wv1_small = median_values_wv1[1:]
    median_values_wv2_small = median_values_wv2[1:]
    ci_lower_wv1_small = ci_lower_wv1[1:]
    ci_upper_wv1_small = ci_upper_wv1[1:]
    ci_lower_wv2_small = ci_lower_wv2[1:]
    ci_upper_wv2_small = ci_upper_wv2[1:]

    # Create linear scale bar plot for just bands 1 and 2
    bars1_small = ax2.bar(x_small - width/2, median_values_wv1_small, width, label='WV1', alpha=0.7)
    bars2_small = ax2.bar(x_small + width/2, median_values_wv2_small, width, label='WV2', alpha=0.7)

    # Add error bars
    ax2.errorbar(x_small - width/2, median_values_wv1_small, 
                yerr=[np.array(median_values_wv1_small) - np.array(ci_lower_wv1_small), 
                    np.array(ci_upper_wv1_small) - np.array(median_values_wv1_small)],
                fmt='none', color='k', capsize=5)

    ax2.errorbar(x_small + width/2, median_values_wv2_small, 
                yerr=[np.array(median_values_wv2_small) - np.array(ci_lower_wv2_small), 
                    np.array(ci_upper_wv2_small) - np.array(median_values_wv2_small)],
                fmt='none', color='k', capsize=5)

    ax2.set_xticks(x_small)
    ax2.set_xticklabels(x_labels_small)
    ax2.set_ylabel('Median PSD')
    ax2.set_title('Zoomed View of Bands 1 & 2')
    ax2.legend()

    # Add value annotations
    for bar in bars1_small:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height*1.05,
                f'{height:.2f}', ha='center', va='bottom', fontsize=9)
                
    for bar in bars2_small:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height*1.05,
                f'{height:.2f}', ha='center', va='bottom', fontsize=9)

    # 3. Distribution boxplots with separate axes for each band
    # Create three separate boxplots for each band with their own y-scales
    # Band 0
    ax3 = fig.add_subplot(gs[1, 0])
    sns.boxplot(data=[df_wv1['mean_psd_band0'].dropna(), df_wv2['mean_psd_band0'].dropna()], ax=ax3, showfliers=False)
    ax3.set_xticklabels(['WV1', 'WV2'])
    ax3.set_ylabel('Power')
    ax3.set_title('Band0 (Low k) Distribution')
    ax3.set_yscale("log")

    # Band 1
    ax4 = fig.add_subplot(gs[1, 1], sharey=ax3)
    sns.boxplot(data=[df_wv1['mean_psd_band1'].dropna(), df_wv2['mean_psd_band1'].dropna()], ax=ax4, showfliers=False)
    ax4.set_xticklabels(['WV1', 'WV2'])
    ax4.set_ylabel('Power')
    ax4.set_title('Band1 (Medium k) Distribution')

    # Band 2
    ax5 = fig.add_subplot(gs[1, 2], sharey=ax3)
    sns.boxplot(data=[df_wv1['mean_psd_band2'].dropna(), df_wv2['mean_psd_band2'].dropna()], ax=ax5, showfliers=False)
    ax5.set_xticklabels(['WV1', 'WV2'])
    ax5.set_ylabel('Power')
    ax5.set_title('Band2 (High k) Distribution')

    # 4. Scale dependence visualization: Band ratios (both log and linear scales)
    ax6 = fig.add_subplot(gs[2, 0])

    # Calculate ratios of medians
    wv1_ratio_01 = np.median(df_wv1['mean_psd_band0'])/np.median(df_wv1['mean_psd_band1'])
    wv1_ratio_12 = np.median(df_wv1['mean_psd_band1'])/np.median(df_wv1['mean_psd_band2'])
    wv2_ratio_01 = np.median(df_wv2['mean_psd_band0'])/np.median(df_wv2['mean_psd_band1'])
    wv2_ratio_12 = np.median(df_wv2['mean_psd_band1'])/np.median(df_wv2['mean_psd_band2'])

    ratio_values = [wv1_ratio_01, wv1_ratio_12, wv2_ratio_01, wv2_ratio_12]

    # Create bar plot for median ratios - Log scale
    ratio_labels = ['WV1\nBand0/Band1', 'WV1\nBand1/Band2', 'WV2\nBand0/Band1', 'WV2\nBand1/Band2']
    ratio_x = np.arange(len(ratio_labels))

    bars_ratio = ax6.bar(ratio_x, ratio_values, alpha=0.7, 
                        color=['blue', 'blue', 'orange', 'orange'])

    ax6.set_xticks(ratio_x)
    ax6.set_xticklabels(ratio_labels, rotation=45, ha='right')
    ax6.set_ylabel('Ratio of Medians (log scale)')
    ax6.set_title('Scale Dependence: Ratios of Median Power')
    ax6.set_yscale('log')  # Use log scale to see all ratios

    # Add ratio values as text
    for i, bar in enumerate(bars_ratio):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height*1.1,
                f'{ratio_values[i]:.1f}', ha='center', va='bottom', rotation=0)

    # 5. Focus on Band1/Band2 ratios only (much smaller values)
    ax7 = fig.add_subplot(gs[2, 1])

    # Gather ratio data (element-wise for violin plots)
    wv1_ratios_01 = df_wv1['mean_psd_band0'] / df_wv1['mean_psd_band1']
    wv1_ratios_12 = df_wv1['mean_psd_band1'] / df_wv1['mean_psd_band2']
    wv2_ratios_01 = df_wv2['mean_psd_band0'] / df_wv2['mean_psd_band1']
    wv2_ratios_12 = df_wv2['mean_psd_band1'] / df_wv2['mean_psd_band2']

    # Create split dataset for band0/band1 and band1/band2 separately
    ratio_data_01 = pd.DataFrame({
        'Ratio': np.concatenate([
            wv1_ratios_01.dropna(), 
            wv2_ratios_01.dropna()
        ]),
        'Dataset': np.concatenate([
            np.full(len(wv1_ratios_01.dropna()), 'WV1'),
            np.full(len(wv2_ratios_01.dropna()), 'WV2')
        ]),
        'Type': 'Band0/Band1'
    })

    ratio_data_12 = pd.DataFrame({
        'Ratio': np.concatenate([
            wv1_ratios_12.dropna(), 
            wv2_ratios_12.dropna()
        ]),
        'Dataset': np.concatenate([
            np.full(len(wv1_ratios_12.dropna()), 'WV1'),
            np.full(len(wv2_ratios_12.dropna()), 'WV2')
        ]),
        'Type': 'Band1/Band2'
    })

    # Combine the data
    ratio_data = pd.concat([ratio_data_01, ratio_data_12])

    # Clip extreme values for better visualization (separately for each type)
    for ratio_type in ['Band0/Band1', 'Band1/Band2']:
        mask = ratio_data['Type'] == ratio_type
        clip_upper = np.percentile(ratio_data.loc[mask, 'Ratio'], 95)
        ratio_data.loc[mask, 'Ratio'] = np.clip(ratio_data.loc[mask, 'Ratio'], 0, clip_upper)

    # Band0/Band1 plot
    sns.violinplot(x='Dataset', y='Ratio', data=ratio_data[ratio_data['Type'] == 'Band0/Band1'], 
                ax=ax7, palette=['blue', 'orange'])
    ax7.set_title('Band0/Band1 Ratios')
    ax7.set_ylabel('Ratio Value')

    # 6. Violin plots for ratio distributions in the main figure
    ax8 = fig.add_subplot(gs[2, 2])

    sns.violinplot(x='Dataset', y='Ratio', data=ratio_data[ratio_data['Type'] == 'Band1/Band2'], 
                ax=ax8, palette=['blue', 'orange'])
    ax8.set_title('Band1/Band2 Ratios')
    ax8.set_ylabel('Ratio Value')

    # Add median values as text
    for i, (ratio_type, ax) in enumerate([('Band0/Band1', ax7), ('Band1/Band2', ax8)]):
        for j, dataset in enumerate(['WV1', 'WV2']):
            filtered_data = ratio_data[(ratio_data['Type'] == ratio_type) & (ratio_data['Dataset'] == dataset)]
            median_val = np.median(filtered_data['Ratio'])
            ax.text(j, median_val*1.05, f'Median: {median_val:.2f}', 
                    ha='center', va='bottom', fontsize=9)

    # Add text explaining what ratios mean
    explanation_text = (
        "Band0/Band1 Ratio: Power drop-off from large to medium scales\n"
        "Band1/Band2 Ratio: Power drop-off from medium to small scales\n"
        "Higher ratios = stronger scale dependence"
    )
    fig.text(0.5, 0.01, explanation_text, ha='center', fontsize=12, 
            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig('images/scale_dependence_median_analysis.png', dpi=300)

    # 6. Create summary DataFrame 
    print("\nCreating summary tables...")
    
    # Prepare summary data
    summary_columns = [
        'Comparison', 'Dataset', 'Normality', 'Location Test', 
        'Location p-value', 'Adjusted p-value', 'Scale Test p-value', 
        'KS Test p-value', 'Scale Dependent?'
    ]
    
    summary_data = []
    
    for dataset_name, df in [('WV1', df_wv1), ('WV2', df_wv2)]:
        # Add omnibus test result
        omnibus_result = results[dataset_name]['omnibus']
        summary_data.append([
            'All bands', dataset_name, 
            'All normal' if omnibus_result['all_normal'] else 'Not all normal',
            omnibus_result['test'], omnibus_result['p_value'], 'N/A', 'N/A', 'N/A',
            'Yes' if omnibus_result['significant'] else 'No'
        ])
        
        # Add pairwise comparison results
        for band1, band2 in band_pairs:
            if not omnibus_result['significant']:
                # Skip pairwise tests if omnibus test is not significant
                continue
                
            results_pair = results[dataset_name]['pairwise'][(band1, band2)]
            both_normal = results_pair['normality']['both_normal']
            
            first_moment_test = results_pair['first_moment']['test']
            first_moment_pvalue = results_pair['first_moment']['p_value']
            first_moment_adj_pvalue = results_pair['first_moment']['adjusted_p_value']
            
            second_moment_test = results_pair['second_moment']['test']
            second_moment_pvalue = results_pair['second_moment']['p_value']
            ks_pvalue = results_pair['distribution']['p_value']
            
            scale_dependent = (
                results_pair['first_moment']['significant'] or
                results_pair['second_moment']['significant'] or
                results_pair['distribution']['significant']
            )
            
            summary_data.append([
                f"{band1} vs {band2}", dataset_name, 
                'Both normal' if both_normal else 'Non-normal',
                first_moment_test, first_moment_pvalue, first_moment_adj_pvalue,
                second_moment_pvalue, ks_pvalue,
                'Yes' if scale_dependent else 'No'
            ])
    
    summary_df = pd.DataFrame(summary_data, columns=summary_columns)
    
    return results, summary_df

