import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from scipy.stats import ttest_ind, levene, ks_2samp, shapiro, bartlett, f_oneway, kruskal
from statsmodels.stats.multitest import multipletests
import xarray as xr
from utils.cmod5n import *

def check_normality(data, alpha=0.01):
    """Test normality of data using Shapiro-Wilk test"""
    # Subsample if data is too large (Shapiro-Wilk limit is 5000)
    if len(data) > 5000:
        np.random.seed(42)  # for reproducibility
        data = np.random.choice(data, size=5000, replace=False)
    
    stat, p_value = shapiro(data)
    
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

def test_scale_dependence(df, band1_col, band2_col, alpha=0.01):
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
        },
    }

def perform_omnibus_tests(df, band_columns, alpha=0.01):
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

def bootstrap_ratio_confidence_intervals(data1, data2, n_bootstrap=1000, confidence=0.99):
    """Calculate bootstrap confidence intervals for the ratio of medians"""
    bootstrap_ratios = []
    
    for _ in range(n_bootstrap):
        # Resample both datasets
        sample1 = np.random.choice(data1, size=len(data1), replace=True)
        sample2 = np.random.choice(data2, size=len(data2), replace=True)
        # Calculate ratio of medians
        ratio = np.median(sample1) / np.median(sample2)
        bootstrap_ratios.append(ratio)
    
    lower_quantile = (1 - confidence) / 2
    upper_quantile = 1 - lower_quantile
    
    return (np.quantile(bootstrap_ratios, lower_quantile),
            np.quantile(bootstrap_ratios, upper_quantile))

# Main analysis function
def analyze_scale_dependence(df_wv1, df_wv2, bootstrap_samples=1000, confidence=0.99, alpha=0.01):
    """
    Comprehensive analysis of scale dependence using median-based methods
    """
    print("\n" + "="*80)
    print("SCALE DEPENDENCE ANALYSIS")
    print("="*80)
    print(f"\nAnalysis Parameters:")
    print(f"- Bootstrap Samples: {bootstrap_samples}")
    print(f"- Confidence Level: {confidence*100}%")
    print(f"- Significance Level (alpha): {alpha}")
    
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
    print("\n" + "-"*80)
    print("STEP 1: NORMALITY ASSESSMENT")
    print("-"*80)
    
    for dataset_name, df in [('WV1', df_wv1), ('WV2', df_wv2)]:
        print(f"\n{dataset_name} Normality Results:")
        for band in band_columns:
            data = df[band].dropna().values
            is_normal, norm_test = check_normality(data)
            print(f"  {band}:")
            print(f"    Normal: {is_normal}")
            print(f"    Shapiro-Wilk p-value: {norm_test['p_value']:.2e}")
    
    print("\nGenerating QQ plots for visual inspection...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('QQ Plots for Normality Assessment', fontsize=16)
    
    for i, (dataset_name, df) in enumerate([('WV1', df_wv1), ('WV2', df_wv2)]):
        for j, band in enumerate(band_columns):
            data = df[band].dropna().values
            is_normal, norm_test = check_normality(data)
            ax = axes[i, j]
            stats.probplot(data, dist="norm", plot=ax)
            ax.set_title(f"{dataset_name} - {band}\nNormal: {is_normal}")
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('images/normality_qq_plots.png', dpi=300)
    print("QQ plots saved as 'normality_qq_plots.png'")
    
    # 2. Bootstrap confidence intervals
    print("\n" + "-"*80)
    print("STEP 2: BOOTSTRAP ANALYSIS")
    print("-"*80)
    print(f"Computing {confidence*100}% confidence intervals using {bootstrap_samples} resamples\n")
    
    for dataset_name, df in [('WV1', df_wv1), ('WV2', df_wv2)]:
        print(f"\n{dataset_name} Bootstrap Results:")
        for band in band_columns:
            valid_data = df[band].dropna().values
            results[dataset_name]['bootstrap'][band] = bootstrap_confidence_intervals(
                valid_data, n_bootstrap=bootstrap_samples, confidence=confidence
            )
            ci = results[dataset_name]['bootstrap'][band]
            print(f"\n  {band}:")
            print(f"    Median CI: [{ci['median'][0]:.2f}, {ci['median'][1]:.2f}]")
            print(f"    IQR CI: [{ci['iqr'][0]:.2f}, {ci['iqr'][1]:.2f}]")
            print(f"    Skewness CI: [{ci['skewness'][0]:.2f}, {ci['skewness'][1]:.2f}]")
    
    # 3. Omnibus tests
    print("\n" + "-"*80)
    print("STEP 3: OMNIBUS TESTS")
    print("-"*80)
    print("Testing for any differences among bands")
    
    for dataset_name, df in [('WV1', df_wv1), ('WV2', df_wv2)]:
        results[dataset_name]['omnibus'] = perform_omnibus_tests(df, band_columns, alpha=alpha)
        
        is_significant = results[dataset_name]['omnibus']['significant']
        test_name = results[dataset_name]['omnibus']['test']
        statistic = results[dataset_name]['omnibus']['statistic']
        p_value = results[dataset_name]['omnibus']['p_value']
        
        print(f"\n{dataset_name} Results:")
        print(f"  Test: {test_name}")
        print(f"  Statistic: {statistic:.2f}")
        print(f"  p-value: {p_value:.2e}")
        print(f"  Conclusion: {'Significant differences exist' if is_significant else 'No significant differences'}")
    
    # 4. Pairwise tests
    print("\n" + "-"*80)
    print("STEP 4: PAIRWISE COMPARISONS")
    print("-"*80)
    
    for dataset_name, df in [('WV1', df_wv1), ('WV2', df_wv2)]:
        print(f"\n{dataset_name} Pairwise Results:")
        if not results[dataset_name]['omnibus']['significant']:
            print("  Skipping (omnibus test not significant)")
            continue
            
        first_moment_pvalues = []
        second_moment_pvalues = []
        distribution_pvalues = []
        
        for band1, band2 in band_pairs:
            print(f"\n  Comparing {band1} vs {band2}:")
            test_result = test_scale_dependence(df, band1, band2, alpha=alpha)
            results[dataset_name]['pairwise'][(band1, band2)] = test_result
            
            # Collect p-values
            first_moment_pvalues.append(test_result['first_moment']['p_value'])
            second_moment_pvalues.append(test_result['second_moment']['p_value'])
            distribution_pvalues.append(test_result['distribution']['p_value'])
            
            # Print detailed results
            print(f"    Mann-Whitney U test:")
            print(f"      Statistic: {test_result['first_moment']['statistic']:.2f}")
            print(f"      p-value: {test_result['first_moment']['p_value']:.2e}")
            
            print(f"    Mood's Median test:")
            print(f"      Statistic: {test_result['second_moment']['statistic']:.2f}")
            print(f"      p-value: {test_result['second_moment']['p_value']:.2e}")
            
            print(f"    KS test:")
            print(f"      Statistic: {test_result['distribution']['statistic']:.2f}")
            print(f"      p-value: {test_result['distribution']['p_value']:.2e}")
        
        # Adjust p-values
        print("\n  Bonferroni-adjusted p-values:")
        adjusted_first = adjust_pvalues(first_moment_pvalues)
        adjusted_second = adjust_pvalues(second_moment_pvalues)
        adjusted_distribution = adjust_pvalues(distribution_pvalues)
        
        for i, (band1, band2) in enumerate(band_pairs):
            results[dataset_name]['pairwise'][(band1, band2)]['first_moment']['adjusted_p_value'] = adjusted_first[i]
            results[dataset_name]['pairwise'][(band1, band2)]['second_moment']['adjusted_p_value'] = adjusted_second[i]
            results[dataset_name]['pairwise'][(band1, band2)]['distribution']['adjusted_p_value'] = adjusted_distribution[i]
            
            print(f"    {band1} vs {band2}:")
            print(f"      Mann-Whitney: {adjusted_first[i]:.2e}")
            print(f"      Mood's Median: {adjusted_second[i]:.2e}")
            print(f"      KS test: {adjusted_distribution[i]:.2e}")
    
    # 5. Visualization
    print("\n" + "-"*80)
    print("STEP 5: VISUALIZATION")
    print("-"*80)
    print("Generating comprehensive visualization plots:")
    print("1. Median PSD across scales (with bootstrap CIs)")
    print("2. Zoomed view of Bands 1 & 2")
    print("3. Distribution boxplots")
    print("4. Scale dependence ratio analysis")
    print("5. Ratio distributions")

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
    
    # Band 0
    ax3 = fig.add_subplot(gs[1, 0])
    sns.boxplot(data=[df_wv1['mean_psd_band0'].dropna(), df_wv2['mean_psd_band0'].dropna()], ax=ax3, showfliers=False)
    ax3.set_xticks([0, 1])  # Add this line
    ax3.set_xticklabels(['WV1', 'WV2'])
    ax3.set_ylabel('Power')
    ax3.set_title('Band0 (Low k) Distribution')
    ax3.set_yscale("log")

    # Band 1
    ax4 = fig.add_subplot(gs[1, 1], sharey=ax3)
    sns.boxplot(data=[df_wv1['mean_psd_band1'].dropna(), df_wv2['mean_psd_band1'].dropna()], ax=ax4, showfliers=False)
    ax4.set_xticks([0, 1])
    ax4.set_xticklabels(['WV1', 'WV2'])
    ax4.set_ylabel('Power')
    ax4.set_title('Band1 (Medium k) Distribution')

    # Band 2
    ax5 = fig.add_subplot(gs[1, 2], sharey=ax3)
    sns.boxplot(data=[df_wv1['mean_psd_band2'].dropna(), df_wv2['mean_psd_band2'].dropna()], ax=ax5, showfliers=False)
    ax5.set_xticks([0, 1])
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
        
    # Calculate ratios of medians and their CIs
    wv1_ratio_01 = np.median(df_wv1['mean_psd_band0'])/np.median(df_wv1['mean_psd_band1'])
    wv1_ratio_12 = np.median(df_wv1['mean_psd_band1'])/np.median(df_wv1['mean_psd_band2'])
    wv2_ratio_01 = np.median(df_wv2['mean_psd_band0'])/np.median(df_wv2['mean_psd_band1'])
    wv2_ratio_12 = np.median(df_wv2['mean_psd_band1'])/np.median(df_wv2['mean_psd_band2'])

    # Calculate CIs for ratios
    wv1_01_ci = bootstrap_ratio_confidence_intervals(
        df_wv1['mean_psd_band0'].dropna(),
        df_wv1['mean_psd_band1'].dropna()
    )
    wv1_12_ci = bootstrap_ratio_confidence_intervals(
        df_wv1['mean_psd_band1'].dropna(),
        df_wv1['mean_psd_band2'].dropna()
    )
    wv2_01_ci = bootstrap_ratio_confidence_intervals(
        df_wv2['mean_psd_band0'].dropna(),
        df_wv2['mean_psd_band1'].dropna()
    )
    wv2_12_ci = bootstrap_ratio_confidence_intervals(
        df_wv2['mean_psd_band1'].dropna(),
        df_wv2['mean_psd_band2'].dropna()
    )

    ratio_values = [wv1_ratio_01, wv1_ratio_12, wv2_ratio_01, wv2_ratio_12]
    ratio_cis = [wv1_01_ci, wv1_12_ci, wv2_01_ci, wv2_12_ci]

    # Add error bars to the ratio plot
    ax6.errorbar(ratio_x, ratio_values,
                yerr=[[r - ci[0] for r, ci in zip(ratio_values, ratio_cis)],
                    [ci[1] - r for r, ci in zip(ratio_values, ratio_cis)]],
                fmt='none', color='k', capsize=5)

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
    sns.violinplot(x='Dataset', y='Ratio', 
               data=ratio_data[ratio_data['Type'] == 'Band0/Band1'],
               ax=ax7, 
               hue='Dataset',
               legend=False,
               palette=['blue', 'orange'])
    
    ax7.set_title('Band0/Band1 Ratios')
    ax7.set_ylabel('Ratio Value')

    # 6. Violin plots for ratio distributions in the main figure
    ax8 = fig.add_subplot(gs[2, 2])

    sns.violinplot(x='Dataset', y='Ratio', 
               data=ratio_data[ratio_data['Type'] == 'Band1/Band2'],
               ax=ax8, 
               hue='Dataset',
               legend=False, 
               palette=['blue', 'orange'])
    
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

    print("\nVisualization saved as 'scale_dependence_median_analysis.png'")

    # 6. Create summary DataFrame 
    
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
    
    print("\n" + "="*80)
    print("ANALYSIS SUMMARY")
    print("="*80)
    print("\nKey Findings:")
    for dataset_name, df in [('WV1', df_wv1), ('WV2', df_wv2)]:
        print(f"\n{dataset_name}:")
        print("  Scale Ratios (with 99% CIs):")
        
        ratio_01 = np.median(df['mean_psd_band0'])/np.median(df['mean_psd_band1'])
        ratio_12 = np.median(df['mean_psd_band1'])/np.median(df['mean_psd_band2'])
        
        ci_01 = bootstrap_ratio_confidence_intervals(
            df['mean_psd_band0'].dropna(),
            df['mean_psd_band1'].dropna()
        )
        ci_12 = bootstrap_ratio_confidence_intervals(
            df['mean_psd_band1'].dropna(),
            df['mean_psd_band2'].dropna()
        )
        
        print(f"    Band0/Band1: {ratio_01:.2f} [{ci_01[0]:.2f}, {ci_01[1]:.2f}]")
        print(f"    Band1/Band2: {ratio_12:.2f} [{ci_12[0]:.2f}, {ci_12[1]:.2f}]")
    
    print("\nDetailed results available in returned DataFrame")
    return results, summary_df

def add_phi_nominal_to_dataset(file_path, wdir_deg_from_north, perturbed_wdir):
    try:
        with xr.open_dataset(file_path,  engine='h5netcdf') as ds:
            ground_heading = ds.ground_heading.values  # Raw satellite heading
            
            # 1. Normalize ground_heading to 0-360°
            ground_heading = np.mod(ground_heading, 360)

            # 2. Calculate true azimuth look direction (Sentinel-1 right-looking adjustment)
            azimuth_look = np.mod(ground_heading + 90, 360)

            # 3. Compute phi with both angles in 0-360° convention
            phi_perturbed = perturbed_wdir - azimuth_look
            phi_nominal = wdir_deg_from_north - azimuth_look

            # 4. Wrap to [-180°, 180°] 
            phi_perturbed = ((phi_perturbed + 180) % 360) - 180           
            phi_nominal = ((phi_nominal + 180) % 360) - 180

            return pd.Series({'phi_nominal_median': np.median(phi_nominal), 
                              'phi_perturbed_median': np.median(phi_perturbed),
                              'ground_heading_median': np.median(ground_heading),
                              'azimuth_look_median': np.median(azimuth_look)})
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return pd.Series({'phi_nominal_median': np.nan, 
                          'phi_perturbed_median': np.nan,
                          'ground_heading_median': np.nan,
                          'azimuth_look_median': np.nan})

def perform_cmod_in_dataset(file_path, wdir_deg_from_north, perturbed_wdir, wspd):

    try:
        with xr.open_dataset(file_path) as ds:
            
            true_sigma0 = ds.sigma0[0].values  
            
            # remove last row of true_sigma0 if its full of nans
            last_row_full_of_nans= False
            if np.all(np.isnan(true_sigma0[-1, :])):
                last_row_full_of_nans = True
                true_sigma0 = true_sigma0[:-1, :]

            # remove lat column of true_sigma0 if its full of nans
            last_column_full_of_nans = False
            if np.all(np.isnan(true_sigma0[:, -1])):
                last_column_full_of_nans = True
                true_sigma0 = true_sigma0[:, :-1]

            # get statistics from true_sigma0
            true_sigma0_median = np.median(true_sigma0)
            true_sigma0_row_var = np.var(true_sigma0, axis=1)
            true_sigma0_column_var = np.var(true_sigma0, axis=0)

            true_sigma0_flatten = true_sigma0.flatten()
            true_sigma0_skew = stats.skew(true_sigma0_flatten)
            true_sigma0_kurtosis = stats.kurtosis(true_sigma0_flatten)

            incidence = ds.incidence.values
            ground_heading = ds.ground_heading.values  # Raw satellite heading
            
            # 1. Normalize ground_heading to 0-360°
            ground_heading = np.mod(ground_heading, 360)
            
            if last_row_full_of_nans:
                incidence = incidence[:-1, :]
                ground_heading = ground_heading[:-1, :]

            if last_column_full_of_nans:
                incidence = incidence[:, :-1]
                ground_heading = ground_heading[:, :-1]

            # 2. Calculate true azimuth look direction (Sentinel-1 right-looking adjustment)
            azimuth_look = np.mod(ground_heading + 90, 360)
            
            # 3. Compute phi with both angles in 0-360° convention
            phi_perturbed = perturbed_wdir - azimuth_look
            phi_nominal = wdir_deg_from_north - azimuth_look

            # 4. Wrap to [-180°, 180°] 
            phi_perturbed = ((phi_perturbed + 180) % 360) - 180           
            phi_nominal = ((phi_nominal + 180) % 360) - 180

            sigma0_cmod = cmod5n_forward(np.full(phi_perturbed.shape, wspd),
                                            phi_perturbed,
                                            incidence
                                        )
            # get statistics from sigma0_cmod
            sigma0_cmod_median = np.median(sigma0_cmod)
            sigma0_cmod_row_var = np.var(sigma0_cmod, axis=1)
            sigma0_cmod_column_var = np.var(sigma0_cmod, axis=0)

            sigma0_cmod_flatten = sigma0_cmod.flatten()
            sigma0_cmod_skew = stats.skew(sigma0_cmod_flatten)
            sigma0_cmod_kurtosis = stats.kurtosis(sigma0_cmod_flatten)
             
                         
            # Execute CMOD5N inversion
            wspd_cmod = cmod5n_inverse(sigma0_cmod,
                                        phi_nominal,
                                        incidence,
                                        )
            
            
            # Calculate statistics (handling potential NaN values)
            wspd_flat = wspd_cmod.flatten()
            wspd_flat = wspd_flat[~np.isnan(wspd_flat)]  # Remove NaN values for skew and kurtosis
            
            wspd_median = np.nanmedian(wspd_cmod)
            wspd_var = np.nanvar(wspd_cmod)
            
            # Handle case where there are insufficient non-NaN values
            if len(wspd_flat) > 3:  # Need at least a few points for meaningful skew/kurtosis
                wspd_skewness = stats.skew(wspd_flat)
                wspd_kurtosis = stats.kurtosis(wspd_flat)
            else:
                wspd_skewness = np.nan
                wspd_kurtosis = np.nan
                
            return pd.Series([true_sigma0_median, true_sigma0_row_var, 
                              true_sigma0_column_var, true_sigma0_skew, 
                              true_sigma0_kurtosis, sigma0_cmod_median,
                              sigma0_cmod_row_var, sigma0_cmod_column_var,
                              sigma0_cmod_skew, sigma0_cmod_kurtosis,
                              wspd_median, wspd_var, wspd_skewness, wspd_kurtosis])
                              
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return pd.Series([np.nan, np.nan, np.nan, np.nan,
                          np.nan, np.nan, np.nan, np.nan,
                          np.nan, np.nan, np.nan, np.nan,
                          np.nan, np.nan])
        
def add_direction_uncertainty(wdir_era5, sigma=30):
    """Add Gaussian noise to wind direction (deg)"""
    noise = np.random.normal(0, sigma, size=wdir_era5.shape)
    return np.mod(wdir_era5 + noise, 360)

def plot_sar_wind(df_wv1_unstable_gt15, idx, cmod5n_inverse):
    """
    Plot SAR wind data with CMOD-derived wind speed, sigma0, and PSD.
    
    Parameters:
    -----------
    df_wv1_unstable_gt15 : pandas.DataFrame
        DataFrame containing SAR metadata
    idx : int
        Index of the record to plot
    cmod5n_inverse : function
        Function to calculate wind speed from SAR data
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure object
    """
    
    # Extract data for the given index
    fn = df_wv1_unstable_gt15.renamed_filename.iloc[idx]
    path_to_data = df_wv1_unstable_gt15.path_to_sar_file.iloc[idx]
    
    wdir_rad = df_wv1_unstable_gt15.wdir.iloc[idx]
    wdir_deg_from_north = np.rad2deg(wdir_rad) % 360
    wspd_era5 = df_wv1_unstable_gt15.wspd.iloc[idx]
    
    # Load SAR data
    ds = xr.open_dataset(path_to_data, engine='h5netcdf')
    sigma0 = ds['sigma0'][0]
    incidence_angle = ds['incidence'].values
    ground_heading = ds['ground_heading'].values
    
    # Calculate relative wind direction
    phi = wdir_deg_from_north - ground_heading
    phi = np.mod(phi + 180, 360) - 180
    
    # Calculate CMOD wind speed
    wspd_cmod = cmod5n_inverse(sigma0, phi, incidence_angle, iterations=10)
    
    # Create 1x3 plot layout
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot 1: CMOD derived wind speed with wind direction arrow
    im1 = axes[0].imshow(wspd_cmod)
    cbar1 = plt.colorbar(im1, ax=axes[0], label="Wind Speed [m/s]")
    axes[0].set_title(f"CMOD5n Wind Speed\nMedian = {np.nanmedian(wspd_cmod):.2f} m/s")
    
    # Get center of the image
    center_y, center_x = np.array(wspd_cmod.shape) / 2
    
    arrow_angle_rad = np.deg2rad(wdir_deg_from_north + 180)
    dx = np.sin(arrow_angle_rad)  
    dy = -np.cos(arrow_angle_rad)  
    
    # Draw arrow
    arrow_length = min(wspd_cmod.shape) / 6  
    axes[0].arrow(center_x, center_y, dx * arrow_length, dy * arrow_length, 
                  head_width=arrow_length/3, head_length=arrow_length/2, 
                  fc='k', ec='k', linewidth=2)
    
    # Add North indicator for reference
    axes[0].text(0.02, 0.98, "N↑", transform=axes[0].transAxes, 
                 fontsize=12, fontweight='bold', va='top')
    
    # Plot 2: Sigma0
    im2 = axes[1].imshow(sigma0)
    cbar2 = plt.colorbar(im2, ax=axes[1], label="Sigma0 [linear]")
    axes[1].set_title("Sigma0 (Normalized Radar Cross Section)")
    
    # Plot 3: PSD of sigma0
    # Calculate 2D Power Spectral Density
    sigma0_clean = np.nan_to_num(sigma0)  
    
    # First, compute 2D FFT and shift zero frequency to center
    f_sigma0 = np.fft.fft2(sigma0_clean)
    f_sigma0_shifted = np.fft.fftshift(f_sigma0)
    psd_2d = np.abs(f_sigma0_shifted)**2
    
    # Plot PSD (log scale is often better for visualizing PSD)
    im3 = axes[2].imshow(np.log10(psd_2d + 1e-10)) 
    cbar3 = plt.colorbar(im3, ax=axes[2], label="Log10 PSD")
    axes[2].set_title("Power Spectral Density of Sigma0")
    
    # Add annotation for wind direction and ERA5 speed
    info_text = f"Wind Direction: {wdir_deg_from_north:.1f}°\nERA5 Wind Speed: {wspd_era5:.2f} m/s"
    axes[0].text(0.02, 0.02, info_text, transform=axes[0].transAxes, 
                 bbox=dict(facecolor='white', alpha=0.7), fontsize=10, va='bottom')
    
    plt.tight_layout()
    
    return fig

def compute_spectral_wind_errors(file_path, era5_wind_speed, wdir_nominal, perturbed_wdir=None, perturbation_sigma=30):
    """
    Compute wind speed errors across different spectral bands using the forward-inverse CMOD approach.
    
    Parameters:
    -----------
    file_path : str
        Path to the SAR data file
    era5_wind_speed : float
        Single ERA5 wind speed value for the entire scene
    wdir_nominal : float
        Nominal wind direction in degrees
    perturbed_wdir : float, optional
        Perturbed wind direction. If None, will add Gaussian noise to wdir_nominal
    perturbation_sigma : float, default=30
        Standard deviation for Gaussian perturbation if perturbed_wdir is None
        
    Returns:
    --------
    dict
        Dictionary containing error metrics for each spectral band
    """
    try:
        # Open dataset and extract data
        with xr.open_dataset(file_path) as ds:
            sigma0_values = ds.sigma0.values
            ground_heading = ds.ground_heading.values
            incidence = ds.incidence.values

            if sigma0_values.ndim == 3:
                sigma0_values = sigma0_values[0]
                        
            # Clean NaN rows/columns if needed
            if np.isnan(sigma0_values[-1, :]).all():
                sigma0_values = sigma0_values[:-1, :]
                ground_heading = ground_heading[:-1, :]
                incidence = incidence[:-1, :]

            if np.isnan(sigma0_values[:, -1]).all():
                sigma0_values = sigma0_values[:, :-1]
                ground_heading = ground_heading[:, :-1]
                incidence = incidence[:, :-1]
            
            # Generate perturbed wind direction if not provided
            if perturbed_wdir is None:
                noise = np.random.normal(0, perturbation_sigma)
                perturbed_wdir = np.mod(wdir_nominal + noise, 360)
            
            # Calculate phi angles
            azimuth_look = np.mod(ground_heading + 90, 360)
            phi_nominal = wdir_nominal - azimuth_look
            phi_perturbed = perturbed_wdir - azimuth_look
            
            # Wrap to [-180°, 180°]
            phi_nominal = ((phi_nominal + 180) % 360) - 180
            phi_perturbed = ((phi_perturbed + 180) % 360) - 180
            
            # Forward model: Generate simulated sigma0 using perturbed direction
            sigma0_cmod = cmod5n_forward(np.full(phi_perturbed.shape, era5_wind_speed),
                                         phi_perturbed,
                                         incidence)
        
        # Compute 2D Fourier transform and power spectrum of the simulated sigma0
        fft_data = np.fft.fft2(sigma0_cmod)
        psd_2d = np.abs(fft_data)**2
        
        # Set up frequency grid
        freq_x = np.fft.fftfreq(sigma0_cmod.shape[1])
        freq_y = np.fft.fftfreq(sigma0_cmod.shape[0])
        kx, ky = np.meshgrid(freq_x, freq_y)
        k_magnitude = np.sqrt(kx**2 + ky**2)
        
        # Define wavenumber bands
        k_bands = [
            (0, 0.1),     # Band 0: Low wavenumbers (large scales)
            (0.1, 0.3),   # Band 1: Medium wavenumbers
            (0.3, np.inf) # Band 2: High wavenumbers (small scales)
        ]
        
        results = {
            # 'true_sigma0_median': np.median(sigma0_values),
            # 'sigma0_cmod_median': np.median(sigma0_cmod),
            # 'sigma0_norm_error': (np.median(sigma0_cmod) - np.median(sigma0_values)) / np.median(sigma0_values),
            # 'wdir_nominal': wdir_nominal,
            # 'wdir_perturbed': perturbed_wdir
        }
        
        # Process each band
        for i, (k_min, k_max) in enumerate(k_bands):
            mask = (k_magnitude >= k_min) & (k_magnitude < k_max)
            
            if np.any(mask):
                # Create filtered version of simulated sigma0 for this band
                fft_filtered = np.zeros_like(fft_data)
                fft_filtered[mask] = fft_data[mask]
                sigma0_filtered = np.real(np.fft.ifft2(fft_filtered))
                
                # Inverse model: Retrieve wind speed using nominal direction
                wspd_band = cmod5n_inverse(sigma0_filtered, phi_nominal, incidence)
                
                # Calculate error metrics
                error = wspd_band - era5_wind_speed
                abs_error = np.abs(error)
                rel_error = error / era5_wind_speed
                
                results[f'band{i}_wspd_mean'] = np.nanmean(wspd_band)
                results[f'band{i}_wspd_median'] = np.nanmedian(wspd_band)
                results[f'band{i}_error_mean'] = np.nanmean(error)
                results[f'band{i}_error_median'] = np.nanmedian(error) 
                results[f'band{i}_abs_error_mean'] = np.nanmean(abs_error)
                results[f'band{i}_rel_error_mean'] = np.nanmean(rel_error)
                results[f'band{i}_rmse'] = np.sqrt(np.nanmean(error**2))
        
        return results
        
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

def analyze_band_dependencies(df, wind_col='wspd', wind_dir_col='wdir'):
    """
    Apply spectral analysis to all files in the dataframe
    
    Parameters:
    -----------
    df : pandas DataFrame
        Contains paths to SAR files and ERA5 wind data
    wind_col : str
        Column name for ERA5 wind speed
    wind_dir_col : str
        Column name for ERA5 wind direction (in radians)
        
    Returns:
    --------
    pandas DataFrame
        Original dataframe with added band-specific error columns
    """
    result_rows = []
    
    for _, row in df.iterrows():
        file_path = row['path_to_sar_file']
        era5_wspd = row[wind_col]
        era5_wdir = np.rad2deg(row[wind_dir_col]) % 360  # Convert to degrees
        
        band_errors = compute_spectral_wind_errors(file_path, era5_wspd, era5_wdir)
        
        if band_errors:
            # Add band results to the row
            result_row = row.to_dict()
            result_row.update(band_errors)
            result_rows.append(result_row)
    
    return pd.DataFrame(result_rows)