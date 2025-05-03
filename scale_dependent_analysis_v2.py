#!/usr/bin/env python

"""
Scale-Dependent Analysis of Wind Stress Variability

This script implements the workflow for analyzing scale-dependent wind stress 
variability using Sentinel-1 WV SAR data, following the coupled perturbation model.
Enhanced with additional scale dependency analysis for CMOD5N.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from multiprocessing import Pool
import argparse
from datetime import datetime
import tqdm

# Import custom functions
from utils.functions_v3 import (
    process_sar_file,
    plot_error_distributions,
    kruskal_wallis_test
)

# Helper function for parallel processing
def process_record(record):
    """Wrapper function for parallel processing"""
    return process_sar_file(
        record['sar_filepath'],
        record['era5_wspd'],
        record['era5_wdir'],
        record.get('seed')
    )

def analyze_scale_dependent_metrics(df_results, output_path):
    """
    Analyze and plot the new scale-dependent metrics.
    
    This function generates additional plots and analyses for the enhanced scale dependency metrics:
    1. Direct sigma0 comparison
    2. Scale-specific transfer function
    3. Cross-scale impact analysis
    4. Spectral coherence
    5. Scale-dependent sensitivity
    """
    # Create directory for enhanced analysis
    enhanced_output_path = output_path / "enhanced_analysis"
    enhanced_output_path.mkdir(exist_ok=True, parents=True)
    
    # Plot and analyze direct sigma0 comparison
    print("Analyzing direct sigma0 comparison...")
    
    # Extract metrics
    sigma0_bias0 = df_results['sigma0_diff_band0'].apply(lambda x: x['bias']).mean()
    sigma0_bias1 = df_results['sigma0_diff_band1'].apply(lambda x: x['bias']).mean()
    sigma0_bias2 = df_results['sigma0_diff_band2'].apply(lambda x: x['bias']).mean()
    
    sigma0_rmse0 = df_results['sigma0_diff_band0'].apply(lambda x: x['rmse']).mean()
    sigma0_rmse1 = df_results['sigma0_diff_band1'].apply(lambda x: x['rmse']).mean()
    sigma0_rmse2 = df_results['sigma0_diff_band2'].apply(lambda x: x['rmse']).mean()
    
    # Bar chart for direct sigma0 comparison
    plt.figure(figsize=(12, 8))
    bands = ['Band 0 (k < 0.1)', 'Band 1 (0.1 < k < 0.3)', 'Band 2 (k > 0.3)']
    
    x = np.arange(len(bands))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 8))
    rects1 = ax.bar(x - width/2, [sigma0_bias0, sigma0_bias1, sigma0_bias2], width, label='Bias')
    rects2 = ax.bar(x + width/2, [sigma0_rmse0, sigma0_rmse1, sigma0_rmse2], width, label='RMSE')
    
    ax.set_ylabel('Sigma0 Error')
    ax.set_title('Direct Sigma0 Comparison by Wavenumber Band')
    ax.set_xticks(x)
    ax.set_xticklabels(bands)
    ax.legend()
    plt.grid(alpha=0.3)
    
    plt.savefig(enhanced_output_path / 'direct_sigma0_comparison.png', dpi=300)
    plt.close()
    
    # Analyze scale-specific transfer function
    print("Analyzing scale-specific transfer function...")
    
    ratio0 = df_results['ratio_band0'].mean()
    ratio1 = df_results['ratio_band1'].mean()
    ratio2 = df_results['ratio_band2'].mean()
    
    plt.figure(figsize=(12, 8))
    plt.bar(bands, [ratio0, ratio1, ratio2], color=['#2C7BB6', '#D7191C', '#FDAE61'])
    plt.axhline(y=1.0, color='black', linestyle='--', alpha=0.7, label='Ideal Ratio (1.0)')
    plt.ylabel('Model-to-Observed Sigma0 Ratio')
    plt.title('Scale-Specific Transfer Function Analysis')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig(enhanced_output_path / 'transfer_function_analysis.png', dpi=300)
    plt.close()
    
    # Analyze cross-scale impact
    print("Analyzing cross-scale impact...")
    
    regular_bias0 = df_results['errors_band0'].apply(lambda x: x['bias']).mean()
    regular_bias1 = df_results['errors_band1'].apply(lambda x: x['bias']).mean()
    regular_bias2 = df_results['errors_band2'].apply(lambda x: x['bias']).mean()
    
    cross_bias_model0_obs12 = df_results['errors_cross_model0_obs12'].apply(lambda x: x['bias']).mean()
    cross_bias_obs0_model12 = df_results['errors_cross_obs0_model12'].apply(lambda x: x['bias']).mean()
    
    labels = ['Regular Band 0', 'Regular Band 1', 'Regular Band 2', 'Model0+Obs12', 'Obs0+Model12']
    values = [regular_bias0, regular_bias1, regular_bias2, cross_bias_model0_obs12, cross_bias_obs0_model12]
    
    plt.figure(figsize=(14, 8))
    bars = plt.bar(labels, values, color=['#2C7BB6', '#D7191C', '#FDAE61', '#7570B3', '#E7298A'])
    plt.ylabel('Bias (m/s)')
    plt.title('Cross-Scale Impact Analysis')
    plt.grid(alpha=0.3)
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{height:.2f}', ha='center', va='bottom')
    
    plt.savefig(enhanced_output_path / 'cross_scale_impact.png', dpi=300)
    plt.close()
    
    # Analyze spectral coherence
    print("Analyzing spectral coherence...")
    
    coherence0 = df_results['coherence_metrics'].apply(lambda x: x['band0']).mean()
    coherence1 = df_results['coherence_metrics'].apply(lambda x: x['band1']).mean()
    coherence2 = df_results['coherence_metrics'].apply(lambda x: x['band2']).mean()
    
    plt.figure(figsize=(12, 8))
    plt.bar(bands, [coherence0, coherence1, coherence2], color=['#2C7BB6', '#D7191C', '#FDAE61'])
    plt.ylabel('Coherence')
    plt.title('Spectral Coherence Analysis')
    plt.grid(alpha=0.3)
    plt.savefig(enhanced_output_path / 'spectral_coherence.png', dpi=300)
    plt.close()
    
    # Analyze scale-dependent sensitivity
    print("Analyzing scale-dependent sensitivity...")
    
    sensitivity0 = df_results['sensitivity_metrics'].apply(lambda x: x['band0']).mean()
    sensitivity1 = df_results['sensitivity_metrics'].apply(lambda x: x['band1']).mean()
    sensitivity2 = df_results['sensitivity_metrics'].apply(lambda x: x['band2']).mean()
    
    plt.figure(figsize=(12, 8))
    plt.bar(bands, [sensitivity0, sensitivity1, sensitivity2], color=['#2C7BB6', '#D7191C', '#FDAE61'])
    plt.ylabel('Sensitivity (∆sigma0/∆wspd)')
    plt.title('Scale-Dependent Sensitivity Analysis')
    plt.grid(alpha=0.3)
    plt.savefig(enhanced_output_path / 'scale_sensitivity.png', dpi=300)
    plt.close()
    
    # Generate summary report
    print("Generating enhanced analysis summary report...")
    
    with open(enhanced_output_path / "enhanced_analysis_summary.txt", "w") as f:
        f.write("Enhanced Scale Dependency Analysis of CMOD5N\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("1. Direct Sigma0 Comparison:\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Band 0 (k < 0.1) - Bias: {sigma0_bias0:.6f}, RMSE: {sigma0_rmse0:.6f}\n")
        f.write(f"  Band 1 (0.1 < k < 0.3) - Bias: {sigma0_bias1:.6f}, RMSE: {sigma0_rmse1:.6f}\n")
        f.write(f"  Band 2 (k > 0.3) - Bias: {sigma0_bias2:.6f}, RMSE: {sigma0_rmse2:.6f}\n\n")
        
        f.write("2. Scale-Specific Transfer Function Analysis:\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Band 0 (k < 0.1) - Model-to-Observed Ratio: {ratio0:.6f}\n")
        f.write(f"  Band 1 (0.1 < k < 0.3) - Model-to-Observed Ratio: {ratio1:.6f}\n")
        f.write(f"  Band 2 (k > 0.3) - Model-to-Observed Ratio: {ratio2:.6f}\n\n")
        
        f.write("3. Cross-Scale Impact Analysis:\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Regular Band 0 - Bias: {regular_bias0:.6f}\n")
        f.write(f"  Regular Band 1 - Bias: {regular_bias1:.6f}\n")
        f.write(f"  Regular Band 2 - Bias: {regular_bias2:.6f}\n")
        f.write(f"  Model0+Obs12 - Bias: {cross_bias_model0_obs12:.6f}\n")
        f.write(f"  Obs0+Model12 - Bias: {cross_bias_obs0_model12:.6f}\n\n")
        
        f.write("4. Spectral Coherence Analysis:\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Band 0 (k < 0.1) - Coherence: {coherence0:.6f}\n")
        f.write(f"  Band 1 (0.1 < k < 0.3) - Coherence: {coherence1:.6f}\n")
        f.write(f"  Band 2 (k > 0.3) - Coherence: {coherence2:.6f}\n\n")
        
        f.write("5. Scale-Dependent Sensitivity Analysis:\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Band 0 (k < 0.1) - Sensitivity: {sensitivity0:.6f}\n")
        f.write(f"  Band 1 (0.1 < k < 0.3) - Sensitivity: {sensitivity1:.6f}\n")
        f.write(f"  Band 2 (k > 0.3) - Sensitivity: {sensitivity2:.6f}\n\n")
        
        f.write("Summary Interpretation:\n")
        f.write("-" * 40 + "\n")
        
        # Determine if GMF has scale dependency from new metrics
        has_scale_dependency = False
        scale_dependency_evidence = []
        
        # Check direct sigma0 comparison
        sigma0_diffs = [sigma0_bias0, sigma0_bias1, sigma0_bias2]
        if max(sigma0_diffs) - min(sigma0_diffs) > 0.1:
            has_scale_dependency = True
            scale_dependency_evidence.append("Significant difference in direct sigma0 bias across scales")
        
        # Check ratio deviation from 1.0
        ratio_diffs = [abs(ratio0 - 1.0), abs(ratio1 - 1.0), abs(ratio2 - 1.0)]
        if max(ratio_diffs) > 0.1:
            has_scale_dependency = True
            scale_dependency_evidence.append("Transfer function ratio deviates significantly from 1.0")
        
        # Check coherence variations
        coherence_vals = [coherence0, coherence1, coherence2]
        if max(coherence_vals) - min(coherence_vals) > 0.2:
            has_scale_dependency = True
            scale_dependency_evidence.append("Significant variation in coherence across scales")
        
        # Check sensitivity variations
        sensitivity_vals = [sensitivity0, sensitivity1, sensitivity2]
        if max(sensitivity_vals) - min(sensitivity_vals) > 0.1:
            has_scale_dependency = True
            scale_dependency_evidence.append("Significant variation in sensitivity across scales")
        
        if has_scale_dependency:
            f.write("The enhanced analysis CONFIRMS scale dependency in the CMOD5N GMF based on:\n")
            for evidence in scale_dependency_evidence:
                f.write(f"- {evidence}\n")
        else:
            f.write("The enhanced analysis does not provide strong additional evidence of scale dependency.\n")
        
        f.write("\nRecommendations for GMF Improvement:\n")
        f.write("-" * 40 + "\n")
        f.write("Based on the analysis, the following improvements to CMOD5N could be considered:\n")
        
        if abs(ratio0 - 1.0) > 0.1:
            f.write(f"- Adjust large-scale (Band 0) transfer function by factor ~{1/ratio0:.4f}\n")
        
        if abs(ratio1 - 1.0) > 0.1:
            f.write(f"- Adjust medium-scale (Band 1) transfer function by factor ~{1/ratio1:.4f}\n")
        
        if abs(ratio2 - 1.0) > 0.1:
            f.write(f"- Adjust small-scale (Band 2) transfer function by factor ~{1/ratio2:.4f}\n")
    
    print(f"Enhanced analysis complete. Results saved to {enhanced_output_path}")
    return has_scale_dependency, scale_dependency_evidence

def main():
    """Main function to execute the workflow."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Analyze scale-dependent wind stress variability.')
    parser.add_argument('--processed_data', type=str, 
                        # default='/home/gfeirreiraseco/msc-thesis/processed_data',
                        default='processed_data',
                        help='Path to processed data directory.')
    parser.add_argument('--sardata2020', type=str, 
                        # default='projects/fluxsar/data/Sentinel1/WV/2020',
                        default = "processed_data/SAR/2020",
                        help='Path to SAR data for 2020.')
    parser.add_argument('--sardata2021', type=str, 
                        # default='projects/fluxsar/data/Sentinel1/WV/2021',
                        default = "processed_data/SAR/2020",
                        help='Path to SAR data for 2021.')
    parser.add_argument('--output', type=str, default='msc-thesis/results',
                        help='Path to output directory.')
    parser.add_argument('--num_processes', type=int, default=48,
                        help='Number of processes to use for parallel processing.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility.')
    parser.add_argument('--enhanced_analysis', action='store_true',
                        help='Enable enhanced scale dependency analysis.')
    args = parser.parse_args()
    
    # Set paths
    processed_data_path = Path(args.processed_data)
    sar_data_path_2020 = Path(args.sardata2020)
    sar_data_path_2021 = Path(args.sardata2021)
    output_path = Path(args.output)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Load data from parquet files
    print("Loading data from parquet files...")
    try:
        df_wv1 = pd.read_parquet(processed_data_path / "wv1_complete.parquet")
        df_wv2 = pd.read_parquet(processed_data_path / "wv2_complete.parquet")
        print(f"Loaded {len(df_wv1)} WV1 records and {len(df_wv2)} WV2 records.")
    except Exception as e:
        print(f"Error loading parquet files: {e}")
        print("Make sure the 'processed_data' directory contains the required parquet files.")
        return
    
    # Check missing SAR data paths
    print("Checking SAR data paths...")
    df_wv1 = df_wv1[df_wv1['path_to_sar_file'].notna()]
    df_wv2 = df_wv2[df_wv2['path_to_sar_file'].notna()]
    print(f"After removing NaN paths: {len(df_wv1)} WV1 records and {len(df_wv2)} WV2 records.")
    
    # Create records for parallel processing
    print("Creating records for parallel processing...")
    records_wv1 = []
    for _, row in df_wv1.iterrows():
        records_wv1.append({
            # 'sar_filepath': row['path_to_sar_file'],
            'sar_filepath': row['path_to_sar_file'].replace("/projects/fluxsar/data/", "processed_data/"),
            'era5_wspd': row['wspd'],
            'era5_wdir': row['wdir_deg_from_north'],  
            'seed': args.seed
        })
    
    records_wv2 = []
    for _, row in df_wv2.iterrows():
        records_wv2.append({
            # 'sar_filepath': row['path_to_sar_file'],
            'sar_filepath': row['path_to_sar_file'].replace("/projects/fluxsar/data/", "processed_data/"),
            'era5_wspd': row['wspd'],
            'era5_wdir': row['wdir_deg_from_north'],  
            'seed': args.seed
        })
    
    # Process SAR files in parallel
    print(f"Processing {len(records_wv1)} WV1 files in parallel using {args.num_processes} processes...")
    start_time = datetime.now()
    
    # Process WV1 files
    with Pool(processes=args.num_processes) as pool:
        results_wv1 = list(tqdm.tqdm(
            pool.imap_unordered(process_record, records_wv1),
            total=len(records_wv1),
        ))
    
    # Filter out None results
    results_wv1 = [result for result in results_wv1 if result is not None]
    
    print(f"Processing {len(records_wv2)} WV2 files in parallel...")
    
    # Process WV2 files
    with Pool(processes=args.num_processes) as pool:
        results_wv2 = list(tqdm.tqdm(
            pool.imap_unordered(process_record, records_wv2),
            total=len(records_wv2),
        ))
    
    # Filter out None results
    results_wv2 = [result for result in results_wv2 if result is not None]
    
    end_time = datetime.now()
    print(f"Processing completed in {end_time - start_time}.")
    print(f"Successfully processed {len(results_wv1)} WV1 files and {len(results_wv2)} WV2 files.")
    
    # Convert results to DataFrames
    df_results_wv1 = pd.DataFrame(results_wv1)
    df_results_wv2 = pd.DataFrame(results_wv2)
    
    # Save results to parquet files
    print("Saving results to parquet files...")
    df_results_wv1.to_parquet(output_path / "wv1_results.parquet")
    df_results_wv2.to_parquet(output_path / "wv2_results.parquet")
    
    # Plot error distributions
    print("Plotting error distributions...")
    plot_error_distributions(df_results_wv1, output_path / "plots_wv1")
    plot_error_distributions(df_results_wv2, output_path / "plots_wv2")
    
    # Combined KW test
    print("Performing combined Kruskal-Wallis test...")
    all_bias0 = df_results_wv1['errors_band0'].apply(lambda x: x['bias']).tolist() + df_results_wv2['errors_band0'].apply(lambda x: x['bias']).tolist()
    all_bias1 = df_results_wv1['errors_band1'].apply(lambda x: x['bias']).tolist() + df_results_wv2['errors_band1'].apply(lambda x: x['bias']).tolist()
    all_bias2 = df_results_wv1['errors_band2'].apply(lambda x: x['bias']).tolist() + df_results_wv2['errors_band2'].apply(lambda x: x['bias']).tolist()
    
    statistic, p_value, is_significant = kruskal_wallis_test(all_bias0, all_bias1, all_bias2)
    
    # Perform enhanced analysis if requested
    enhanced_evidence = []
    has_enhanced_scale_dependency = False
    if args.enhanced_analysis:
        print("Performing enhanced scale dependency analysis...")
        # Perform enhanced analysis on WV1 data
        print("Analyzing WV1 data...")
        has_scale_dependency_wv1, evidence_wv1 = analyze_scale_dependent_metrics(
            df_results_wv1, output_path / "wv1_enhanced"
        )
        
        # Perform enhanced analysis on WV2 data
        print("Analyzing WV2 data...")
        has_scale_dependency_wv2, evidence_wv2 = analyze_scale_dependent_metrics(
            df_results_wv2, output_path / "wv2_enhanced"
        )
        
        # Combine evidence from both analyses
        enhanced_evidence = list(set(evidence_wv1 + evidence_wv2))
        has_enhanced_scale_dependency = has_scale_dependency_wv1 or has_scale_dependency_wv2
    
    # Write conclusion to file
    with open(output_path / "conclusion.txt", "w") as f:
        f.write("Statistical Assessment of Scale-Dependent Wind Stress Variability\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("Kruskal-Wallis Test Results:\n")
        f.write(f"H statistic: {statistic:.4f}\n")
        f.write(f"p-value: {p_value:.10f}\n\n")
        
        if is_significant:
            f.write("CONCLUSION: GMF is Scale-Dependent (Reject H_0)\n")
            f.write("The analysis provides evidence that the wind stress field exhibits\n")
            f.write("statistically significant scale-dependent patterns.\n")
        else:
            f.write("CONCLUSION: GMF is Not Scale-Dependent (Accept H_0)\n")
            f.write("The analysis does not provide sufficient evidence to conclude that\n")
            f.write("the wind stress field exhibits scale-dependent patterns.\n")
        
        # Add enhanced analysis results if available
        if args.enhanced_analysis:
            f.write("\nEnhanced Scale Dependency Analysis:\n")
            f.write("-" * 40 + "\n")
            
            if has_enhanced_scale_dependency:
                f.write("The enhanced analysis CONFIRMS scale dependency in the CMOD5N GMF based on:\n")
                for evidence in enhanced_evidence:
                    f.write(f"- {evidence}\n")
            else:
                f.write("The enhanced analysis does not provide strong additional evidence of scale dependency.\n")
        
        # Add summary statistics
        f.write("\nSummary Statistics:\n")
        f.write("-" * 40 + "\n")
        f.write("WV1 Analysis:\n")
        f.write(f"  Number of processed files: {len(results_wv1)}\n")
        f.write(f"  Scale-dependent files: {df_results_wv1['is_scale_dependent'].sum()} ({df_results_wv1['is_scale_dependent'].mean() * 100:.1f}%)\n")
        f.write(f"  Mean bias - Band 0: {df_results_wv1['errors_band0'].apply(lambda x: x['bias']).mean():.4f} m/s\n")
        f.write(f"  Mean bias - Band 1: {df_results_wv1['errors_band1'].apply(lambda x: x['bias']).mean():.4f} m/s\n")
        f.write(f"  Mean bias - Band 2: {df_results_wv1['errors_band2'].apply(lambda x: x['bias']).mean():.4f} m/s\n")
        
        if args.enhanced_analysis:
            f.write("\n  Enhanced Analysis Metrics:\n")
            f.write(f"  Direct sigma0 bias - Band 0: {df_results_wv1['sigma0_diff_band0'].apply(lambda x: x['bias']).mean():.6f}\n")
            f.write(f"  Direct sigma0 bias - Band 1: {df_results_wv1['sigma0_diff_band1'].apply(lambda x: x['bias']).mean():.6f}\n")
            f.write(f"  Direct sigma0 bias - Band 2: {df_results_wv1['sigma0_diff_band2'].apply(lambda x: x['bias']).mean():.6f}\n")
            f.write(f"  Model-to-observed ratio - Band 0: {df_results_wv1['ratio_band0'].mean():.6f}\n")
            f.write(f"  Model-to-observed ratio - Band 1: {df_results_wv1['ratio_band1'].mean():.6f}\n")
            f.write(f"  Model-to-observed ratio - Band 2: {df_results_wv1['ratio_band2'].mean():.6f}\n")
        
        f.write("\nWV2 Analysis:\n")
        f.write(f"  Number of processed files: {len(results_wv2)}\n")
        f.write(f"  Scale-dependent files: {df_results_wv2['is_scale_dependent'].sum()} ({df_results_wv2['is_scale_dependent'].mean() * 100:.1f}%)\n")
        f.write(f"  Mean bias - Band 0: {df_results_wv2['errors_band0'].apply(lambda x: x['bias']).mean():.4f} m/s\n")
        f.write(f"  Mean bias - Band 1: {df_results_wv2['errors_band1'].apply(lambda x: x['bias']).mean():.4f} m/s\n")
        f.write(f"  Mean bias - Band 2: {df_results_wv2['errors_band2'].apply(lambda x: x['bias']).mean():.4f} m/s\n")
        
        if args.enhanced_analysis:
            f.write("\n  Enhanced Analysis Metrics:\n")
            f.write(f"  Direct sigma0 bias - Band 0: {df_results_wv2['sigma0_diff_band0'].apply(lambda x: x['bias']).mean():.6f}\n")
            f.write(f"  Direct sigma0 bias - Band 1: {df_results_wv2['sigma0_diff_band1'].apply(lambda x: x['bias']).mean():.6f}\n")
            f.write(f"  Direct sigma0 bias - Band 2: {df_results_wv2['sigma0_diff_band2'].apply(lambda x: x['bias']).mean():.6f}\n")
            f.write(f"  Model-to-observed ratio - Band 0: {df_results_wv2['ratio_band0'].mean():.6f}\n")
            f.write(f"  Model-to-observed ratio - Band 1: {df_results_wv2['ratio_band1'].mean():.6f}\n")
            f.write(f"  Model-to-observed ratio - Band 2: {df_results_wv2['ratio_band2'].mean():.6f}\n")
    
    print(f"Analysis complete. Results saved to {output_path}")
    
    # Print conclusion
    print("\nKruskal-Wallis Test Results:")
    print(f"H statistic: {statistic:.4f}")
    print(f"p-value: {p_value:.10f}")
    
    if is_significant:
        print("\nCONCLUSION: GMF is Scale-Dependent (Reject H_0)")
        print("The analysis provides evidence that the wind stress field exhibits")
        print("statistically significant scale-dependent patterns.")
        
        if args.enhanced_analysis and has_enhanced_scale_dependency:
            print("\nEnhanced analysis provides additional evidence of scale dependency:")
            for evidence in enhanced_evidence:
                print(f"- {evidence}")
    else:
        print("\nCONCLUSION: GMF is Not Scale-Dependent (Accept H_0)")
        print("The analysis does not provide sufficient evidence to conclude that")
        print("the wind stress field exhibits scale-dependent patterns.")
        
        if args.enhanced_analysis and has_enhanced_scale_dependency:
            print("\nHowever, enhanced analysis suggests possible scale dependency:")
            for evidence in enhanced_evidence:
                print(f"- {evidence}")

if __name__ == '__main__':
    main()
    
# This script implements enhanced scale dependency analysis for CMOD5N function
# Based on recommendations from feedback to include:
# 1. Direct comparison of observed vs. modeled sigma0 bands
# 2. Scale-specific transfer function analysis
# 3. Cross-scale impact analysis
# 4. Spectral coherence analysis
# 5. Scale-dependent sensitivity analysis