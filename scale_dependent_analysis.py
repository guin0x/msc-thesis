#!/usr/bin/env python

"""
Scale-Dependent Analysis of Wind Stress Variability (v2)

This script implements a streamlined workflow for analyzing scale-dependent wind stress 
variability using Sentinel-1 WV SAR data. 

This version focuses on the most relevant analyses:
1. Transfer function ratios
2. Scale-dependent sensitivity
3. Cross-scale impact analysis

The script uses updated wavenumber bands:
- Band 0a: k < 0.05
- Band 0b: 0.05 ≤ k < 0.1
- Band 1: 0.1 ≤ k < 0.3
- Band 2: k ≥ 0.3
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
from utils.functions import (
    process_sar_file,
    plot_focused_analysis,
    kruskal_wallis_test,
    perform_statistical_tests,
    plot_statistical_test_results
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

def analyze_scale_dependency(df_results, output_path, band_names=None):
    """
    Perform focused scale dependency analysis.
    
    Parameters:
    -----------
    df_results : pandas.DataFrame
        DataFrame containing the analysis results
    output_path : Path
        Path to save the analysis results
    band_names : list, optional
        List of band names to analyze
    """
    if band_names is None:
        band_names = ['band0a', 'band0b', 'band0c', 'band1', 'band2']
    
    # Create directory for analysis
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Perform statistical tests
    print("Performing statistical tests for scale dependency...")
    test_results = perform_statistical_tests(df_results, band_names)
    
    # Visualize statistical test results
    plot_statistical_test_results(test_results, output_path)
    
    # Calculate mean values for key metrics
    ratio_means = {
        'band0a': df_results['ratio_band0a'].mean(),
        'band0b': df_results['ratio_band0b'].mean(),
        'band0c': df_results['ratio_band0c'].mean(),
        'band1': df_results['ratio_band1'].mean(),
        'band2': df_results['ratio_band2'].mean()
    }
    
    sensitivity_means = {}
    for band in band_names:
        sensitivity_means[band] = df_results['sensitivity_metrics'].apply(lambda x: x[band]).mean()
    
    # Determine if GMF has scale dependency from statistical tests
    has_scale_dependency = False
    scale_dependency_evidence = []

    if test_results['transfer_ratio']['significant']:
        has_scale_dependency = True
        scale_dependency_evidence.append("Statistically significant differences in transfer function ratios")
    
    if test_results['sensitivity']['significant']:
        has_scale_dependency = True
        scale_dependency_evidence.append("Statistically significant differences in scale-dependent sensitivity")
    
    if test_results['cross_scale']['significant']:
        has_scale_dependency = True
        scale_dependency_evidence.append("Statistically significant differences in cross-scale analysis")
    
    # Check transfer function ratio variation
    ratio_vals = list(ratio_means.values())
    if max(ratio_vals) - min(ratio_vals) > 0.1:
        has_scale_dependency = True
        scale_dependency_evidence.append("Significant variation in transfer function ratios across scales")
    
    # Check sensitivity variations
    sensitivity_vals = list(sensitivity_means.values())
    if max(sensitivity_vals) - min(sensitivity_vals) > 0.1:
        has_scale_dependency = True
        scale_dependency_evidence.append("Significant variation in sensitivity across scales")
    
    return has_scale_dependency, scale_dependency_evidence, test_results

def main():
    """Main function to execute the workflow."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Analyze scale-dependent wind stress variability.')
    parser.add_argument('--processed_data', type=str, 
                        default='/home/gfeirreiraseco/msc-thesis/processed_data',
                        # default='processed_data',
                        help='Path to processed data directory.')
    parser.add_argument('--sardata2020', type=str, 
                        default='projects/fluxsar/data/Sentinel1/WV/2020',
                        # default = "processed_data/SAR/2020",
                        help='Path to SAR data for 2020.')
    parser.add_argument('--sardata2021', type=str, 
                        default='projects/fluxsar/data/Sentinel1/WV/2021',
                        # default = "processed_data/SAR/2021",
                        help='Path to SAR data for 2021.')
    parser.add_argument('--output', type=str, default='msc-thesis/results_v2',
                        help='Path to output directory.')
    parser.add_argument('--num_processes', type=int, default=48,
                        help='Number of processes to use for parallel processing.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility.')
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
            'sar_filepath': row['path_to_sar_file'],
            # 'sar_filepath': row['path_to_sar_file'].replace("/projects/fluxsar/data/", "processed_data/"),
            'era5_wspd': row['wspd'],
            'era5_wdir': row['wdir_deg_from_north'],  
            'seed': args.seed
        })
    
    records_wv2 = []
    for _, row in df_wv2.iterrows():
        records_wv2.append({
            'sar_filepath': row['path_to_sar_file'],
            # 'sar_filepath': row['path_to_sar_file'].replace("/projects/fluxsar/data/", "processed_data/"),
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
    
    # Define band names
    band_names = ['band0a', 'band0b', 'band0c', 'band1', 'band2']
    
    # Analyze WV1 data
    print("Analyzing WV1 data...")
    has_scale_dependency_wv1, evidence_wv1, test_results_wv1 = analyze_scale_dependency(
        df_results_wv1, output_path / "wv1_analysis", band_names
    )
    
    # Analyze WV2 data
    print("Analyzing WV2 data...")
    has_scale_dependency_wv2, evidence_wv2, test_results_wv2 = analyze_scale_dependency(
        df_results_wv2, output_path / "wv2_analysis", band_names
    )
    
    # Combined analysis
    print("Performing combined analysis...")
    df_results_combined = pd.concat([df_results_wv1, df_results_wv2], ignore_index=True)
    
    has_scale_dependency_combined, evidence_combined, test_results_combined = analyze_scale_dependency(
        df_results_combined, output_path / "combined_analysis", band_names
    )
    
    # Combined KW test for all bands
    all_bias0a = df_results_wv1['errors_band0a'].apply(lambda x: x['bias']).tolist() + df_results_wv2['errors_band0a'].apply(lambda x: x['bias']).tolist()
    all_bias0b = df_results_wv1['errors_band0b'].apply(lambda x: x['bias']).tolist() + df_results_wv2['errors_band0b'].apply(lambda x: x['bias']).tolist()
    all_bias0c = df_results_wv1['errors_band0c'].apply(lambda x: x['bias']).tolist() + df_results_wv2['errors_band0c'].apply(lambda x: x['bias']).tolist()
    all_bias1 = df_results_wv1['errors_band1'].apply(lambda x: x['bias']).tolist() + df_results_wv2['errors_band1'].apply(lambda x: x['bias']).tolist()
    all_bias2 = df_results_wv1['errors_band2'].apply(lambda x: x['bias']).tolist() + df_results_wv2['errors_band2'].apply(lambda x: x['bias']).tolist()

    statistic, p_value, is_significant = kruskal_wallis_test(all_bias0a, all_bias0b, all_bias0c, all_bias1, all_bias2)
    
    # Print summary findings to console
    print("\nOverall Scale Dependency Assessment:")
    print(f"WV1 Analysis: {'SCALE DEPENDENT' if has_scale_dependency_wv1 else 'Not scale dependent'}")
    for evidence in evidence_wv1:
        print(f"  - {evidence}")
        
    print(f"\nWV2 Analysis: {'SCALE DEPENDENT' if has_scale_dependency_wv2 else 'Not scale dependent'}")
    for evidence in evidence_wv2:
        print(f"  - {evidence}")
        
    print(f"\nCombined Analysis: {'SCALE DEPENDENT' if has_scale_dependency_combined else 'Not scale dependent'}")
    for evidence in evidence_combined:
        print(f"  - {evidence}")
    
    print(f"\nStatistical Results (Combined Dataset):")
    print(f"Transfer Function Ratios: p-value = {test_results_combined['transfer_ratio']['p_value']:.6f}, {'SIGNIFICANT' if test_results_combined['transfer_ratio']['significant'] else 'not significant'}")
    print(f"Scale-Dependent Sensitivity: p-value = {test_results_combined['sensitivity']['p_value']:.6f}, {'SIGNIFICANT' if test_results_combined['sensitivity']['significant'] else 'not significant'}")
    print(f"Cross-Scale Analysis: p-value = {test_results_combined['cross_scale']['p_value']:.6f}, {'SIGNIFICANT' if test_results_combined['cross_scale']['significant'] else 'not significant'}")
    
    print(f"\nConclusion: GMF is {'SCALE-DEPENDENT' if is_significant else 'NOT scale-dependent'} based on combined Kruskal-Wallis test (p-value: {p_value:.6f})")
    
    print(f"\nAnalysis complete. Results saved to {output_path}")

if __name__ == '__main__':
    main()