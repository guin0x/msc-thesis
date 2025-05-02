#!/usr/bin/env python

"""
Scale-Dependent Analysis of Wind Stress Variability

This script implements the workflow for analyzing scale-dependent wind stress 
variability using Sentinel-1 WV SAR data, following the coupled perturbation model.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from multiprocessing import Pool
import argparse
from datetime import datetime

# Import custom functions
from utils.functions_v2 import (
    process_sar_file,
    plot_error_distributions,
    kruskal_wallis_test
)

def main():
    """Main function to execute the workflow."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Analyze scale-dependent wind stress variability.')
    parser.add_argument('--processed_data', type=str, default='processed_data',
                        help='Path to processed data directory.')
    parser.add_argument('--sardata2020', type=str, default='projects/flux/sardata/Sentinel1/WV/2020',
                        help='Path to SAR data for 2020.')
    parser.add_argument('--sardata2021', type=str, default='projects/flux/sardata/Sentinel1/WV/2021',
                        help='Path to SAR data for 2021.')
    parser.add_argument('--output', type=str, default='results',
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
            'era5_wspd': row['wspd'],
            'era5_wdir': np.rad2deg(row['wdir']) % 360,  # Convert from radians to degrees
            'seed': args.seed
        })
    
    records_wv2 = []
    for _, row in df_wv2.iterrows():
        records_wv2.append({
            'sar_filepath': row['path_to_sar_file'],
            'era5_wspd': row['wspd'],
            'era5_wdir': np.rad2deg(row['wdir']) % 360,  # Convert from radians to degrees
            'seed': args.seed
        })
    
    # Process SAR files in parallel
    print(f"Processing {len(records_wv1)} WV1 files in parallel using {args.num_processes} processes...")
    start_time = datetime.now()
    
    # Helper function for parallel processing
    def process_record(record):
        return process_sar_file(record['sar_filepath'], record['era5_wspd'], record['era5_wdir'], record['seed'])
    
    # Process WV1 files
    with Pool(processes=args.num_processes) as pool:
        results_wv1 = pool.map(process_record, records_wv1)
    
    # Filter out None results
    results_wv1 = [result for result in results_wv1 if result is not None]
    
    print(f"Processing {len(records_wv2)} WV2 files in parallel...")
    
    # Process WV2 files
    with Pool(processes=args.num_processes) as pool:
        results_wv2 = pool.map(process_record, records_wv2)
    
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
    
    # Write conclusion to file
    with open(output_path / "conclusion.txt", "w") as f:
        f.write("Statistical Assessment of Scale-Dependent Wind Stress Variability\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("Kruskal-Wallis Test Results:\n")
        f.write(f"H statistic: {statistic:.4f}\n")
        f.write(f"p-value: {p_value:.10f}\n\n")
        
        if is_significant:
            f.write("CONCLUSION: GMF is Scale-Dependent (Reject H₀)\n")
            f.write("The analysis provides evidence that the wind stress field exhibits\n")
            f.write("statistically significant scale-dependent patterns.\n")
        else:
            f.write("CONCLUSION: GMF is Not Scale-Dependent (Accept H₀)\n")
            f.write("The analysis does not provide sufficient evidence to conclude that\n")
            f.write("the wind stress field exhibits scale-dependent patterns.\n")
        
        # Add summary statistics
        f.write("\nSummary Statistics:\n")
        f.write("-" * 40 + "\n")
        f.write("WV1 Analysis:\n")
        f.write(f"  Number of processed files: {len(results_wv1)}\n")
        f.write(f"  Scale-dependent files: {df_results_wv1['is_scale_dependent'].sum()} ({df_results_wv1['is_scale_dependent'].mean() * 100:.1f}%)\n")
        f.write(f"  Mean bias - Band 0: {df_results_wv1['errors_band0'].apply(lambda x: x['bias']).mean():.4f} m/s\n")
        f.write(f"  Mean bias - Band 1: {df_results_wv1['errors_band1'].apply(lambda x: x['bias']).mean():.4f} m/s\n")
        f.write(f"  Mean bias - Band 2: {df_results_wv1['errors_band2'].apply(lambda x: x['bias']).mean():.4f} m/s\n")
        
        f.write("\nWV2 Analysis:\n")
        f.write(f"  Number of processed files: {len(results_wv2)}\n")
        f.write(f"  Scale-dependent files: {df_results_wv2['is_scale_dependent'].sum()} ({df_results_wv2['is_scale_dependent'].mean() * 100:.1f}%)\n")
        f.write(f"  Mean bias - Band 0: {df_results_wv2['errors_band0'].apply(lambda x: x['bias']).mean():.4f} m/s\n")
        f.write(f"  Mean bias - Band 1: {df_results_wv2['errors_band1'].apply(lambda x: x['bias']).mean():.4f} m/s\n")
        f.write(f"  Mean bias - Band 2: {df_results_wv2['errors_band2'].apply(lambda x: x['bias']).mean():.4f} m/s\n")
    
    print(f"Analysis complete. Results saved to {output_path}")
    
    # Print conclusion
    print("\nKruskal-Wallis Test Results:")
    print(f"H statistic: {statistic:.4f}")
    print(f"p-value: {p_value:.10f}")
    
    if is_significant:
        print("\nCONCLUSION: GMF is Scale-Dependent (Reject H₀)")
        print("The analysis provides evidence that the wind stress field exhibits")
        print("statistically significant scale-dependent patterns.")
    else:
        print("\nCONCLUSION: GMF is Not Scale-Dependent (Accept H₀)")
        print("The analysis does not provide sufficient evidence to conclude that")
        print("the wind stress field exhibits scale-dependent patterns.")

if __name__ == '__main__':
    main()
