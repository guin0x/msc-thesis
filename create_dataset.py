#!/usr/bin/env python3
"""
Process and filter SAR data based on wind speed ranges and stability conditions.
This script loads parquet files, creates filtered dataframes, adds filename information,
and saves the processed data.
"""

import pandas as pd
from glob import glob
import os
import logging
import numpy as np
from utils.functions import create_filtered_dfs, rename_filename, check_file_exists, create_path_to_sar_file


def main():
    """Main function to execute the data processing pipeline."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Load data
    logger.info("Loading parquet files")
    data_path = '/projects/fluxsar/data/Sentinel1/dataframes_w_chen_classes'
    output_dir = '/home/gfeirreiraseco/msc-thesis/processed_data'
    
    all_df_files = glob(os.path.join(data_path, '*.parquet'))
    logger.info(f"Found {len(all_df_files)} files")
    
    all_dfs = [pd.read_parquet(f) for f in all_df_files]
    df = pd.concat(all_dfs)
    df = df.dropna()
    logger.info(f"Loaded {len(df)} total records")
    
    # Split by wave type
    wv1_df = df[df.wm_type == 'wv1']
    wv2_df = df[df.wm_type == 'wv2']
    logger.info(f"Split into {len(wv1_df)} WV1 and {len(wv2_df)} WV2 records")
    
    # Create filtered dataframes
    wv1_filtered = create_filtered_dfs(wv1_df, 'wv1')
    wv2_filtered = create_filtered_dfs(wv2_df, 'wv2')
    
    all_filtered_dfs = {**wv1_filtered, **wv2_filtered}
    
    # Process each filtered dataframe more elegantly using a loop
    for df_name, df in all_filtered_dfs.items():
        logger.info(f"Processing {df_name}")
        # Add renamed filename
        df.loc[:, "renamed_filename"] = df.loc[:, "filename"].apply(rename_filename).copy()

        df.loc[:, "wdir_deg_from_north"] = np.rad2deg(df.wdir) % 360
        
        # Check file existence
        results = [check_file_exists(f) for f in df["renamed_filename"].values]
        df.loc[:, "exists_ok"] = [r[0] for r in results]
        
        df.loc[:, "path_to_sar_file"] = df.apply(create_path_to_sar_file, axis=1)
        
        # Drop NaNs and convert path to string
        # df = df.dropna()
        
        
        # Save to parquet
        output_path = os.path.join(output_dir, f"batch2_{df_name}.parquet")
        logger.info(f"Saving {df_name} with {len(df)} rows to {output_path}")
        df.to_parquet(output_path)
    
    logger.info("Processing complete")


if __name__ == "__main__":
    main()