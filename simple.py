#!/usr/bin/env python3
import os
import pandas as pd

csv_path = "~/msc-thesis/2020_filename.csv"
folder_path = "/projects/fluxsar/data/Sentinel1/WV/2020"

def clean_folder_against_csv(folder_path=folder_path, csv_path=csv_path):
    """
    Compares files in `folder_path` to filenames in `csv_path`.
    Prints counts of matching and non-matching files.
    Deletes non-matching files only after displaying the counts.

    Parameters:
    folder_path (str): Path to the folder with files to clean
    csv_path (str): Path to the CSV file with valid filenames (1st column)
    """
    valid_files = set(pd.read_csv(csv_path, header=None)[0])
    all_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    matching = [f for f in all_files if f in valid_files]
    non_matching = [f for f in all_files if f not in valid_files]

    print(f'Matching files: {len(matching)}')
    print(f'Non-matching files: {len(non_matching)}')
