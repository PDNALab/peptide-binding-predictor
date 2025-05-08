# Directory: working_data/scripts
import os
import pandas as pd
from data_loader import load_folder_data  # Make sure data_loader.py is in working_data/scripts

def aggregate_data(final_data_dir, output_file):
   """
    Aggregates data from all folders in final_data_dir by loading each folder's combined.txt
    and output_sequences.txt files. The resulting DataFrame (with columns 'sequence' and 'binding')
    is saved as a CSV file at output_file.
    
    Parameters:
    -----------
    final_data_dir : str
        Path to the directory containing the data folders
    output_file : str
        Path where the aggregated data will be saved as CSV
    
    Returns:
    --------
    None
        Results are saved to disk at the specified output_file path
    """
    aggregated_list = []
    folder_list = [f for f in os.listdir(final_data_dir) if os.path.isdir(os.path.join(final_data_dir, f))]
    print(f"Found {len(folder_list)} folders in {final_data_dir}")
    
    for folder in folder_list:
        folder_path = os.path.join(final_data_dir, folder)
        df = load_folder_data(folder_path)
        if df is not None:
            aggregated_list.append(df)
        else:
            print(f"Skipping folder {folder} due to a loading error.")
    
    if aggregated_list:
        aggregated_df = pd.concat(aggregated_list, ignore_index=True)
        # Create the output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        aggregated_df.to_csv(output_file, index=False)
        print(f"Aggregated data saved to {output_file}")
    else:
        print("No data to aggregate.")

# Define the path to the final_data directory.
# Assumes final_data is one directory level up from working_data.
final_data_dir = os.path.join("../../", "final_data")

# Define the output CSV file path inside working_data/data.
# Fix the output path: create a 'data' directory at the same level as 'scripts'
output_file = os.path.join("..", "data", "aggregated_data.csv")

aggregate_data(final_data_dir, output_file)