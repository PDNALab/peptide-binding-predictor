#Directory: working_data/scripts
import os
import pandas as pd

def load_folder_data(folder_path):
    """
    Load data from a given folder.
    Reads:
    - combined.txt: assumes whitespace-separated values, with binding label in column index 1 
      (i.e., second column)
    - output_sequences.txt: assumes sequences are in column index 1 if there are at least two columns,
      otherwise in column index 0.
      
    Parameters:
    -----------
    folder_path : str
        Path to the folder containing combined.txt and output_sequences.txt files
      
    Returns:
    --------
    pandas.DataFrame or None
        DataFrame with columns 'sequence' and 'binding' (binding: 1 for 'Y' and 0 for 'N')
        Returns None if there's an error loading either file
    """
    combined_file = os.path.join(folder_path, "combined.txt")
    seq_file = os.path.join(folder_path, "output_sequences.txt")
    
    # Load combined.txt (no header, whitespace delimited)
    try:
        combined_df = pd.read_csv(combined_file, sep="\s+", header=None)
    except Exception as e:
        print(f"Error loading combined.txt in {folder_path}: {e}")
        return None
    
    # Load output_sequences.txt (no header, whitespace delimited)
    try:
        seq_df = pd.read_csv(seq_file, sep="\s+", header=None)
    except Exception as e:
        print(f"Error loading output_sequences.txt in {folder_path}: {e}")
        return None
    
    # Ensure both files have the same number of rows
    if len(combined_df) != len(seq_df):
        print("Mismatch in number of rows between combined.txt and output_sequences.txt")
        return None
    
    # Extract binding label from combined_df: column index 1 (second column)
    # Convert 'Y' to 1 and 'N' to 0.
    binding = combined_df.iloc[:, 1].apply(lambda x: 1 if str(x).strip().upper()=='Y' else 0)
    
    # Extract sequences from seq_df.
    # If there are at least two columns, assume the second column holds the sequence.
    if seq_df.shape[1] >= 2:
        sequences = seq_df.iloc[:, 1]
    else:
        sequences = seq_df.iloc[:, 0]
        
    # Combine into a DataFrame
    df = pd.DataFrame({"sequence": sequences, "binding": binding})
    return df

# Testing the function on one folder in final_data.
# Assumes 'final_data' is located one directory above 'working_data'.
final_data_dir = os.path.join("../..", "final_data")
folder_list = [f for f in os.listdir(final_data_dir) if os.path.isdir(os.path.join(final_data_dir, f))]

if folder_list:
    test_folder = os.path.join(final_data_dir, folder_list[0])
    df_test = load_folder_data(test_folder)
    if df_test is not None:
        print("Loaded data from folder:", folder_list[0])
        print(df_test.head())
    else:
        print("Failed to load data from folder:", folder_list[0])
else:
    print("No folders found in final_data.")