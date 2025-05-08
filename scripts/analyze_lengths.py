import os
import pandas as pd
"""
This script analyzes the length distribution of biological sequences loaded from a 
shuffled dataset. 
It reads data from 'shuffled_data.csv', calculates the length of each sequence, and outputs
statistical information about these lengths (count, mean, standard deviation, min/max values,
and percentiles). 
The analysis helps to understand the size distribution of sequences in the dataset.
"""

# Set up paths - find the parent directory of the script
working_dir = os.path.join(os.path.dirname(__file__), '..')
# Path to the shuffled dataset
csv_path = os.path.join(working_dir, 'results', 'shuffled_data.csv')

# Load the shuffled dataset
df = pd.read_csv(csv_path)

# Add a new column 'length' that contains the length of each sequence
df['length'] = df['sequence'].apply(len)

# Display statistical summary of the sequence lengths
# This includes count, mean, std, min, 25%, 50%, 75%, max
print("\nBasic Stats for Sequence Lengths:")
print(df['length'].describe())