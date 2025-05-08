import os
import pandas as pd

"""
Load and Preview Data Script

This script performs the following operations:
1. Loads the aggregated data from a CSV file created by aggregate_data.py
2. Displays basic statistics and information about the dataset (head, info, column names)
3. Shows the distribution of binding labels (positive/negative samples)
4. Shuffles the dataset randomly to eliminate any potential ordering bias
5. Saves the shuffled dataset to a new CSV file in the results folder

This script is used for initial data exploration and preprocessing before 
model training or further analysis.
"""

# Set up paths - get the parent directory of the script's directory
working_dir = os.path.join(os.path.dirname(__file__), '..')
# Path to the aggregated data CSV created by aggregate_data.py
csv_path = os.path.join(working_dir, 'data', 'aggregated_data.csv')

# Load the aggregated data
df = pd.read_csv(csv_path)
print(df.head())  # Display the first 5 rows of the dataset

# Display general information about the DataFrame
print("\nDataFrame Info:")
print(df.info())  # Shows data types and non-null counts

# List the column names for reference
print("\nColumn Names:")
print(df.columns.tolist())

# Check the distribution of binding labels (0=no binding, 1=binding)
print("\nBinding Label Distribution:")
print(df['binding'].value_counts())  # Count of each unique value in 'binding' column

# Shuffle the dataset randomly to avoid any potential ordering bias
# random_state=42 ensures reproducibility of the shuffle
df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save the shuffled dataset to the results folder
output_path = os.path.join(working_dir, 'results', 'shuffled_data.csv')
df_shuffled.to_csv(output_path, index=False)

print(f"\nShuffled dataset saved to: {output_path}")