import os
import pandas as pd

"""
This script processes a dataset of amino acid sequences for machine learning modeling.
It performs the following tasks:
1. Loads a dataset of amino acid sequences from 'shuffled_data.csv'
2. Creates a mapping to convert amino acid letters to numerical indices
3. Encodes each sequence into a list of integers using this mapping
4. Extracts features (X) and labels (y) from the encoded sequences
5. Saves the encoded data as NumPy arrays for use in machine learning models

The encoding is necessary because machine learning models require numerical inputs
rather than text. The script handles unknown amino acids by encoding them as -1.
"""

# Set up paths
working_dir = os.path.join(os.path.dirname(__file__), '..')
csv_path = os.path.join(working_dir, 'results', 'shuffled_data.csv')
# Load the shuffled dataset
df = pd.read_csv(csv_path)

# Define a mapping from amino acid letters to integer indices
# This is needed for the neural network model which requires numerical input
amino_acids = 'ACDEFGHIKLMNPQRSTVWY'  # 20 standard amino acids
aa_to_index = {aa: idx for idx, aa in enumerate(amino_acids)}

# Helper function to convert an amino acid sequence to a list of integers
def encode_sequence(seq):
    """
    Convert amino acid letters to integers using the aa_to_index mapping.
    
    Parameters:
    -----------
    seq : str
        String of amino acid letters
        
    Returns:
    --------
    list
        List of integers corresponding to each amino acid
        Unknown amino acids are encoded as -1
    """
    return [aa_to_index.get(aa, -1) for aa in seq]  # -1 for unknowns

# Apply the encoding function to each sequence
df['encoded'] = df['sequence'].apply(encode_sequence)

# Display an example of original and encoded sequence
print("Original:", df.loc[0, 'sequence'])
print("Encoded: ", df.loc[0, 'encoded'])

import numpy as np

# Convert to NumPy arrays for use with machine learning models
# X contains the encoded sequences, y contains the binding labels
X = np.array(df['encoded'].tolist())
y = df['binding'].values

# Create the output directory for the NumPy arrays
output_dir = os.path.join(working_dir, 'npy')
os.makedirs(output_dir, exist_ok=True)

# Save the arrays to disk
np.save(os.path.join(output_dir, 'X.npy'), X)  # Save features
np.save(os.path.join(output_dir, 'y.npy'), y)  # Save labels

print(f"\nSaved encoded data to {output_dir}/X.npy and y.npy")