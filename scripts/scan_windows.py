import os
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

"""
This script performs sliding window analysis on amino acid sequences to predict binding sites.
It performs the following tasks:

1. Loads a trained deep learning model for binding prediction
2. Processes a dataset of amino acid sequences from 'shuffled_data.csv'
3. For each sequence, creates all possible 25-residue windows (subsequences)
4. Encodes each window using the same amino acid to index mapping as during training
5. Predicts binding probability for each window using the loaded model
6. Calculates summary statistics for each sequence:
   - Maximum binding score (likely indicating the strongest binding site)
   - Mean binding score (indicating overall binding tendency)
7. Saves the prediction results to 'binding_predictions.csv' for further analysis

This sliding window approach allows identifying potential binding regions within longer 
sequences by examining each possible fixed-length segment independently.
"""

# Setup paths
working_dir = os.path.join(os.path.dirname(__file__), '..')
csv_path = os.path.join(working_dir, 'results', 'shuffled_data.csv')
model_path = os.path.join(working_dir, 'models', 'cnn_binding_model.h5')

# Load the trained binding prediction model
model = load_model(model_path)

# Define the same amino acid encoding used during training
amino_acids = 'ACDEFGHIKLMNPQRSTVWY'  # 20 standard amino acids
aa_to_index = {aa: idx for idx, aa in enumerate(amino_acids)}

def encode_sequence(seq):
    """
    Convert amino acid letters to numeric indices.
    
    Parameters:
    -----------
    seq : str
        String of amino acid letters
        
    Returns:
    --------
    list
        List of integers corresponding to each amino acid
        Unknown amino acids are encoded as 0 (A)
    """
    return [aa_to_index.get(aa, 0) for aa in seq]  # Fallback to A (0) for unknown amino acids

def sliding_windows(seq, size=25):
    """
    Generate all possible windows of specified size from a sequence.
    
    Parameters:
    -----------
    seq : str
        Input sequence
    size : int
        Window size (default: 25)
        
    Returns:
    --------
    list
        List of subsequences of length 'size'
    """
    return [seq[i:i+size] for i in range(len(seq) - size + 1)]

# Load the peptide dataset
df = pd.read_csv(csv_path)

# List to store the results for each sequence
results = []

# Process each sequence in the dataset
for i, row in df.iterrows():
    seq = row['sequence']
    label = row['binding']

    # Skip sequences that are too short for our window size
    if len(seq) < 25:
        continue

    # Generate all possible 25-residue windows from this sequence
    windows = sliding_windows(seq)
    
    # Encode each window for the model
    encoded = [encode_sequence(w) for w in windows]
    
    # Predict binding probability for each window
    # verbose=0 suppresses progress bar output
    predictions = model.predict(np.array(encoded), verbose=0).flatten()

    # Calculate summary statistics from the window predictions
    max_score = predictions.max()    # Highest binding score (most likely binding site)
    mean_score = predictions.mean()  # Average binding tendency across the sequence

    # Store the results for this sequence
    results.append({
        "sequence": seq,
        "true_label": label,
        "num_windows": len(windows),
        "max_score": max_score,
        "mean_score": mean_score
    })

# Save the prediction results to a CSV file
output_path = os.path.join(working_dir, 'results', 'binding_predictions.csv')
pd.DataFrame(results).to_csv(output_path, index=False)
print(f"Saved sliding window predictions to {output_path}")