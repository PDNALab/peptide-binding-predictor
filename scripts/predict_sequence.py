import os
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
working_dir = os.path.join(os.path.dirname(__file__), '..')
model_path = os.path.join(working_dir, 'models', 'cnn_binding_model.h5')
model = load_model(model_path)

# Define the same amino acid encoding used during training
amino_acids = 'ACDEFGHIKLMNPQRSTVWY'  # 20 standard amino acids
aa_to_index = {aa: idx for idx, aa in enumerate(amino_acids)}

def encode_sequence(seq):
    """
    Convert an amino acid sequence to numeric indices.
    
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
    return [aa_to_index.get(aa, 0) for aa in seq]  # 0 = 'A' fallback for unknown amino acids

def predict_binding(peptide):
    """
    Predict binding probability for a given peptide sequence.
    
    Handles three cases:
    1. Peptide shorter than 25 residues: pad with zeros
    2. Peptide exactly 25 residues: use directly
    3. Peptide longer than 25: use sliding window and take max score
    
    Parameters:
    -----------
    peptide : str
        Amino acid sequence to predict binding for
        
    Returns:
    --------
    float
        Binding probability score (0-1)
    """
    # Convert amino acid letters to integers
    encoded = encode_sequence(peptide)
    
    if len(encoded) < 25:
        # Handle sequences shorter than 25 residues by padding with zeros
        padded = encoded + [0] * (25 - len(encoded))
        input_arr = np.array([padded])
        return float(model.predict(input_arr)[0][0])
    
    elif len(encoded) == 25:
        # Handle sequences of exactly 25 residues (model's expected input length)
        input_arr = np.array([encoded])
        return float(model.predict(input_arr)[0][0])
    
    else:
        # Handle sequences longer than 25 residues using sliding window
        # Calculate scores for all possible 25-residue windows
        # Return the maximum score (most likely binding region)
        scores = []
        for i in range(len(encoded) - 24):
            window = encoded[i:i+25]
            input_arr = np.array([window])
            score = float(model.predict(input_arr)[0][0])
            scores.append(score)
        return max(scores)

# Example peptide to test the prediction function
peptide = "VHLTPEEKSAAVTALWGKVNVDEVGGEALGRLLVVYPWTQR"  # Sample sequence
score = predict_binding(peptide)

# Display results
print(f"\nBinding prediction score: {score:.4f}")
print("Prediction:", "BINDS" if score >= 0.5 else "DOES NOT BIND")