import os
import numpy as np

"""
This script validates the encoded sequence data that was created by encode_sequences.py.
It performs the following tasks:
1. Loads the NumPy arrays (X.npy for encoded sequences, y.npy for binding labels)
2. Displays the shape/dimensions of the arrays to verify correct data structure
3. Shows an example of the first encoded sequence and its corresponding label

This validation step is important to ensure that the data is properly formatted
before using it in machine learning models, confirming that the encoding process
worked as expected and the arrays have the correct dimensions.
"""

# Set up paths
working_dir = os.path.join(os.path.dirname(__file__), '..')
npy_dir = os.path.join(working_dir, 'npy')

# Load the saved NumPy arrays created by encode_sequences.py
X = np.load(os.path.join(npy_dir, 'X.npy'))  # Encoded sequences
y = np.load(os.path.join(npy_dir, 'y.npy'))  # Binding labels

# Display shape information (rows x columns) to verify the data dimensions
print("X shape:", X.shape)  # Should be (num_samples, sequence_length)
print("y shape:", y.shape)  # Should be (num_samples,)

# Show the first example from the dataset
print("\nFirst encoded sequence:", X[0])  # The integer encoding of the first sequence
print("Corresponding label:    ", y[0])  # The binding label (0 or 1) for the first sequence