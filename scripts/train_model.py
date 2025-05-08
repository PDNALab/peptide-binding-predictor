import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense
from tensorflow.keras.optimizers import Adam

"""
This script builds and trains a deep learning model for predicting binding properties 
of amino acid sequences. It performs the following tasks:

1. Loads encoded sequence data (X) and binding labels (y) prepared by encode_sequences.py
2. Splits the data into training and validation sets (80/20 split)
3. Defines a convolutional neural network architecture for sequence classification:
   - Embedding layer to convert amino acid indices to vectors
   - Conv1D layer to detect local patterns in sequences
   - GlobalMaxPooling to extract the most important features
   - Dense layers for final classification
4. Trains the model for 5 epochs with binary cross-entropy loss
5. Saves the trained model to disk for later use in predictions

The model is designed to identify binding patterns in fixed-length amino acid sequences,
outputting a probability (0-1) that represents the likelihood of binding.
"""

# Load the previously prepared encoded data
working_dir = os.path.join(os.path.dirname(__file__), '..')
npy_dir = os.path.join(working_dir, 'npy')
X = np.load(os.path.join(npy_dir, 'X.npy'))  # Encoded sequences
y = np.load(os.path.join(npy_dir, 'y.npy'))  # Binary binding labels

# Split the dataset into training and validation sets
# 80% for training, 20% for validation, with a fixed random seed for reproducibility
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define model hyperparameters
vocab_size = 20         # 20 standard amino acids in the vocabulary
embedding_dim = 16      # Dimension of the embedding space for amino acids
input_length = 25       # Fixed length of input sequences (25 residues)

# Define the neural network architecture using Keras Sequential API
model = Sequential([
    # Embedding layer converts integer indices to dense vectors
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=input_length),
    
    # 1D Convolutional layer to detect local patterns in sequences
    # 64 filters with kernel size 3 (looks at 3 amino acids at a time)
    Conv1D(64, kernel_size=3, activation='relu'),
    
    # Global max pooling to reduce dimensionality while keeping the most important features
    GlobalMaxPooling1D(),
    
    # Dense hidden layer with 32 neurons
    Dense(32, activation='relu'),
    
    # Output layer with sigmoid activation for binary classification
    # Outputs a single value between 0-1 (probability of binding)
    Dense(1, activation='sigmoid')
])

# Compile the model with binary cross-entropy loss (standard for binary classification)
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train,
                    validation_data=(X_val, y_val),  # Validation data to monitor during training
                    epochs=5,                        # Number of complete passes through the dataset
                    batch_size=64)                   # Number of samples per gradient update

# Create directory for model storage if it doesn't exist
model_path = os.path.join(working_dir, 'models', 'cnn_binding_model.h5')
os.makedirs(os.path.dirname(model_path), exist_ok=True)

# Save the trained model to disk in Keras's H5 format
model.save(model_path)

print(f"\nModel saved to {model_path}")