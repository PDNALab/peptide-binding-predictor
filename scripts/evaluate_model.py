import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split

"""
This script evaluates the performance of a trained deep learning model for predicting binding 
properties of amino acid sequences. It performs the following tasks:

1. Loads the encoded sequence data (X) and binding labels (y) 
2. Loads the trained CNN model from disk
3. Splits data into training/validation sets using the same parameters as training
4. Generates binding predictions on the validation set
5. Evaluates model performance using multiple metrics:
   - ROC curve and AUC score
   - Confusion matrix
   - Detailed classification report (precision, recall, F1-score)
6. Saves performance visualizations (ROC curve, confusion matrix) to the results directory

The evaluation provides insights into how well the model can distinguish between binding 
and non-binding amino acid sequences, which is essential for assessing model quality
before deploying it for real-world predictions.
"""

# Load the encoded data
working_dir = os.path.join(os.path.dirname(__file__), '..')
X = np.load(os.path.join(working_dir, 'npy', 'X.npy'))  # Encoded sequences
y = np.load(os.path.join(working_dir, 'npy', 'y.npy'))  # Binary binding labels

# Load the trained model from disk
model_path = os.path.join(working_dir, 'models', 'cnn_binding_model.h5')
model = load_model(model_path)

# Split the data using the same parameters as in train_model.py
# This ensures we're evaluating on the same validation set
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Generate predictions on the validation set
# These are probabilities between 0-1
y_pred_probs = model.predict(X_val).flatten()

# Convert probabilities to binary predictions using 0.5 as threshold
y_pred_labels = (y_pred_probs >= 0.5).astype(int)

# Calculate ROC curve and Area Under Curve (AUC)
fpr, tpr, _ = roc_curve(y_val, y_pred_probs)
roc_auc = auc(fpr, tpr)

# Plot and save the ROC curve
plt.figure()
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')  # Diagonal reference line
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(working_dir, "results", "roc_curve.png"))
plt.close()

# Create and save the confusion matrix visualization
cm = confusion_matrix(y_val, y_pred_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Bind", "Bind"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(working_dir, "results", "confusion_matrix.png"))
plt.close()

print("Saved ROC curve and confusion matrix to 'working_data/results/'")

# Import tools for detailed classification metrics
from sklearn.metrics import classification_report

# Generate a detailed classification report with precision, recall, and F1-score
report = classification_report(y_val, y_pred_labels, target_names=["No Bind", "Bind"])
print("\nClassification Report:\n")
print(report)