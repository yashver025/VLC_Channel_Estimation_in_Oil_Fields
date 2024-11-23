import numpy as np
from tensorflow.keras.models import load_model

# Define the preprocessing function
def preprocess_user_input(real_part, imag_part):
    """
    Prepares user input for prediction.
    Args:
        real_part (list): Real parts of the channel matrix.
        imag_part (list): Imaginary parts of the channel matrix.
    Returns:
        numpy.ndarray: Preprocessed input array.
    """
    real_part = np.array(real_part)
    imag_part = np.array(imag_part)
    user_input = np.stack((real_part, imag_part), axis=-1)  # Combine along the last axis
    user_input = user_input[np.newaxis, ...]  # Add batch dimension
    return user_input

# Load the trained model
model = load_model("best_model.keras")

# Ask for input from the user
print("Enter the real and imaginary parts of the channel matrix.")
rows = int(input("Enter the number of rows in the matrix: "))
cols = int(input("Enter the number of columns in the matrix: "))

print("\nEnter the values row by row (comma-separated):")
real_part = []
for i in range(rows):
    row = input(f"Real part - Row {i + 1}: ").split(",")
    real_part.append([float(val) for val in row])

imag_part = []
for i in range(rows):
    row = input(f"Imaginary part - Row {i + 1}: ").split(",")
    imag_part.append([float(val) for val in row])

# Preprocess the inputs
user_input = preprocess_user_input(real_part, imag_part)

# Make predictions
prediction = model.predict(user_input)

# Display the result
print("\nPredicted output:")
print(prediction)
