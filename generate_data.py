import numpy as np

# Function to generate synthetic data
def generate_synthetic_data(samples=1000, size=(32, 32)):
    X = np.random.randn(samples, size[0], size[1], 2)  # Input: real & imaginary parts
    y = np.random.randn(samples, size[0], size[1], 2)  # Target: real & imaginary parts
    return X, y

# Generate data and split it into train, validation, and test sets
X, y = generate_synthetic_data()
X_train, y_train = X[:700], y[:700]
X_val, y_val = X[700:900], y[700:900]
X_test, y_test = X[900:], y[900:]

# Save the data as .npy files
np.save("X_train.npy", X_train)
np.save("y_train.npy", y_train)
np.save("X_val.npy", X_val)
np.save("y_val.npy", y_val)
np.save("X_test.npy", X_test)
np.save("y_test.npy", y_test)

print("Synthetic data generated and saved successfully.")
