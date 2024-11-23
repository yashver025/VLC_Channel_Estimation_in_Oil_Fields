from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

# Load test data
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

# Load the trained model
model = load_model("best_model.keras")

# Evaluate the model
loss, mae = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}, Test MAE: {mae:.4f}")

# Make predictions on the test data
predictions = model.predict(X_test)

# Visualize a sample prediction
index = 0  # Change this index to visualize different samples
plt.figure(figsize=(12, 4))

# Input real part
plt.subplot(1, 3, 1)
plt.title("Input Real Part")
plt.imshow(X_test[index, :, :, 0], cmap='viridis')

# Ground truth real part
plt.subplot(1, 3, 2)
plt.title("Ground Truth Real Part")
plt.imshow(y_test[index, :, :, 0], cmap='viridis')

# Predicted real part
plt.subplot(1, 3, 3)
plt.title("Predicted Real Part")
plt.imshow(predictions[index, :, :, 0], cmap='viridis')

plt.show()
