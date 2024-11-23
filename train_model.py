from model import build_model
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np

# Load training and validation data
X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")
X_val = np.load("X_val.npy")
y_val = np.load("y_val.npy")

# Build the model
model = build_model(input_shape=X_train.shape[1:])
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Save the best model during training
checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True)

# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    callbacks=[checkpoint]
)

# Save the final trained model
model.save("final_model.keras")
print("Model training completed and saved.")
