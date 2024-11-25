# VLC Channel Matrix Estimation Using Autoencoder-based Deep Learning

This project uses an **Autoencoder-based Deep Learning model** for efficient estimation of wireless communication channel matrices. The approach enhances channel estimation, enabling improved spectral efficiency and signal quality for wireless systems like 5G and IoT networks.

---

## **Overview**
In wireless communication, accurate estimation of the channel matrix \( H \) is crucial for reliable data transmission. This project leverages an autoencoder to process the real and imaginary components of the channel matrix, compressing them into a latent vector and reconstructing a refined matrix for optimal performance.

---

## **Key Features**
- **Autoencoder Architecture**:
  - **Encoder**: Compresses real and imaginary parts into a low-dimensional representation.
  - **Decoder**: Reconstructs the refined channel matrix from the encoded vector.
- **Output**: A refined channel matrix optimized for wireless communication systems.
- **Pre-trained Model**: The model is saved in `.keras` format for reuse.

---

## **Folder Structure**

```
.
├── .idea/                   # PyCharm project settings
├── __pycache__/             # Cached Python files
├── best_model.keras         # Best-performing model checkpoint
├── evaluate_model.py        # Script to evaluate the model on test data
├── final_model.keras        # Final trained model
├── generate_data.py         # Script for generating synthetic data
├── model.py                 # Contains the autoencoder model definition
├── test_user_input.py       # Script to test the model with user-defined inputs
├── train_model.py           # Script to train the autoencoder model
├── X_train.npy              # Training data (real and imaginary parts)
├── X_val.npy                # Validation data
├── X_test.npy               # Test data
├── y_train.npy              # Training labels (refined matrices)
├── y_val.npy                # Validation labels
├── y_test.npy               # Test labels
```

---

## **Project Workflow**

1. **Data Generation**:
   - Synthetic channel matrices are created using `generate_data.py`.
   - Real and imaginary parts of the channel coefficients are saved as `.npy` files.

2. **Training**:
   - Train the model using `train_model.py` with training and validation data.
   - The best model is saved as `best_model.keras`, and the final trained model is saved as `final_model.keras`.

3. **Evaluation**:
   - Use `evaluate_model.py` to evaluate the trained model on test data.

4. **Testing with User Input**:
   - Run `test_user_input.py` to test the model with custom user-defined inputs for real and imaginary parts of the channel matrix.
   - The script provides the refined channel matrix as output.

---

## **How to Run**

1. **Install Dependencies**:
   ```bash
   pip install tensorflow numpy
   ```

2. **Train the Model**:
   ```bash
   python train_model.py
   ```

3. **Evaluate the Model**:
   ```bash
   python evaluate_model.py
   ```

4. **Test with User Input**:
   ```bash
   python test_user_input.py
   ```

---

## **Applications**
- **5G Networks**: Enhances channel estimation in massive MIMO systems.
- **IoT**: Optimizes communication in complex and dynamic environments.
- **Wireless Systems**: Reduces interference and improves spectral efficiency.

---

## **Contributing**
Feel free to fork the repository and submit pull requests to contribute.

---
