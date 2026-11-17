import numpy as np

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Returns elementwise sigmoid values.
    """
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float = 0.1, steps: int = 1000) -> tuple[np.ndarray, float]:
    """
    Returns the trained weights and bias as (w, b).
    """
    # Write code here
     # 1. Make sure inputs are float NumPy arrays (they might arrive as lists)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    # 2. Get the sizes: n_samples = 4 rows, n_features = 1 column
    n_samples, n_features = X.shape

    # 3. Start with no knowledge: all weights 0, bias 0
    w = np.zeros(n_features)   # shape (1,) -> [0.]
    b = 0.0

    # 4. Repeat the learning step many times
    for _ in range(steps):
        # Forward pass: make predictions
        z = X @ w + b          # shape (4,): one logit per sample
        p = _sigmoid(z)        # shape (4,): probabilities between 0 and 1

        # How wrong is each prediction?
        error = p - y          # shape (4,)

        # Gradients: which direction increases the loss
        dw = (X.T @ error) / n_samples   # shape (1,), same as w
        db = np.sum(error) / n_samples   # a single number

        # Update: step in the opposite direction (downhill)
        w -= lr * dw
        b -= lr * db

    # 5. Return the learned parameters
    return w, float(b)