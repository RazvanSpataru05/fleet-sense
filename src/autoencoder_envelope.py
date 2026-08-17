"""
Autoencoder anomaly detector on the envelope representation.

sklearn's MLPRegressor trained to reconstruct its own (scaled) input is a lightweight
autoencoder -- appropriate here since the envelope representation is only 13 features
and we have ~1440 training windows, well within classical-ML territory rather than
deep-learning territory. Anomaly score = per-window mean squared reconstruction error.
"""
from pathlib import Path

import joblib
import numpy as np
from sklearn.neural_network import MLPRegressor

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts" / "envelope"

HIDDEN_LAYERS = (8, 4, 8)  # encoder 13->8->4, decoder 4->8->13


def reconstruction_error(model, X) -> np.ndarray:
    X_hat = model.predict(X)
    return np.mean((X - X_hat) ** 2, axis=1)


def main():
    X_train = np.load(ARTIFACTS_DIR / "X_train_envelope.npy")
    X_val = np.load(ARTIFACTS_DIR / "X_val_envelope.npy")

    model = MLPRegressor(
        hidden_layer_sizes=HIDDEN_LAYERS,
        activation="relu",
        solver="adam",
        max_iter=2000,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=42,
    )
    model.fit(X_train, X_train)

    joblib.dump(model, ARTIFACTS_DIR / "autoencoder_envelope.pkl")

    train_errors = reconstruction_error(model, X_train)
    val_errors = reconstruction_error(model, X_val)

    print(f"Training stopped after {model.n_iter_} iterations")
    print("Train reconstruction error:")
    print(f"  mean: {train_errors.mean():.5f}  std: {train_errors.std():.5f}")
    print("Validation reconstruction error (healthy, unseen):")
    print(f"  mean: {val_errors.mean():.5f}  std: {val_errors.std():.5f}")
    print(f"  min:  {val_errors.min():.5f}  max: {val_errors.max():.5f}")


if __name__ == "__main__":
    main()
