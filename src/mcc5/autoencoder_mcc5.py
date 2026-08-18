"""
Autoencoder anomaly detector on the MCC5-THU envelope representation.

Same approach as Paderborn (sklearn MLPRegressor reconstructing its own scaled input),
but with stronger regularization -- 287 training windows against 87 features is a much
thinner ratio than Paderborn's envelope model had, so overfitting risk is real.

Trains one model per split (speed_circulation, torque_circulation) -- healthy files
score systematically differently by split alone, so a single shared model compared
against a mixed-split baseline would confound "which archive" with "healthy vs faulty".
"""
from pathlib import Path

import joblib
import numpy as np
from sklearn.neural_network import MLPRegressor

from load_mcc5 import SPLIT_DIRS
from envelope_dataset_mcc5 import artifacts_dir

HIDDEN_LAYERS = (32, 8, 32)  # encoder 87->32->8, decoder 8->32->87
ALPHA = 0.01  # stronger L2 than sklearn's default (0.0001), given the thin sample:feature ratio
# Tried a narrower (16,4,16)/alpha=0.03 config too: nearly identical percentile results,
# just larger absolute error values. That ruled out model capacity/overfitting as the
# bottleneck -- the limiting factor is the feature signal itself, not the network.


def reconstruction_error(model, X) -> np.ndarray:
    X_hat = model.predict(X)
    return np.mean((X - X_hat) ** 2, axis=1)


def train_for_split(split: str):
    out_dir = artifacts_dir(split)
    X_train = np.load(out_dir / "X_train_envelope.npy")
    X_val = np.load(out_dir / "X_val_envelope.npy")

    model = MLPRegressor(
        hidden_layer_sizes=HIDDEN_LAYERS,
        activation="relu",
        solver="adam",
        alpha=ALPHA,
        max_iter=2000,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=42,
    )
    model.fit(X_train, X_train)

    joblib.dump(model, out_dir / "autoencoder_mcc5.pkl")

    train_errors = reconstruction_error(model, X_train)
    val_errors = reconstruction_error(model, X_val)

    print(f"=== {split} ===")
    print(f"Training stopped after {model.n_iter_} iterations")
    print(f"Train error:      mean={train_errors.mean():.5f}  std={train_errors.std():.5f}")
    print(f"Val error:        mean={val_errors.mean():.5f}  std={val_errors.std():.5f}")
    print(f"                  min={val_errors.min():.5f}  max={val_errors.max():.5f}\n")


def main():
    for split in SPLIT_DIRS:
        train_for_split(split)


if __name__ == "__main__":
    main()
