from pathlib import Path

import numpy as np
import joblib
from sklearn.ensemble import IsolationForest

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts" / "windowed"

X_train = np.load(ARTIFACTS_DIR / 'X_train_windowed.npy')
X_val = np.load(ARTIFACTS_DIR / 'X_val_windowed.npy')

model = IsolationForest(random_state=42)
model.fit(X_train)

joblib.dump(model, ARTIFACTS_DIR / 'isolation_forest_windowed.pkl')

val_scores = model.score_samples(X_val)

print("Validation scores (healthy, unseen, windowed):")
print(f"  mean:  {val_scores.mean():.4f}")
print(f"  std:   {val_scores.std():.4f}")
print(f"  min:   {val_scores.min():.4f}")
print(f"  max:   {val_scores.max():.4f}")
