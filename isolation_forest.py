import numpy as np
import joblib
from sklearn.ensemble import IsolationForest

X_train = np.load('X_train.npy')
X_val = np.load('X_val.npy')

model = IsolationForest(random_state=42)
model.fit(X_train)

joblib.dump(model, 'isolation_forest.pkl')

val_scores = model.score_samples(X_val)

print("Validation scores (healthy, unseen):")
print(f"  mean:  {val_scores.mean():.4f}")
print(f"  std:   {val_scores.std():.4f}")
print(f"  min:   {val_scores.min():.4f}")
print(f"  max:   {val_scores.max():.4f}")