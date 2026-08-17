from pathlib import Path

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts" / "scalar"

IDENTIFIER_COLS = ['bearing_code', 'condition_code', 'repetition']

# Scope training to a single operating condition, matching the KA04/KI04 test files
# (N09_M07_F10). Mixing all 4 conditions made the between-condition spread bigger than
# the bearing-fault signal, diluting the Isolation Forest's anomaly scores. Set to None
# to go back to training on all conditions mixed.
TRAIN_CONDITION = 'N09_M07_F10'

def drop_columns(df):
    new_df = df.drop(columns=IDENTIFIER_COLS)
    return new_df

def main():
    df_features = pd.read_csv(ARTIFACTS_DIR / 'training_features.csv')
    print(df_features.size)

    if TRAIN_CONDITION is not None:
        df_features = df_features[df_features['condition_code'] == TRAIN_CONDITION].reset_index(drop=True)
        print(f"Filtered to condition_code == {TRAIN_CONDITION!r}: {len(df_features)} rows")

    identifiers = df_features[IDENTIFIER_COLS].reset_index(drop=True)
    df_numeric = drop_columns(df_features)
    print(df_numeric.size)

    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(df_numeric)
    features_normalized_df = pd.DataFrame(features_normalized, columns=df_numeric.columns)
    print(features_normalized_df.mean())
    print(features_normalized_df.std())

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, ARTIFACTS_DIR / 'scaler.pkl')

    X_train, X_val, identifiers_train, identifiers_val = train_test_split(
        features_normalized, identifiers, test_size=0.2, random_state=42
    )

    print(X_train.shape)
    print(X_val.shape)

    np.save(ARTIFACTS_DIR / 'X_train.npy', X_train)
    np.save(ARTIFACTS_DIR / 'X_val.npy', X_val)
    np.save(ARTIFACTS_DIR / 'condition_train.npy', identifiers_train['condition_code'].to_numpy())
    np.save(ARTIFACTS_DIR / 'condition_val.npy', identifiers_val['condition_code'].to_numpy())

if __name__ == "__main__":
    main()