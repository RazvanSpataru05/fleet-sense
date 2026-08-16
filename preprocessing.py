import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def drop_columns(df):
    new_df = df.drop(columns=['bearing_code', 'condition_code', 'repetition'])
    return new_df

def main():
    df_features = pd.read_csv('training_features.csv')
    print(df_features.size)
    df_features = drop_columns(df_features)
    print(df_features.size)

    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(df_features)
    features_normalized_df = pd.DataFrame(features_normalized, columns=df_features.columns)
    print(features_normalized_df.mean())
    print(features_normalized_df.std())

    joblib.dump(scaler, 'scaler.pkl')
    X_train, X_val = train_test_split(features_normalized, test_size=0.2, random_state=42)

    print(X_train.shape)
    print(X_val.shape)

    np.save('X_train.npy', X_train)
    np.save('X_val.npy', X_val)

if __name__ == "__main__":
    main()