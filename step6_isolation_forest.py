"""
Step 6 (v3): Train Isolation Forest (Stage 1 - Anomaly Detection)
Fixes poor recall from v2 by:
  1. log1p-transforming heavily skewed flow features (durations, byte
     rates, inter-arrival times) so extreme values don't distort
     Isolation Forest's random-split mechanism.
  2. Setting contamination to the dataset's real attack ratio instead
     of 'auto', so the anomaly threshold is calibrated correctly.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

FOLDER = r"F:\all_10_csv"
INPUT_PATH = f"{FOLDER}\\cleaned_final_grouped.parquet"
MODEL_PATH = f"{FOLDER}\\isolation_forest_model.joblib"
TEST_SET_PATH = f"{FOLDER}\\stage1_test_set.parquet"

print("RUNNING SCRIPT VERSION: v3-logtransform-contamination")

FEATURE_COLS = [
    'Bwd Pkt Len Mean', 'Init Fwd Win Byts', 'Dst Port', 'Bwd Pkt Len Max',
    'Fwd Pkt Len Mean', 'Fwd Pkt Len Max', 'Pkt Len Var', 'Pkt Len Mean',
    'Flow IAT Max', 'Init Bwd Win Byts', 'Flow Duration', 'Flow IAT Mean',
    'Tot Bwd Pkts', 'Flow Pkts/s', 'Flow IAT Std', 'Bwd IAT Min', 'Bwd Pkts/s',
    'Tot Fwd Pkts', 'Bwd IAT Tot', 'Bwd IAT Max', 'Bwd IAT Std', 'Bwd IAT Mean',
    'Flow Byts/s', 'Fwd Seg Size Min', 'Idle Min', 'Protocol', 'Bwd Pkt Len Min',
    'Fwd Pkt Len Min', 'Pkt Len Min', 'PSH Flag Cnt'
]

# Columns that are counts/rates/durations spanning huge ranges -- log1p these.
# Leave small-range categorical-ish columns (Protocol, Dst Port, PSH Flag Cnt) alone.
SKEWED_COLS = [c for c in FEATURE_COLS if c not in ('Protocol', 'Dst Port', 'PSH Flag Cnt')]

print("Loading dataset...")
df = pd.read_parquet(INPUT_PATH)
print(f"Loaded shape: {df.shape}")

benign_df = df[df['Label_binary'] == 'BENIGN']
attack_df = df[df['Label_binary'] == 'ATTACK']
print(f"BENIGN rows: {len(benign_df)}, ATTACK rows: {len(attack_df)}")

# Real-world contamination rate (used to calibrate the anomaly threshold)
contamination_rate = len(attack_df) / (len(benign_df) + len(attack_df))
print(f"Contamination rate used: {contamination_rate:.4f}")

benign_train, benign_test = train_test_split(
    benign_df, test_size=0.2, random_state=42
)

test_df = pd.concat([benign_test, attack_df], ignore_index=True)
test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Train (BENIGN only): {len(benign_train)}")
print(f"Test (BENIGN + ATTACK): {len(test_df)}")


def transform(frame):
    X = frame[FEATURE_COLS].astype('float64').copy()
    for col in SKEWED_COLS:
        X[col] = np.log1p(X[col].clip(lower=0))
    return X.astype('float32').values


X_train = transform(benign_train)
X_test = transform(test_df)

print("\nTraining Isolation Forest...")
iso_forest = IsolationForest(
    n_estimators=300,
    contamination=contamination_rate,
    random_state=42,
    n_jobs=-1
)
iso_forest.fit(X_train)

print("Predicting on test set...")
raw_preds = iso_forest.predict(X_test)
test_df['Predicted'] = np.where(raw_preds == 1, 'BENIGN', 'ATTACK')

print("\n--- Classification Report (Normal vs Anomaly) ---")
print(classification_report(test_df['Label_binary'], test_df['Predicted']))

print("--- Confusion Matrix ---")
print(confusion_matrix(test_df['Label_binary'], test_df['Predicted'], labels=['BENIGN', 'ATTACK']))

joblib.dump(iso_forest, MODEL_PATH)
test_df.to_parquet(TEST_SET_PATH, index=False)

print(f"\nModel saved to: {MODEL_PATH}")
print(f"Test set (with predictions) saved to: {TEST_SET_PATH}")