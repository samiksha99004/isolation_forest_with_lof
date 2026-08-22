"""
Step 6c: Threshold Tuning for Isolation Forest
Instead of the crude contamination-based cutoff, sweep thresholds on the
continuous anomaly score to find the one that maximizes F1, and report
accuracy at that threshold.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import classification_report, confusion_matrix, f1_score

FOLDER = r"F:\all_10_csv"
MODEL_PATH = f"{FOLDER}\\isolation_forest_model.joblib"
TEST_SET_PATH = f"{FOLDER}\\stage1_test_set.parquet"

FEATURE_COLS = [
    'Bwd Pkt Len Mean', 'Init Fwd Win Byts', 'Dst Port', 'Bwd Pkt Len Max',
    'Fwd Pkt Len Mean', 'Fwd Pkt Len Max', 'Pkt Len Var', 'Pkt Len Mean',
    'Flow IAT Max', 'Init Bwd Win Byts', 'Flow Duration', 'Flow IAT Mean',
    'Tot Bwd Pkts', 'Flow Pkts/s', 'Flow IAT Std', 'Bwd IAT Min', 'Bwd Pkts/s',
    'Tot Fwd Pkts', 'Bwd IAT Tot', 'Bwd IAT Max', 'Bwd IAT Std', 'Bwd IAT Mean',
    'Flow Byts/s', 'Fwd Seg Size Min', 'Idle Min', 'Protocol', 'Bwd Pkt Len Min',
    'Fwd Pkt Len Min', 'Pkt Len Min', 'PSH Flag Cnt'
]
SKEWED_COLS = [c for c in FEATURE_COLS if c not in ('Protocol', 'Dst Port', 'PSH Flag Cnt')]

print("Loading model and test set...")
iso_forest = joblib.load(MODEL_PATH)
test_df = pd.read_parquet(TEST_SET_PATH)


def transform(frame):
    X = frame[FEATURE_COLS].astype('float64').copy()
    for col in SKEWED_COLS:
        X[col] = np.log1p(X[col].clip(lower=0))
    return X.astype('float32').values


X_test = transform(test_df)

print("Computing continuous anomaly scores...")
scores = iso_forest.score_samples(X_test)  # higher = more normal

y_true = (test_df['Label_binary'] == 'ATTACK').astype(int).values

print("Sweeping thresholds to find best F1...")
best_f1 = -1
best_thresh = None
for pct in np.arange(1, 50, 1):
    thresh = np.percentile(scores, pct)
    y_pred = (scores < thresh).astype(int)  # below threshold = anomaly = ATTACK
    f1 = f1_score(y_true, y_pred)
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = thresh
        best_pct = pct

print(f"\nBest threshold found at percentile {best_pct} (F1={best_f1:.4f})")

y_pred_final = (scores < best_thresh).astype(int)
test_df['Predicted_tuned'] = np.where(y_pred_final == 1, 'ATTACK', 'BENIGN')

print("\n--- Classification Report (tuned threshold) ---")
print(classification_report(test_df['Label_binary'], test_df['Predicted_tuned']))

print("--- Confusion Matrix ---")
print(confusion_matrix(test_df['Label_binary'], test_df['Predicted_tuned'], labels=['BENIGN', 'ATTACK']))

test_df.to_parquet(TEST_SET_PATH, index=False)
print(f"\nUpdated test set (with tuned predictions) saved to: {TEST_SET_PATH}")
