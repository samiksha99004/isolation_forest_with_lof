"""
Step 6d: Local Outlier Factor (LOF) as an alternative to Isolation Forest
LOF flags points with locally low density relative to their neighbors,
which can separate a dense-but-different attack cluster from the dense
normal cluster better than Isolation Forest in some cases.

Note: LOF's novelty=True mode (fit on BENIGN only, predict on new data)
is used here to match the same train/test setup as Isolation Forest.
LOF is more computationally expensive than Isolation Forest, so we
subsample the training set to keep runtime reasonable.
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

FOLDER = r"F:\all_10_csv"
INPUT_PATH = f"{FOLDER}\\cleaned_final_grouped.parquet"

print("RUNNING SCRIPT VERSION: v1-lof")

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

# LOF is O(n^2)-ish for neighbor search; full 8.6M-row training set is not
# feasible. Subsample BENIGN training data and also subsample the test set
# to keep this runnable.
TRAIN_SAMPLE_SIZE = 200_000
TEST_SAMPLE_SIZE = 100_000

print("Loading dataset...")
df = pd.read_parquet(INPUT_PATH)

benign_df = df[df['Label_binary'] == 'BENIGN']
attack_df = df[df['Label_binary'] == 'ATTACK']

benign_train_full, benign_test_full = train_test_split(
    benign_df, test_size=0.2, random_state=42
)

benign_train = benign_train_full.sample(n=min(TRAIN_SAMPLE_SIZE, len(benign_train_full)), random_state=42)

test_full = pd.concat([benign_test_full, attack_df], ignore_index=True)
test_full = test_full.sample(frac=1, random_state=42).reset_index(drop=True)
test_df = test_full.sample(n=min(TEST_SAMPLE_SIZE, len(test_full)), random_state=42).reset_index(drop=True)

print(f"Train sample (BENIGN only): {len(benign_train)}")
print(f"Test sample (BENIGN + ATTACK): {len(test_df)}")
print(test_df['Label_binary'].value_counts())


def transform(frame):
    X = frame[FEATURE_COLS].astype('float64').copy()
    for col in SKEWED_COLS:
        X[col] = np.log1p(X[col].clip(lower=0))
    return X.astype('float32').values


X_train = transform(benign_train)
X_test = transform(test_df)

contamination_rate = len(attack_df) / (len(benign_df) + len(attack_df))
print(f"\nContamination rate used: {contamination_rate:.4f}")

print("Fitting LOF (novelty mode)...")
lof = LocalOutlierFactor(
    n_neighbors=35,
    contamination=contamination_rate,
    novelty=True,
    n_jobs=-1
)
lof.fit(X_train)

print("Predicting on test sample...")
raw_preds = lof.predict(X_test)  # 1 = normal (inlier), -1 = anomaly (outlier)
test_df['Predicted_lof'] = np.where(raw_preds == 1, 'BENIGN', 'ATTACK')

print("\n--- Classification Report (LOF) ---")
print(classification_report(test_df['Label_binary'], test_df['Predicted_lof']))

print("--- Confusion Matrix ---")
print(confusion_matrix(test_df['Label_binary'], test_df['Predicted_lof'], labels=['BENIGN', 'ATTACK']))
