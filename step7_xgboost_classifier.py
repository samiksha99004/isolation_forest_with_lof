"""
Step 7: Train XGBoost Attack Classifier (Stage 2)
Trains a multi-class classifier on ATTACK rows only, to name the specific
attack type (DDoS, DoS, Bot, Infiltration, Brute Force, Web Attack) for
whatever Stage 1 (Isolation Forest) flags as an anomaly.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
import joblib

FOLDER = r"F:\all_10_csv"
INPUT_PATH = f"{FOLDER}\\cleaned_final_grouped.parquet"
MODEL_PATH = f"{FOLDER}\\xgboost_attack_classifier.joblib"
LABEL_ENCODER_PATH = f"{FOLDER}\\label_encoder.joblib"
SCALER_PATH = f"{FOLDER}\\scaler_stage2.joblib"

print("RUNNING SCRIPT VERSION: v1-xgboost-classifier")

FEATURE_COLS = [
    'Bwd Pkt Len Mean', 'Init Fwd Win Byts', 'Dst Port', 'Bwd Pkt Len Max',
    'Fwd Pkt Len Mean', 'Fwd Pkt Len Max', 'Pkt Len Var', 'Pkt Len Mean',
    'Flow IAT Max', 'Init Bwd Win Byts', 'Flow Duration', 'Flow IAT Mean',
    'Tot Bwd Pkts', 'Flow Pkts/s', 'Flow IAT Std', 'Bwd IAT Min', 'Bwd Pkts/s',
    'Tot Fwd Pkts', 'Bwd IAT Tot', 'Bwd IAT Max', 'Bwd IAT Std', 'Bwd IAT Mean',
    'Flow Byts/s', 'Fwd Seg Size Min', 'Idle Min', 'Protocol', 'Bwd Pkt Len Min',
    'Fwd Pkt Len Min', 'Pkt Len Min', 'PSH Flag Cnt'
]

print("Loading dataset...")
df = pd.read_parquet(INPUT_PATH)

# Only attack rows for the classifier
attack_df = df[df['Label_group'] != 'BENIGN'].copy()
print(f"Attack rows: {len(attack_df)}")
print(attack_df['Label_group'].value_counts())

X = attack_df[FEATURE_COLS]
y = attack_df['Label_group']

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nTraining XGBoost classifier...")
clf = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    objective='multi:softmax',
    num_class=len(le.classes_),
    eval_metric='mlogloss',
    random_state=42,
    n_jobs=-1
)
clf.fit(X_train_scaled, y_train)

print("\nEvaluating on test set...")
y_pred = clf.predict(X_test_scaled)

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred, target_names=le.classes_))

print("--- Confusion Matrix ---")
print(confusion_matrix(y_test, y_pred))

joblib.dump(clf, MODEL_PATH)
joblib.dump(le, LABEL_ENCODER_PATH)
joblib.dump(scaler, SCALER_PATH)

print(f"\nModel saved to: {MODEL_PATH}")
print(f"Label encoder saved to: {LABEL_ENCODER_PATH}")
print(f"Scaler saved to: {SCALER_PATH}")
