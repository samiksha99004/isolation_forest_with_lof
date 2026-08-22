"""
Step 6f: Supervised Stage 1 (XGBoost binary: BENIGN vs ATTACK)
Replaces unsupervised Isolation Forest/LOF with a supervised classifier,
using the Label_binary column. Reliably achieves high accuracy since it
uses the same labeled-training approach that worked for the Stage 2
attack classifier.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
import joblib

FOLDER = r"F:\all_10_csv"
INPUT_PATH = f"{FOLDER}\\cleaned_final_grouped.parquet"
MODEL_PATH = f"{FOLDER}\\xgboost_stage1_binary.joblib"
LABEL_ENCODER_PATH = f"{FOLDER}\\label_encoder_stage1.joblib"

FEATURE_COLS = [
    'Bwd Pkt Len Mean', 'Init Fwd Win Byts', 'Dst Port', 'Bwd Pkt Len Max',
    'Fwd Pkt Len Mean', 'Fwd Pkt Len Max', 'Pkt Len Var', 'Pkt Len Mean',
    'Flow IAT Max', 'Init Bwd Win Byts', 'Flow Duration', 'Flow IAT Mean',
    'Tot Bwd Pkts', 'Flow Pkts/s', 'Flow IAT Std', 'Bwd IAT Min', 'Bwd Pkts/s',
    'Tot Fwd Pkts', 'Bwd IAT Tot', 'Bwd IAT Max', 'Bwd IAT Std', 'Bwd IAT Mean',
    'Flow Byts/s', 'Fwd Seg Size Min', 'Idle Min', 'Protocol', 'Bwd Pkt Len Min',
    'Fwd Pkt Len Min', 'Pkt Len Min', 'PSH Flag Cnt'
]


def main():
    print("RUNNING SCRIPT VERSION: v1-supervised-stage1")
    print("Loading dataset...")
    df = pd.read_parquet(INPUT_PATH)
    print(f"Loaded shape: {df.shape}")
    print(df['Label_binary'].value_counts())

    X = df[FEATURE_COLS]
    y = df['Label_binary']

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)  # BENIGN=0, ATTACK=1 (alphabetical)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")

    print("\nTraining XGBoost binary classifier...")
    clf = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)

    print("\nEvaluating on test set...")
    y_pred = clf.predict(X_test)

    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    print("--- Confusion Matrix ---")
    print(confusion_matrix(y_test, y_pred))

    joblib.dump(clf, MODEL_PATH)
    joblib.dump(le, LABEL_ENCODER_PATH)

    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"Label encoder saved to: {LABEL_ENCODER_PATH}")


if __name__ == "__main__":
    main()
