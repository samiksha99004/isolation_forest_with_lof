"""
Step 8: Test Stage 1 Model on the Full Cleaned Dataset
Runs the trained XGBoost Stage 1 model on cleaned_final_grouped.parquet
(all attack types mixed together) and reports:
  - Correct detections (accurate)
  - Misses (actual ATTACK predicted as BENIGN -- false negatives)
  - False alarms (actual BENIGN predicted as ATTACK -- false positives)
  - Overall accuracy (out of 100)

Note: this dataset includes rows the model was trained on (it's the full
cleaned dataset, not a held-out test set), so this number reflects
performance on data the model has seen -- not a true generalization test.
"""

import pandas as pd
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
    print("RUNNING SCRIPT VERSION: v1-full-dataset-test")

    print("Loading model...")
    clf = joblib.load(MODEL_PATH)
    le = joblib.load(LABEL_ENCODER_PATH)

    print("Loading full cleaned dataset...")
    df = pd.read_parquet(INPUT_PATH)
    print(f"Total rows: {len(df)}")

    X = df[FEATURE_COLS]
    y_true_encoded = le.transform(df['Label_binary'])

    print("Predicting on all rows...")
    y_pred_encoded = clf.predict(X)

    attack_idx = list(le.classes_).index('ATTACK')
    benign_idx = list(le.classes_).index('BENIGN')

    is_actual_attack = (y_true_encoded == attack_idx)
    is_actual_benign = (y_true_encoded == benign_idx)
    is_pred_attack = (y_pred_encoded == attack_idx)
    is_pred_benign = (y_pred_encoded == benign_idx)

    true_positives = (is_actual_attack & is_pred_attack).sum()      # correctly caught attacks
    false_negatives = (is_actual_attack & is_pred_benign).sum()     # MISSES: attack called normal
    true_negatives = (is_actual_benign & is_pred_benign).sum()      # correctly identified normal
    false_positives = (is_actual_benign & is_pred_attack).sum()     # FALSE ALARMS: normal called attack

    total = len(df)
    correct = true_positives + true_negatives
    accuracy_pct = 100 * correct / total

    print("\n=== RESULTS ===")
    print(f"Total rows tested: {total}")
    print(f"Accurate (correctly identified): {correct}")
    print(f"  - Correctly identified ATTACK (true positives): {true_positives}")
    print(f"  - Correctly identified NORMAL (true negatives): {true_negatives}")
    print(f"Misses (real attack missed, called NORMAL): {false_negatives}")
    print(f"False alarms (real normal flagged as ATTACK): {false_positives}")
    print(f"\nAccuracy: {accuracy_pct:.2f} / 100")


if __name__ == "__main__":
    main()
