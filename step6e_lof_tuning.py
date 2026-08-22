"""
Step 6e (v2, Windows-safe): LOF Hyperparameter Tuning
Fixes a Windows multiprocessing hang: n_jobs=-1 spawns child processes,
which on Windows re-import this script unless guarded by
`if __name__ == "__main__":`. Also set n_jobs=1 to avoid the risk
entirely, and dropped the 1M-row option (too slow to be practical for LOF).
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

FOLDER = r"F:\all_10_csv"
INPUT_PATH = f"{FOLDER}\\cleaned_final_grouped.parquet"

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

TEST_SAMPLE_SIZE = 100_000
TRAIN_SIZES = [200_000, 500_000]
N_NEIGHBORS_OPTIONS = [20, 35, 50, 75]


def transform(frame):
    X = frame[FEATURE_COLS].astype('float64').copy()
    for col in SKEWED_COLS:
        X[col] = np.log1p(X[col].clip(lower=0))
    return X.astype('float32').values


def main():
    print("RUNNING SCRIPT VERSION: v2-windows-safe")
    print("Loading dataset...")
    df = pd.read_parquet(INPUT_PATH)

    benign_df = df[df['Label_binary'] == 'BENIGN']
    attack_df = df[df['Label_binary'] == 'ATTACK']
    contamination_rate = len(attack_df) / (len(benign_df) + len(attack_df))

    benign_train_full, benign_test_full = train_test_split(
        benign_df, test_size=0.2, random_state=42
    )

    test_full = pd.concat([benign_test_full, attack_df], ignore_index=True)
    test_full = test_full.sample(frac=1, random_state=42).reset_index(drop=True)
    test_df = test_full.sample(n=min(TEST_SAMPLE_SIZE, len(test_full)), random_state=42).reset_index(drop=True)

    X_test = transform(test_df)
    y_true = (test_df['Label_binary'] == 'ATTACK').astype(int).values

    results = []

    for train_size in TRAIN_SIZES:
        benign_train = benign_train_full.sample(
            n=min(train_size, len(benign_train_full)), random_state=42
        )
        X_train = transform(benign_train)

        for n_neighbors in N_NEIGHBORS_OPTIONS:
            print(f"\nTraining LOF: train_size={train_size}, n_neighbors={n_neighbors}...")
            lof = LocalOutlierFactor(
                n_neighbors=n_neighbors,
                contamination=contamination_rate,
                novelty=True,
                n_jobs=1  # single-threaded: avoids Windows multiprocessing hang
            )
            lof.fit(X_train)
            raw_preds = lof.predict(X_test)
            y_pred = (raw_preds == -1).astype(int)

            f1 = f1_score(y_true, y_pred)
            acc = (y_pred == y_true).mean()
            print(f"  Accuracy: {acc:.4f}, F1 (ATTACK): {f1:.4f}")

            results.append({
                'train_size': train_size,
                'n_neighbors': n_neighbors,
                'accuracy': acc,
                'f1_attack': f1
            })

    results_df = pd.DataFrame(results).sort_values('f1_attack', ascending=False)
    print("\n\n=== ALL RESULTS (sorted by F1) ===")
    print(results_df.to_string(index=False))

    best = results_df.iloc[0]
    print(f"\nBest config: train_size={int(best['train_size'])}, n_neighbors={int(best['n_neighbors'])}")
    print(f"Accuracy={best['accuracy']:.4f}, F1={best['f1_attack']:.4f}")


if __name__ == "__main__":
    main()