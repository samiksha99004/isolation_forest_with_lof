"""
Step 6b: Diagnose Isolation Forest recall PER attack type.
Uses the test set + predictions already saved by step 6.
"""

import pandas as pd

FOLDER = r"F:\all_10_csv"
TEST_SET_PATH = f"{FOLDER}\\stage1_test_set.parquet"

df = pd.read_parquet(TEST_SET_PATH)

print("Recall per attack type (Label_group):\n")
for group in df['Label_group'].unique():
    if group == 'BENIGN':
        continue
    subset = df[df['Label_group'] == group]
    total = len(subset)
    caught = (subset['Predicted'] == 'ATTACK').sum()
    recall = caught / total if total > 0 else 0
    print(f"  {group}: {caught}/{total} caught ({recall:.2%})")

print("\nBENIGN false positive rate:")
benign_subset = df[df['Label_group'] == 'BENIGN']
fp = (benign_subset['Predicted'] == 'ATTACK').sum()
print(f"  {fp}/{len(benign_subset)} ({fp/len(benign_subset):.2%})")
