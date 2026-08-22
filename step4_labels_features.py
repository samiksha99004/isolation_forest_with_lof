"""
Step 4: Normalize Labels + Create Binary Label + Select Features
- Cleans up inconsistent label spelling/casing
- Creates a new 'Label_binary' column: BENIGN vs ATTACK (for Isolation Forest)
- Keeps original 'Label' column intact with full attack-type names (for XGBoost)
- Selects the 30 model features + both label columns
- Saves final cleaned dataset
"""

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

FOLDER = r"F:\all_10_csv"
INPUT_PATH = f"{FOLDER}\\combined_deduped.parquet"
OUTPUT_PATH = f"{FOLDER}\\cleaned_final.parquet"
BATCH_SIZE = 200_000

print("RUNNING SCRIPT VERSION: v1-labels-features")

SELECTED_FEATURES = [
    'Bwd Pkt Len Mean', 'Init Fwd Win Byts', 'Dst Port', 'Bwd Pkt Len Max',
    'Fwd Pkt Len Mean', 'Fwd Pkt Len Max', 'Pkt Len Var', 'Pkt Len Mean',
    'Flow IAT Max', 'Init Bwd Win Byts', 'Flow Duration', 'Flow IAT Mean',
    'Tot Bwd Pkts', 'Flow Pkts/s', 'Flow IAT Std', 'Bwd IAT Min', 'Bwd Pkts/s',
    'Tot Fwd Pkts', 'Bwd IAT Tot', 'Bwd IAT Max', 'Bwd IAT Std', 'Bwd IAT Mean',
    'Flow Byts/s', 'Fwd Seg Size Min', 'Idle Min', 'Protocol', 'Bwd Pkt Len Min',
    'Fwd Pkt Len Min', 'Pkt Len Min', 'PSH Flag Cnt'
]

# Label cleanup map -- add more mappings here if step output shows others
LABEL_CLEANUP = {
    'Benign': 'BENIGN',
    'BENIGN': 'BENIGN',
}

pf = pq.ParquetFile(INPUT_PATH)
writer = None
total_rows = 0
label_value_counts = {}

for batch in pf.iter_batches(batch_size=BATCH_SIZE):
    df = batch.to_pandas()

    # Normalize label text
    df['Label'] = df['Label'].str.strip()
    df['Label'] = df['Label'].replace(LABEL_CLEANUP)

    # Track label distribution
    for label, cnt in df['Label'].value_counts().items():
        label_value_counts[label] = label_value_counts.get(label, 0) + cnt

    # Binary label for Isolation Forest
    df['Label_binary'] = df['Label'].apply(lambda x: 'BENIGN' if x == 'BENIGN' else 'ATTACK')

    # Check all selected features are present
    missing = [f for f in SELECTED_FEATURES if f not in df.columns]
    if missing:
        raise ValueError(f"Missing expected feature columns: {missing}")

    final_cols = SELECTED_FEATURES + ['Label', 'Label_binary']
    df = df[final_cols]

    total_rows += len(df)

    table = pa.Table.from_pandas(df, preserve_index=False)
    if writer is None:
        writer = pq.ParquetWriter(OUTPUT_PATH, table.schema)
    else:
        table = table.cast(writer.schema)
    writer.write_table(table)

if writer is not None:
    writer.close()

print(f"\nTotal rows written: {total_rows}")
print("\nLabel distribution:")
for label, cnt in sorted(label_value_counts.items(), key=lambda x: -x[1]):
    print(f"  {label}: {cnt}")

print(f"\nSaved final cleaned dataset to: {OUTPUT_PATH}")
