"""
Step 5: Group Fine-Grained Labels into Attack Classifier Categories
Fixes typos/casing, and adds a 'Label_group' column matching your
Attack Classifier's target categories: BENIGN, DDoS, DoS, Bot,
Infiltration, Brute Force, Web Attack.
Original detailed 'Label' and binary 'Label_binary' are kept as-is.
"""

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

FOLDER = r"F:\all_10_csv"
INPUT_PATH = f"{FOLDER}\\cleaned_final.parquet"
OUTPUT_PATH = f"{FOLDER}\\cleaned_final_grouped.parquet"
BATCH_SIZE = 200_000

print("RUNNING SCRIPT VERSION: v1-label-grouping")

# Fix typos / casing first
LABEL_FIX = {
    'Infilteration': 'Infiltration',
    'DDOS attack-HOIC': 'DDoS attack-HOIC',
    'DDOS attack-LOIC-UDP': 'DDoS attack-LOIC-UDP',
}

# Map cleaned label -> classifier group
GROUP_MAP = {
    'BENIGN': 'BENIGN',
    'Bot': 'Bot',
    'Infiltration': 'Infiltration',
    'DDoS attacks-LOIC-HTTP': 'DDoS',
    'DDoS attack-HOIC': 'DDoS',
    'DDoS attack-LOIC-UDP': 'DDoS',
    'DoS attacks-Hulk': 'DoS',
    'DoS attacks-GoldenEye': 'DoS',
    'DoS attacks-Slowloris': 'DoS',
    'DoS attacks-SlowHTTPTest': 'DoS',
    'SSH-Bruteforce': 'Brute Force',
    'FTP-BruteForce': 'Brute Force',
    'Brute Force -Web': 'Web Attack',
    'Brute Force -XSS': 'Web Attack',
    'SQL Injection': 'Web Attack',
}

pf = pq.ParquetFile(INPUT_PATH)
writer = None
total_rows = 0
group_counts = {}
unmapped = set()

for batch in pf.iter_batches(batch_size=BATCH_SIZE):
    df = batch.to_pandas()

    df['Label'] = df['Label'].replace(LABEL_FIX)
    df['Label_group'] = df['Label'].map(GROUP_MAP)

    missing_mask = df['Label_group'].isna()
    if missing_mask.any():
        unmapped.update(df.loc[missing_mask, 'Label'].unique().tolist())
        df.loc[missing_mask, 'Label_group'] = df.loc[missing_mask, 'Label']

    for g, cnt in df['Label_group'].value_counts().items():
        group_counts[g] = group_counts.get(g, 0) + cnt

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
print("\nLabel_group distribution:")
for g, cnt in sorted(group_counts.items(), key=lambda x: -x[1]):
    print(f"  {g}: {cnt}")

if unmapped:
    print(f"\nWARNING: unmapped labels found (left as-is): {unmapped}")

print(f"\nSaved to: {OUTPUT_PATH}")
