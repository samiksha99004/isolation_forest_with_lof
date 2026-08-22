"""
Step 3: Fix Data Types + Deduplicate
Reads combined_no_na.parquet, downcasts numeric columns to save memory,
removes exact duplicate rows, saves result.
"""

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

FOLDER = r"F:\all_10_csv"
INPUT_PATH = f"{FOLDER}\\combined_no_na.parquet"
OUTPUT_PATH = f"{FOLDER}\\combined_deduped.parquet"
BATCH_SIZE = 200_000

print("RUNNING SCRIPT VERSION: v1-dedup")

pf = pq.ParquetFile(INPUT_PATH)
writer = None
total_rows_in = 0
total_rows_out = 0

# Track duplicates across ALL batches using a hash set of row tuples.
# (Dataset is ~16M rows x 79 cols -- this uses more RAM than dedup within
# a single batch, but is needed to catch duplicates that span batches.)
seen_hashes = set()

for batch in pf.iter_batches(batch_size=BATCH_SIZE):
    df = batch.to_pandas()
    total_rows_in += len(df)

    # Downcast numeric dtypes
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')

    # Hash each row to check for duplicates across batches
    row_hashes = pd.util.hash_pandas_object(df, index=False)
    mask_new = ~row_hashes.isin(seen_hashes)

    # Also drop duplicates within this batch itself
    df = df[mask_new]
    row_hashes = row_hashes[mask_new]
    dup_in_batch = df.duplicated()
    df = df[~dup_in_batch]
    row_hashes = row_hashes[~dup_in_batch]

    seen_hashes.update(row_hashes.tolist())

    total_rows_out += len(df)

    if df.empty:
        continue

    table = pa.Table.from_pandas(df, preserve_index=False)
    if writer is None:
        writer = pq.ParquetWriter(OUTPUT_PATH, table.schema)
    else:
        table = table.cast(writer.schema)
    writer.write_table(table)

if writer is not None:
    writer.close()

print(f"Total rows read: {total_rows_in}")
print(f"Total rows after deduplication: {total_rows_out}")
print(f"Duplicates removed: {total_rows_in - total_rows_out}")
print(f"Saved to: {OUTPUT_PATH}")
