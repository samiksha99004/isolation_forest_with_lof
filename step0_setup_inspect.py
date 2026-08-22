"""
Step 1 (fixed): Load + Align Schema + Drop Junk Rows
Streams cleaned chunks directly to a Parquet file instead of holding
everything in memory, to avoid MemoryError on large files.
"""

import pandas as pd
import glob
import pyarrow as pa
import pyarrow.parquet as pq

FOLDER = r"F:\all_10_csv"
files = glob.glob(f"{FOLDER}\\*.csv")

EXTRA_COLS = ['Flow ID', 'Src IP', 'Src Port', 'Dst IP']
CHUNK_SIZE = 100_000  # smaller chunks = lower peak memory
OUTPUT_PATH = f"{FOLDER}\\combined_raw.parquet"

writer = None
total_rows = 0

for f in files:
    print(f"Loading: {f}")
    file_rows = 0

    for chunk in pd.read_csv(f, chunksize=CHUNK_SIZE, low_memory=False):
        chunk.columns = chunk.columns.str.strip()

        cols_to_drop = [c for c in EXTRA_COLS if c in chunk.columns]
        if cols_to_drop:
            chunk = chunk.drop(columns=cols_to_drop)

        if 'Label' in chunk.columns:
            chunk = chunk[chunk['Label'] != 'Label']

        if chunk.empty:
            continue

        # Downcast numeric columns to shrink memory footprint
        for col in chunk.select_dtypes(include=['float64']).columns:
            chunk[col] = pd.to_numeric(chunk[col], downcast='float')
        for col in chunk.select_dtypes(include=['int64']).columns:
            chunk[col] = pd.to_numeric(chunk[col], downcast='integer')

        table = pa.Table.from_pandas(chunk, preserve_index=False)

        if writer is None:
            writer = pq.ParquetWriter(OUTPUT_PATH, table.schema)

        # Align schema across files/chunks (columns may differ in order)
        table = table.cast(writer.schema)
        writer.write_table(table)

        file_rows += len(chunk)

    print(f"  Rows written: {file_rows}")
    total_rows += file_rows

if writer is not None:
    writer.close()

print(f"\nTotal rows written: {total_rows}")
print(f"Saved combined raw data to: {OUTPUT_PATH}")