"""
Step 3 (v2, faster): Fix Data Types + Deduplicate
Uses pandas.read_parquet with column downcast, then a single vectorized
drop_duplicates() call -- much faster than manual cross-batch hashing.
Reads the whole file at once; if this hits a MemoryError, tell Claude
and it'll switch to a disk-sort-based approach instead.
"""

import pandas as pd

FOLDER = r"F:\all_10_csv"
INPUT_PATH = f"{FOLDER}\\combined_no_na.parquet"
OUTPUT_PATH = f"{FOLDER}\\combined_deduped.parquet"

print("RUNNING SCRIPT VERSION: v2-fast-dedup")

print("Loading parquet (single read)...")
df = pd.read_parquet(INPUT_PATH)
print(f"Loaded shape: {df.shape}")

# Downcast floats to save memory before dedup
for col in df.select_dtypes(include=['float64']).columns:
    df[col] = pd.to_numeric(df[col], downcast='float')

print("Deduplicating...")
before = len(df)
df = df.drop_duplicates()
after = len(df)

print(f"Rows before: {before}")
print(f"Rows after: {after}")
print(f"Duplicates removed: {before - after}")

print("Saving...")
df.to_parquet(OUTPUT_PATH, index=False)
print(f"Saved to: {OUTPUT_PATH}")